"""
Complex Swin Transformer.

This file mirrors `Models/swin_transformer.py` but supports complex-valued inputs
and produces an SSR-ready complex token sequence with a learned CLS token.
"""

import math
from functools import partial
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def c_stochastic_depth(input: Tensor, p: float, mode: str, training: bool = True) -> Tensor:
    # Same as in `Models/swin_transformer.py`; works for complex tensors too.
    if p < 0.0 or p > 1.0:
        raise ValueError(f"drop probability has to be between 0 and 1, but got {p}")
    if mode not in ["batch", "row"]:
        raise ValueError(f"mode has to be either 'batch' or 'row', but got {mode}")
    if not training or p == 0.0:
        return input

    survival_rate = 1.0 - p
    if mode == "row":
        size = [input.shape[0]] + [1] * (input.ndim - 1)
    else:
        size = [1] * input.ndim
    noise = torch.empty(size, dtype=input.real.dtype, device=input.device).bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    return input * noise


torch.fx.wrap("c_stochastic_depth")


class c_StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, input: Tensor) -> Tensor:
        return c_stochastic_depth(input, self.p, self.mode, self.training)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p}, mode={self.mode})"


class c_Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)


class c_ComplexLayerNorm(nn.Module):
    """LayerNorm applied separately to real and imaginary parts."""

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.real_ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
        self.imag_ln = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: Tensor) -> Tensor:
        return self.real_ln(x.real) + 1j * self.imag_ln(x.imag)


class c_ComplexGELU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x.real) + 1j * F.gelu(x.imag)


class c_ComplexDropout(nn.Module):
    def __init__(self, p: float = 0.0, inplace: Optional[bool] = None):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0.0:
            return x
        if not self.training:
            return x
        real = F.dropout(x.real, p=self.p, training=True, inplace=self.inplace if self.inplace is not None else False)
        imag = F.dropout(x.imag, p=self.p, training=True, inplace=self.inplace if self.inplace is not None else False)
        return torch.complex(real, imag)


class c_MLP(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        activation_layer: Callable[..., nn.Module] = c_ComplexGELU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        params = {} if inplace is None else {"inplace": inplace}

        layers: List[nn.Module] = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias, dtype=torch.cfloat))
            layers.append(activation_layer(**params))
            layers.append(c_ComplexDropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_channels[-1], bias=bias, dtype=torch.cfloat))
        layers.append(c_ComplexDropout(dropout, **params))
        super().__init__(*layers)


def _patch_merging_pad(x: Tensor) -> Tensor:
    H, W, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    x0 = x[..., 0::2, 0::2, :]
    x1 = x[..., 1::2, 0::2, :]
    x2 = x[..., 0::2, 1::2, :]
    x3 = x[..., 1::2, 1::2, :]
    return torch.cat([x0, x1, x2, x3], -1)


torch.fx.wrap("_patch_merging_pad")


def _get_relative_position_bias(
    relative_position_bias_table: Tensor, relative_position_index: Tensor, window_size: List[int]
) -> Tensor:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    return relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)


torch.fx.wrap("_get_relative_position_bias")


class c_PatchMerging(nn.Module):
    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = c_ComplexLayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False, dtype=torch.cfloat)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor) -> Tensor:
        x = _patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class c_PatchMergingV2(nn.Module):
    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = c_ComplexLayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False, dtype=torch.cfloat)
        self.norm = norm_layer(2 * dim)

    def forward(self, x: Tensor) -> Tensor:
        x = _patch_merging_pad(x)
        x = self.reduction(x)
        x = self.norm(x)
        return x


def c__complex_normalize(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    # Normalize by complex magnitude.
    denom = torch.linalg.vector_norm(x, dim=dim, keepdim=True).clamp_min(eps)
    return x / denom


def c_shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[Tensor] = None,
    training: bool = True,
) -> Tensor:
    B, H, W, C = input.shape
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)

    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()

    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]  # complex

    if logit_scale is not None:
        # Cosine attention on complex vectors -> real similarity.
        qn = c__complex_normalize(q, dim=-1)
        kn = c__complex_normalize(k, dim=-1)
        attn = (qn * kn.conj()).sum(dim=-1).real  # [B*nW, heads, N, N] after broadcasting below
        attn = attn.unsqueeze(-2)  # [B*nW, heads, N, 1] ??? not right; handle via matmul form
        # Use matmul form for correct NxN:
        attn = (qn @ kn.transpose(-2, -1).conj()).real
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = (q @ k.transpose(-2, -1).conj()).real  # real scores

    attn = attn + relative_position_bias.to(dtype=attn.dtype, device=attn.device)

    if sum(shift_size) > 0:
        attn_mask = torch.zeros((pad_H, pad_W), device=attn.device, dtype=attn.dtype)
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    # attn is real, v is complex -> complex output
    out = attn.to(dtype=v.dtype) @ v
    out = out.transpose(1, 2).reshape(x.size(0), x.size(1), C)
    out = F.linear(out, proj_weight, proj_bias)
    # apply dropout separately to real/imag
    if dropout > 0.0:
        out = torch.complex(
            F.dropout(out.real, p=dropout, training=training),
            F.dropout(out.imag, p=dropout, training=training),
        )

    out = out.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    out = out.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    if sum(shift_size) > 0:
        out = torch.roll(out, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    out = out[:, :H, :W, :].contiguous()
    return out


torch.fx.wrap("c_shifted_window_attention")


class c_ShiftedWindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias, dtype=torch.cfloat)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, dtype=torch.cfloat)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
        )

    def forward(self, x: Tensor) -> Tensor:
        relative_position_bias = self.get_relative_position_bias()
        return c_shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
        )


class c_ShiftedWindowAttentionV2(c_ShiftedWindowAttention):
    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__(
            dim,
            window_size,
            shift_size,
            num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False))
        if qkv_bias and self.qkv.bias is not None:
            length = self.qkv.bias.numel() // 3
            self.qkv.bias[length : 2 * length].data.zero_()

    def define_relative_position_bias_table(self):
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing="ij"))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)

        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0
        self.register_buffer("relative_coords_table", relative_coords_table)

    def get_relative_position_bias(self) -> Tensor:
        relative_position_bias = _get_relative_position_bias(
            self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads),
            self.relative_position_index,  # type: ignore[arg-type]
            self.window_size,
        )
        return 16 * torch.sigmoid(relative_position_bias)

    def forward(self, x: Tensor) -> Tensor:
        relative_position_bias = self.get_relative_position_bias()
        return c_shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            logit_scale=self.logit_scale,
            training=self.training,
        )


class c_SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = c_ComplexLayerNorm,
        attn_layer: Callable[..., nn.Module] = c_ShiftedWindowAttention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = c_StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = c_MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=c_ComplexGELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.real)
                nn.init.xavier_uniform_(m.weight.imag)
                if m.bias is not None:
                    nn.init.normal_(m.bias.real, std=1e-6)
                    nn.init.normal_(m.bias.imag, std=1e-6)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.stochastic_depth(self.attn(self.norm1(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class c_SwinTransformerBlockV2(c_SwinTransformerBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = c_ComplexLayerNorm,
        attn_layer: Callable[..., nn.Module] = c_ShiftedWindowAttentionV2,
    ):
        super().__init__(
            dim,
            num_heads,
            window_size,
            shift_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            norm_layer=norm_layer,
            attn_layer=attn_layer,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x + self.stochastic_depth(self.norm2(self.mlp(x)))
        return x


class c_SwinTransformer(nn.Module):
    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        in_channels: int = 3,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = c_PatchMerging,
    ):
        super().__init__()
        self.num_classes = num_classes

        if block is None:
            block = c_SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(c_ComplexLayerNorm, eps=1e-5)

        layers: List[nn.Module] = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    embed_dim,
                    kernel_size=(patch_size[0], patch_size[1]),
                    stride=(patch_size[0], patch_size[1]),
                    dtype=torch.cfloat,
                ),
                c_Permute([0, 2, 3, 1]),
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1) if total_stage_blocks > 1 else 0.0
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))

        self.features = nn.Sequential(*layers)

        self.num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(self.num_features)
        self.cls_token = nn.Parameter(torch.zeros((1, 1, self.num_features), dtype=torch.cfloat))
        self.head = nn.Linear(self.num_features, num_classes, dtype=torch.cfloat)

        # Init: mirror real Swin init style for complex weights.
        nn.init.trunc_normal_(self.cls_token.real, std=0.02)
        nn.init.trunc_normal_(self.cls_token.imag, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.real, std=0.02)
                nn.init.trunc_normal_(m.weight.imag, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.real)
                    nn.init.zeros_(m.bias.imag)

    def forward(self, x: Tensor):
        # x: (B, C, H, W), dtype complex
        x = self.features(x)  # (B, H', W', C'), complex channels-last
        x = self.norm(x)      # (B, H', W', C')

        B, H, W, C = x.shape
        tokens = x.view(B, H * W, C)
        # padding the fake cls is only for supporting the SSR loss calculation
        cls = self.cls_token.expand(B, -1, -1)
        feature_tokens = torch.cat([cls, tokens], dim=1)  # (B, 1+H*W, C) complex

        pooled = tokens.mean(dim=1)  # (B, C) complex; no CLS pooling for Swin by default
        complex_logits = self.head(pooled)  # (B, num_classes) complex
        logits = complex_logits.abs()       # real logits for CE
        return logits, feature_tokens


def _swin_transformer(
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    stochastic_depth_prob: float,
    **kwargs: Any,
) -> c_SwinTransformer:
    return c_SwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )


complex_swin_t = partial(
    _swin_transformer,
    patch_size=[4, 4],
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=[7, 7],
    stochastic_depth_prob=0.2,
)

complex_swin_s = partial(
    _swin_transformer,
    patch_size=[4, 4],
    embed_dim=96,
    depths=[2, 2, 18, 2],
    num_heads=[3, 6, 12, 24],
    window_size=[7, 7],
    stochastic_depth_prob=0.3,
)

complex_swin_b = partial(
    _swin_transformer,
    patch_size=[4, 4],
    embed_dim=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=[7, 7],
    stochastic_depth_prob=0.5,
)

complex_swin_v2_t = partial(
    _swin_transformer,
    patch_size=[4, 4],
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=[8, 8],
    stochastic_depth_prob=0.2,
    block=c_SwinTransformerBlockV2,
    downsample_layer=c_PatchMergingV2,
)

complex_swin_v2_s = partial(
    _swin_transformer,
    patch_size=[4, 4],
    embed_dim=96,
    depths=[2, 2, 18, 2],
    num_heads=[3, 6, 12, 24],
    window_size=[8, 8],
    stochastic_depth_prob=0.3,
    block=c_SwinTransformerBlockV2,
    downsample_layer=c_PatchMergingV2,
)

complex_swin_v2_b = partial(
    _swin_transformer,
    patch_size=[4, 4],
    embed_dim=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=[8, 8],
    stochastic_depth_prob=0.5,
    block=c_SwinTransformerBlockV2,
    downsample_layer=c_PatchMergingV2,
)

# Backwards-compatible aliases (so external imports keep working).
stochastic_depth = c_stochastic_depth
StochasticDepth = c_StochasticDepth
Permute = c_Permute
ComplexLayerNorm = c_ComplexLayerNorm
ComplexGELU = c_ComplexGELU
ComplexDropout = c_ComplexDropout
MLP = c_MLP
PatchMerging = c_PatchMerging
PatchMergingV2 = c_PatchMergingV2
shifted_window_attention = c_shifted_window_attention
ShiftedWindowAttention = c_ShiftedWindowAttention
ShiftedWindowAttentionV2 = c_ShiftedWindowAttentionV2
SwinTransformerBlock = c_SwinTransformerBlock
SwinTransformerBlockV2 = c_SwinTransformerBlockV2
SwinTransformer = c_SwinTransformer

