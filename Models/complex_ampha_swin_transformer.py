"""
Amplitude-Phase Swin Transformer (real-valued backbone, complex-aware input).

Motivation
----------
The existing complex Swin (`Models/complex_swin_transformer.py`) performs many operations
separately on real/imag parts. That can unintentionally distort coupled amplitude/phase
structure. This implementation instead converts complex inputs into an amplitude + phase
representation and processes it with a standard (real) Swin Transformer.

Key idea
--------
Given a complex tensor x, we build real channels:
  - amplitude: |x|
  - phase (wrapped): angle(x), encoded stably as sin(angle(x)) and cos(angle(x))
So the Swin sees 3 * in_channels channels: [amp, sin(phase), cos(phase)].

This file returns (logits, feature_tokens) to match the output contract used by the
training pipeline for complex Swin models (SSR regularizer expects feature tokens).
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, List, Optional

import torch
from torch import Tensor, nn

from .swin_transformer import (
    PatchMerging,
    PatchMergingV2,
    Permute,
    SwinTransformerBlock,
    SwinTransformerBlockV2,
)


def _complex_to_amp_phase_channels(
    x: Tensor,
    *,
    amp_log: bool = False,
    amp_eps: float = 1e-6,
    phase_eps: float = 0.0,
) -> Tensor:
    """
    Convert complex BCHW tensor into real BCHW tensor with channels:
      [amplitude, sin(phase), cos(phase)] concatenated along C.

    Args:
        x: Complex tensor of shape (B, C, H, W). If x is real, it's treated as having 0 imag.
        amp_log: If True, use log1p(|x|) to compress dynamic range.
        amp_eps: Clamp for amplitude to avoid extreme gradients around zero.
        phase_eps: Optional small value to add inside atan2 via torch.angle stabilization.
                   Kept for experimentation; default 0.0.
    """
    if not torch.is_complex(x):
        x = torch.complex(x, torch.zeros_like(x))

    amp = x.abs().clamp_min(amp_eps)
    if amp_log:
        amp = torch.log1p(amp)

    # torch.angle is stable and handles zero-magnitude, but phase is still wrapped.
    if phase_eps != 0.0:
        phase = torch.atan2(x.imag + phase_eps, x.real + phase_eps)
    else:
        phase = torch.angle(x)

    sinp = torch.sin(phase)
    cosp = torch.cos(phase)
    return torch.cat([amp, sinp, cosp], dim=1)  # (B, 3C, H, W) real


class AmphaSwinTransformer(nn.Module):
    """
    Swin Transformer that consumes complex input via amplitude/phase encoding.

    Input:
      x: (B, C, H, W) complex tensor
    Output:
      logits: (B, num_classes) float tensor
      feature_tokens: (B, 1 + H'*W', D) complex tensor (for SSR / analysis)
    """

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
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        *,
        amp_log: bool = False,
        amp_eps: float = 1e-6,
        phase_eps: float = 0.0,
        feature_complex: bool = True,
    ) -> None:
        super().__init__()

        if block is None:
            block = SwinTransformerBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.amp_log = amp_log
        self.amp_eps = amp_eps
        self.phase_eps = phase_eps
        self.feature_complex = feature_complex

        # Patch embedding: 3 * in_channels real channels (amp, sinp, cosp)
        layers: List[nn.Module] = []
        layers.append(
            nn.Sequential(
                nn.Conv2d(
                    3 * in_channels,
                    embed_dim,
                    kernel_size=(patch_size[0], patch_size[1]),
                    stride=(patch_size[0], patch_size[1]),
                ),
                Permute([0, 2, 3, 1]),  # B,C,H,W -> B,H,W,C
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                sd_prob = (
                    stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                    if total_stage_blocks > 1
                    else 0.0
                )
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

        # SSR-ready tokens: prepend CLS (fake CLS to match other complex models).
        self.cls_token = nn.Parameter(torch.zeros((1, 1, self.num_features)))

        # Classification head is real-valued.
        self.head = nn.Linear(self.num_features, num_classes)

        # Optional complex token projection (so downstream can treat feature_tokens as complex).
        # We keep the complex dimension equal to num_features.
        if feature_complex:
            self.to_complex = nn.Linear(self.num_features, 2 * self.num_features)
        else:
            self.to_complex = None

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor):
        # x: (B, C, H, W) complex
        x_real = _complex_to_amp_phase_channels(
            x, amp_log=self.amp_log, amp_eps=self.amp_eps, phase_eps=self.phase_eps
        )  # (B, 3C, H, W) float

        y = self.features(x_real)  # (B, H', W', D), channels-last
        y = self.norm(y)  # (B, H', W', D)

        B, H, W, D = y.shape
        tokens = y.view(B, H * W, D)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        feature_tokens_real = torch.cat([cls, tokens], dim=1)  # (B, 1+N, D)

        pooled = tokens.mean(dim=1)  # (B, D)
        logits = self.head(pooled)  # (B, num_classes)

        if self.to_complex is None:
            # Keep backward compatibility with callers that only consume feature tokens.
            # Represent "complex" as real dtype if requested.
            feature_tokens = feature_tokens_real
        else:
            z = self.to_complex(feature_tokens_real)  # (B, 1+N, 2D)
            real, imag = z.chunk(2, dim=-1)
            feature_tokens = torch.complex(real, imag)  # (B, 1+N, D) complex

        return logits, feature_tokens


def _ampha_swin_transformer(
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    stochastic_depth_prob: float,
    **kwargs: Any,
) -> AmphaSwinTransformer:
    return AmphaSwinTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )


# Factory functions aligned with the existing naming scheme.
complex_ampha_swin_t = partial(
    _ampha_swin_transformer,
    patch_size=[4, 4],
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=[7, 7],
    stochastic_depth_prob=0.2,
    block=SwinTransformerBlock,
    downsample_layer=PatchMerging,
)

complex_ampha_swin_s = partial(
    _ampha_swin_transformer,
    patch_size=[4, 4],
    embed_dim=96,
    depths=[2, 2, 18, 2],
    num_heads=[3, 6, 12, 24],
    window_size=[7, 7],
    stochastic_depth_prob=0.3,
    block=SwinTransformerBlock,
    downsample_layer=PatchMerging,
)

complex_ampha_swin_b = partial(
    _ampha_swin_transformer,
    patch_size=[4, 4],
    embed_dim=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=[7, 7],
    stochastic_depth_prob=0.5,
    block=SwinTransformerBlock,
    downsample_layer=PatchMerging,
)

complex_ampha_swin_v2_t = partial(
    _ampha_swin_transformer,
    patch_size=[4, 4],
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=[8, 8],
    stochastic_depth_prob=0.2,
    block=SwinTransformerBlockV2,
    downsample_layer=PatchMergingV2,
)

complex_ampha_swin_v2_s = partial(
    _ampha_swin_transformer,
    patch_size=[4, 4],
    embed_dim=96,
    depths=[2, 2, 18, 2],
    num_heads=[3, 6, 12, 24],
    window_size=[8, 8],
    stochastic_depth_prob=0.3,
    block=SwinTransformerBlockV2,
    downsample_layer=PatchMergingV2,
)

complex_ampha_swin_v2_b = partial(
    _ampha_swin_transformer,
    patch_size=[4, 4],
    embed_dim=128,
    depths=[2, 2, 18, 2],
    num_heads=[4, 8, 16, 32],
    window_size=[8, 8],
    stochastic_depth_prob=0.5,
    block=SwinTransformerBlockV2,
    downsample_layer=PatchMergingV2,
)

