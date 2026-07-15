import numpy as np
import itertools
from typing import Tuple
from scipy.signal import butter, filtfilt
from scipy import signal
import torch
import os
import sys
from tqdm import tqdm
from CSIKit.reader import IWLBeamformReader

def get_CSI(csi_data: 'CSIData', squeeze_output: bool = False) -> Tuple[np.array, int, int]:
    frames = csi_data.frames
    try:
        csi_shape = frames[0].csi_matrix.shape
    except:
        return (None,0,0)

    no_frames = len(frames)
    no_subcarriers = csi_shape[0]

    # Matrices should be Frames * Subcarriers * Rx * Tx.
    # Single Rx/Tx streams should be squeezed.
    if len(csi_shape) == 3:
        # Intel data comes as Subcarriers * Rx * Tx.
        no_rx_antennas = csi_shape[1]
        no_tx_antennas = csi_shape[2]
    elif len(csi_shape) == 2 or len(csi_shape) == 1:
        # Single antenna stream.
        no_rx_antennas = 1
        no_tx_antennas = 1
    else:
        # Error. Unknown CSI shape.
        print("Error: Unknown CSI shape.")

    csi = np.zeros((no_frames, no_subcarriers, no_rx_antennas, no_tx_antennas), dtype=complex)
    ranges = itertools.product(*[range(n) for n in [no_frames, no_subcarriers, no_rx_antennas, no_tx_antennas]])
    is_single_antenna = no_rx_antennas == 1 and no_tx_antennas == 1

    drop_indices = []
    for frame, subcarrier, rx_antenna_index, tx_antenna_index in ranges:
        frame_data = frames[frame].csi_matrix
        if subcarrier >= frame_data.shape[0]:
            # Inhomogenous component
            # Skip frame for now. Need a better method soon.
            continue

        subcarrier_data = frame_data[subcarrier]
        if subcarrier_data.shape != (no_rx_antennas, no_tx_antennas) and not is_single_antenna:
            if rx_antenna_index >= subcarrier_data.shape[0] or tx_antenna_index >= subcarrier_data.shape[1]:
                # Inhomogenous component
                # Skip frame for now. Need a better method soon.
                drop_indices.append(frame)
                continue
        csi[frame][subcarrier][rx_antenna_index][tx_antenna_index] = subcarrier_data if is_single_antenna else \
            subcarrier_data[rx_antenna_index][tx_antenna_index]  
    csi = np.delete(csi, drop_indices, 0)
    csi_data.timestamps = [x for i, x in enumerate(csi_data.timestamps) if i not in drop_indices]
    timestamps = np.array(csi_data.timestamps)
    if squeeze_output:
        csi = np.squeeze(csi)
    return (csi, no_frames, no_subcarriers, timestamps)



if __name__ == '__main__':
    import os
from pathlib import Path

CSI_ROOT = Path("CSI/")
OUT_CSI_ROOT = Path("Timestamp")
ALLOW_EXT = {".dat"}  # 按需增减

my_reader = IWLBeamformReader()

success_count = 0
failure_count = 0
skipped_count = 0

# 递归遍历 CSI_ROOT 下的所有文件
for dirpath, dirnames, filenames in os.walk(CSI_ROOT):
    dirpath = Path(dirpath)

    # 只处理允许的扩展名
    data_files = [f for f in filenames if Path(f).suffix.lower() in ALLOW_EXT]
    if not data_files:
        continue

    # 计算相对 CSI_ROOT 的层级路径，用于镜像输出目录
    rel_dir = dirpath.relative_to(CSI_ROOT)

    # 确保对应输出目录存在（完整镜像层级，无论是 1/2/3/N 层都适配）
    out_csi_dir = OUT_CSI_ROOT / rel_dir
    out_csi_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing dir: {rel_dir} | {len(data_files)} files")

    for name in tqdm(data_files):
        in_path = dirpath / name
        stem = Path(name).stem
        out_csi_path = out_csi_dir / f"{stem}.npy"

        # 已处理则跳过
        if out_csi_path.exists():
            # print("File already exists, skipping:", name)
            skipped_count += 1
            continue

        try:
            csi_data = my_reader.read_file(str(in_path))
            csi_matrix, _, _, timestamps = get_CSI(csi_data)  # (frames, subcarriers, Rx, Tx)
            if csi_matrix is None or csi_matrix.ndim != 4:
                failure_count += 1
                continue

            num_frames = csi_matrix.shape[0]
            # if num_frames < 1000 or num_frames > 2560:
            #     failure_count += 1
            #     continue

            # 保存
            np.save(out_csi_path, timestamps)
            # np.save(out_dfs_path, freq_time_prof)
            success_count += 1

        except Exception as e:
            # print("Error in file:", name, e)
            failure_count += 1
            continue