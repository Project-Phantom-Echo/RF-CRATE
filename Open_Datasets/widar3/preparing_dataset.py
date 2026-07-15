import os
from CSIKit.reader import IWLBeamformReader
import pickle
from tqdm import tqdm
import time
import itertools
from typing import Tuple
import numpy as np
import pandas as pd
from multiprocessing import Pool

# Loading all the csi files and calculating the time length of each csi file: time_lengths.npy


def get_CSI(csi_data: 'CSIData', squeeze_output: bool = False) -> Tuple[np.array, int, int]:
    frames = csi_data.frames
    try:
        csi_shape = frames[0].csi_matrix.shape
    except:
        return (0,0,0)

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
    if squeeze_output:
        csi = np.squeeze(csi)
    return (csi, no_frames, no_subcarriers)


class loading_csi_data:
    def __init__(self, file_name_list):
        self.file_name = file_name_list
        self.my_reader = IWLBeamformReader()
        print("The number of instances: ", len(file_name_list))


    def load_csi_data(self, indexes):
        start_index, end_index = indexes
        loading_files = self.file_name[start_index:end_index]
        csi_data_list = []
        csi_data_len_list = []
        for file_name in loading_files:
            csi_data = self.my_reader.read_file(file_name)
            csi_matrix, num_frames, _ = get_CSI(csi_data)
            if num_frames <=0:
                continue
            # csi_data_list.append(csi_matrix)
            csi_data_len_list.append(csi_matrix.shape[0])
        return csi_data_len_list
    
    def get_len(self):
        return len(self.file_name)
        


if __name__ == "__main__":
    file_names = []

    for root, dirs, files in os.walk('CSI/'):
        for file in files:
            if file.endswith('.dat'):
                file_path = os.path.join(root, file)
                file_names.append(file_path)

    print('len(CSI_file_list): ', len(file_names))
    print(file_names[0])
    
    csi_loader = loading_csi_data(file_names)
    num_files = csi_loader.get_len()
    
    # split the data into 100 parts
    num_processes = 100
    b = []
    
    for i in range(num_processes):
        start_index = i * num_files // num_processes
        end_index = (i + 1) * num_files // num_processes
        b.append((start_index, end_index))
    
    start_time = time.time()
    time_lengths = []
    with Pool(num_processes) as p:
        csi_data_len_list = p.map(csi_loader.load_csi_data, b)
        time_lengths.extend(csi_data_len_list)
    
    print("Time taken: ", time.time() - start_time)
     # save the time_lengths as numpy array
    np.save("time_lengths.npy", np.array(time_lengths))  # Time taken:  4780.858397245407



