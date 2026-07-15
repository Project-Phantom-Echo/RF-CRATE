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
    # receive a argument index 
    folder_index = int(sys.argv[1])
    print('Folder index:', folder_index)
    CSI_folders = os.listdir('CSI')
    print('Processing folder:', CSI_folders[folder_index])
    # make a subfolder in the CSI_Processed folder
    if os.path.exists('Timestamp/' + CSI_folders[folder_index]):
        print('Folder already exists')
    else:
        os.makedirs('Timestamp/' + CSI_folders[folder_index])

        
    sub_folders = os.listdir('CSI/' + CSI_folders[folder_index])
    print('Number of subfolders:', len(sub_folders))
    
    my_reader = IWLBeamformReader()
    
    for sub_folder_index in range(len(sub_folders)):
        success_count = 0
        failure_count = 0
        print('Processing subfolder:', sub_folders[sub_folder_index])
        if os.path.exists('Timestamp/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index]):
            print('Subfolder already exists')
        else:
            os.makedirs('Timestamp/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index])
            
        files = os.listdir('CSI/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index])
        for file_index in tqdm(range(len(files))):
            # if the file is already processed, skip it
            if os.path.exists('Timestamp/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index] + '/' + files[file_index][:-3] + 'npy'):
                print('File already exists, skipping:', files[file_index])
                continue
            csi_data = my_reader.read_file('CSI/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index] + '/' + files[file_index])
            csi_matrix, _, _, timestamps = get_CSI(csi_data)  # shape: (frames, subcarriers, Rx, Tx)
            if csi_matrix is None:
                failure_count += 1
                continue
            num_frames, num_subcarriers, num_rx, num_tx = csi_matrix.shape
            if num_frames < 1000 or num_frames > 2560:  # filter out the files with less than 1s or more than 2.56s
                # print('Error in file:', files[file_index])
                failure_count += 1
                continue
            
            np.save('Timestamp/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index] + '/' + files[file_index][:-3] + 'npy', timestamps)
            print('Subfolder done')
        # add log to the log file with name: 'timestamp_log.txt'
        with open('timestamp_log.txt', 'a') as f:
            f.write('Timestamp/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index] + '\n')
            f.write('Success: ' + str(success_count) + '\n')
            f.write('Failure: ' + str(failure_count) + '\n')
            f.write('\n')     
    