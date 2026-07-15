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
    if squeeze_output:
        csi = np.squeeze(csi)
    return (csi, no_frames, no_subcarriers)


def preprocess_CSI(csi_matrix: np.array, csi_sample_rate: int = 1000):
    '''ArithmeticError
    Preprocesses the CSI matrix by applying phase cleaning (Widar3.0) and a lowpass and highpass filter.
    Then, calculates the motion statistics.
    The shape of the CSI matrix (complex-valued) should be Frames (time dimension) * Subcarriers * Rx * Tx.
    Returns the processed CSI matrix and the motion statistics.
    '''
    # define the lowpass and highpass filter
    half_rate = csi_sample_rate / 2
    uppe_orde = 6
    uppe_stop = 60
    if uppe_stop > half_rate:
        uppe_stop = half_rate - 1
    lowe_orde = 3
    lowe_stop = 2
    lu, ld = butter(uppe_orde, uppe_stop / half_rate, 'low')  
    hu, hd = butter(lowe_orde, lowe_stop / half_rate, 'high')
    num_frames, num_subcarriers, num_rx, num_tx = csi_matrix.shape
    
    # apply the l2 norm on the subcarrier dim for eliminating the impact of AGC (automatic gain control)
    csi_norm = np.linalg.norm(csi_matrix, ord=1, axis=1)
    csi_matrix = csi_matrix / csi_norm[:, np.newaxis, :, :]

    # phase cleaning
    if num_rx == 1:
        cleaned_csi = csi_matrix
    elif num_rx == 2:
        cleaned_csi = np.zeros((num_frames, num_subcarriers, 1, num_tx), dtype=complex)
        cleaned_csi = csi_matrix[:, :, 0, :] * np.conj(csi_matrix[:, :, 1, :])
    else:
        cleaned_csi = np.zeros((num_frames, num_subcarriers, num_rx, num_tx), dtype=complex)
        for i in range(num_rx):
            cleaned_csi[:, :, i, :] = csi_matrix[:, :, i, :] * np.conj(csi_matrix[:, :, (i + 1)%num_rx, :])

     # apply the lowpass filter
    for i in range(num_subcarriers):
        for j in range(num_rx):
            for k in range(num_tx):
                cleaned_csi[:, i, j, k] = filtfilt(lu, ld, cleaned_csi[:, i, j, k])
    # apply the highpass filter
    for i in range(num_subcarriers):
        for j in range(num_rx):
            for k in range(num_tx):
                cleaned_csi[:, i, j, k] = filtfilt(lu, ld, cleaned_csi[:, i, j, k])

    return cleaned_csi


def get_csi_dfs(csi_data, samp_rate = 1000, window_size = 256, window_step = 10):
    '''
    input csi_data: [Time_dim, num_subcarriers, Rx*Tx,] (complex numpy array)
    return dfs spetrum with shape: [Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2]
    '''
    
    half_rate = samp_rate / 2
    uppe_stop = 60
    freq_bins_unwrap = np.concatenate((np.arange(0, half_rate, 1) / samp_rate, np.arange(-half_rate, 0, 1) / samp_rate))
    freq_lpf_sele = np.logical_and(np.less_equal(freq_bins_unwrap,(uppe_stop / samp_rate)),np.greater_equal(freq_bins_unwrap,(-uppe_stop / samp_rate)))
    freq_lpf_positive_max = 60
    
    # DC removal
    csi_data = csi_data.numpy()
    csi_data = csi_data - np.mean(csi_data, axis=0)
    noverlap = window_size - window_step
    
    freq, ticks, freq_time_prof_allfreq = signal.stft(csi_data, 
                                                      fs=samp_rate, 
                                                      nfft=samp_rate,
                                                        window=('gaussian', 
                                                                window_size), 
                                                        nperseg=window_size, 
                                                        noverlap=noverlap, 
                                                        return_onesided=False,
                                                        padded=True, 
                                                        axis=0)
    
    freq_time_prof_allfreq = np.array(freq_time_prof_allfreq)
    freq_time_prof = freq_time_prof_allfreq[freq_lpf_sele, :]  # shape: [freq_bins, num_subcarriers, Rx*Tx, Time_bins]
    freq_time_prof = np.roll(freq_time_prof, freq_lpf_positive_max, axis=0) # shape: [freq_bins, num_subcarriers, Rx*Tx, Time_bins]
    freq_bin = np.array(freq)[freq_lpf_sele]
    freq_bin = np.roll(freq_bin, freq_lpf_positive_max, axis=0)
    # change freq_time_prof shape to: [Time_bins, freq_bins, num_subcarriers, Rx*Tx]
    freq_time_prof = np.transpose(freq_time_prof, (3, 0, 1, 2))
    freq_time_prof_real = freq_time_prof.real
    freq_time_prof_imag = freq_time_prof.imag
    # we get a tensor with shape: [Time_bins, freq_bins, num_subcarriers, Rx*Tx, 2]
    freq_time_prof = np.stack([freq_time_prof_real, freq_time_prof_imag], axis=-1)
    return freq_bin, ticks, torch.tensor(freq_time_prof)



if __name__ == '__main__':
    # receive a argument index 
    folder_index = int(sys.argv[1])
    print('Folder index:', folder_index)
    CSI_folders = os.listdir('CSI')
    print('Processing folder:', CSI_folders[folder_index])
    # make a subfolder in the CSI_Processed folder
    if os.path.exists('CSI_Processed/' + CSI_folders[folder_index]):
        print('Folder already exists')
    else:
        os.makedirs('CSI_Processed/' + CSI_folders[folder_index])
    # make forlder for the dfs data
    if os.path.exists('DFS/' + CSI_folders[folder_index]):
        print('Folder already exists')
    else:
        os.makedirs('DFS/' + CSI_folders[folder_index])
        
    sub_folders = os.listdir('CSI/' + CSI_folders[folder_index])
    print('Number of subfolders:', len(sub_folders))
    
    my_reader = IWLBeamformReader()
    
    for sub_folder_index in range(len(sub_folders)):
        success_count = 0
        failure_count = 0
        print('Processing subfolder:', sub_folders[sub_folder_index])
        if os.path.exists('CSI_Processed/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index]):
            print('Subfolder already exists')
        else:
            os.makedirs('CSI_Processed/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index])
        if os.path.exists('DFS/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index]):
            print('Subfolder already exists')
        else:
            os.makedirs('DFS/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index])
            
        files = os.listdir('CSI/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index])
        for file_index in tqdm(range(len(files))):
            # if the file is already processed, skip it
            if os.path.exists('CSI_Processed/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index] + '/' + files[file_index][:-3] + 'npy'):
                print('File already exists, skipping:', files[file_index])
                continue
            csi_data = my_reader.read_file('CSI/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index] + '/' + files[file_index])
            csi_matrix, _, _ = get_CSI(csi_data)  # shape: (frames, subcarriers, Rx, Tx)
            if csi_matrix is None:
                # print('Error in file:', files[file_index])
                failure_count += 1
                continue
            num_frames, num_subcarriers, num_rx, num_tx = csi_matrix.shape
            if num_frames < 1000 or num_frames > 2560:  # filter out the files with less than 1s or more than 2.56s
                # print('Error in file:', files[file_index])
                failure_count += 1
                continue
            
            try:
                csi_matrix = preprocess_CSI(csi_matrix)
                csi_matrix = csi_matrix.astype(np.complex64)
                csi = torch.tensor(csi_matrix)  # shape: (frames, subcarriers, Rx, Tx)
                csi = csi.view(csi.shape[0], csi.shape[1], csi.shape[2]*csi.shape[3])  # shape: (frames, subcarriers, Rx*Tx)
                freq_bin, ticks, freq_time_prof = get_csi_dfs(csi, samp_rate = 1000, window_size = 256, window_step = 10) 
                freq_time_prof = freq_time_prof.numpy()
                freq_time_prof = freq_time_prof.astype(np.float32)
                success_count += 1
            except:
                # print('Error in file:', files[file_index])
                failure_count += 1
                continue
            np.save('CSI_Processed/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index] + '/' + files[file_index][:-3] + 'npy', csi_matrix)
            np.save('DFS/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index] + '/' + files[file_index][:-3] + 'npy', freq_time_prof)
        print('Subfolder done')
        # add log to the log file with name: 'preprocessing_log.txt'
        with open('preprocessing_log.txt', 'a') as f:
            f.write('CSI_Processed/' + CSI_folders[folder_index] + '/' + sub_folders[sub_folder_index] + '\n')
            f.write('Success: ' + str(success_count) + '\n')
            f.write('Failure: ' + str(failure_count) + '\n')
            f.write('\n')     
    