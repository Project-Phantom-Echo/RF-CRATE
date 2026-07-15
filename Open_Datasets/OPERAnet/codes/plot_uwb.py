import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.fftpack import fft

# Load data from CSV files
uwb1 = pd.read_csv('../uwb1/uwb1_exp018.csv')
uwb2 = pd.read_csv('../uwb2/uwb2_exp018.csv')

# Extract necessary columns
tx_id_uwb1 = uwb1['tx_id']
rx_id_uwb1 = uwb1['rx_id']
tx_id_uwb2 = uwb2['tx_id']
rx_id_uwb2 = uwb2['rx_id']

# Filter bidirectional data between nodes 0 and 3 (UWB1)
idx_1 = uwb1[(tx_id_uwb1 == 0) & (rx_id_uwb1 == 3) | (tx_id_uwb1 == 3) & (rx_id_uwb1 == 0)].index
Filtered_data1 = uwb1.loc[idx_1]

# Filter bidirectional data between nodes 1 and 2 (UWB2)
idx_2 = uwb2[(tx_id_uwb2 == 1) & (rx_id_uwb2 == 2) | (tx_id_uwb2 == 2) & (rx_id_uwb2 == 1)].index
Filtered_data2 = uwb2.loc[idx_2]

# Convert timestamps to datetime objects
date1 = pd.to_datetime(Filtered_data1['timestamp'], format='%H:%M:%S.%f')
date2 = pd.to_datetime(Filtered_data2['timestamp'], format='%H:%M:%S.%f')

# Convert CIR data to CFR using FFT
CFR_1 = fft(Filtered_data1.iloc[:, 29:64].values, axis=1)
CFR_2 = fft(Filtered_data2.iloc[:, 29:79].values, axis=1)

# Define start and end times
start_time = datetime.strptime('18:04:44.553', '%H:%M:%S.%f')
end_time = datetime.strptime('18:08:11.553', '%H:%M:%S.%f')

# Find indices for the start and end times
idx1 = np.argmin([abs((start_time - t).total_seconds()) for t in date1])
idx11 = np.argmin([abs((end_time - t).total_seconds()) for t in date1])
idx2 = np.argmin([abs((start_time - t).total_seconds()) for t in date2])
idx22 = np.argmin([abs((end_time - t).total_seconds()) for t in date2])

# Calculate time durations
t_uwb1 = date1[idx1:idx11+1]
out = np.diff([t.timestamp() for t in t_uwb1])
actualtime1 = np.insert(out, 0, 0)
time_duration = np.cumsum(actualtime1)

t_uwb2 = date2[idx2:idx22+1]
out2 = np.diff([t.timestamp() for t in t_uwb2])
actualtime2 = np.insert(out2, 0, 0)
time_duration2 = np.cumsum(actualtime2)

# Plot raw un-filtered data
plt.figure()
plt.plot(time_duration, np.abs(CFR_1[idx1:idx11+1, 9]), label='UWB system 1')
plt.plot(time_duration2, np.abs(CFR_2[idx2:idx22+1, 9]), label='UWB system 2')
plt.legend()
plt.xlabel('Duration (s)')
plt.ylabel('CFR Amplitude')
plt.show()

# Plot aligned CIR measurements
len1 = 35
len2 = 50
first_path_offset = -2
CIR_samp_period = 1 / (2 * (499.2e6))
SpeedofLight = 299792458
CIR_prop_time_axis1 = np.arange(first_path_offset, first_path_offset + len1 * CIR_samp_period * 1e9, CIR_samp_period * 1e9)
CIR_prop_time_axis2 = np.arange(first_path_offset, first_path_offset + len2 * CIR_samp_period * 1e9, CIR_samp_period * 1e9)

CIR1 = Filtered_data1.iloc[:, -len1:].values
CIR2 = Filtered_data2.iloc[:, -len2:].values
FP_idx1 = Filtered_data1['fp_index']
FP_idx2 = Filtered_data2['fp_index']
fp1 = np.mod(np.abs(FP_idx1), 1)
fp2 = np.mod(np.abs(FP_idx2), 1)
time_Offset1 = fp1 * CIR_samp_period * 1e9
time_tOffset2 = fp2 * CIR_samp_period * 1e9
CIR_mag1 = np.abs(CIR1)
CIR_mag2 = np.abs(CIR2)

# Plot bidirectional CIR data between a given pair of nodes
start_idx = 0
end_idx = 1000

plt.figure()
plt.plot((CIR_prop_time_axis1 - time_Offset1[start_idx:end_idx]).T, (CIR_mag1[start_idx:end_idx, :]).T, '.')
plt.xlabel(r'$\tau - \tau_{FP}$ (ns)')
plt.ylabel('CIR magnitude')
plt.show()

plt.figure()
plt.plot((CIR_prop_time_axis2 - time_tOffset2[start_idx:end_idx]).T, (CIR_mag2[start_idx:end_idx, :]).T, '.')
plt.xlabel(r'$\tau - \tau_{FP}$ (ns)')
plt.ylabel('CIR magnitude')
plt.show()
