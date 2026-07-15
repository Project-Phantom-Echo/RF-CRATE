import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load data from CSV files
uwb1 = pd.read_csv('../uwb1/uwb1_exp028.csv')
uwb2 = pd.read_csv('../uwb2/uwb2_exp028.csv')

# Convert timestamps to datetime objects
uwb1['timestamp'] = pd.to_datetime(uwb1['timestamp'], format='%H:%M:%S.%f')
uwb2['timestamp'] = pd.to_datetime(uwb2['timestamp'], format='%H:%M:%S.%f')

tx_id_uwb1 = uwb1['tx_id']
rx_id_uwb1 = uwb1['rx_id']
tx_id_uwb2 = uwb2['tx_id']
rx_id_uwb2 = uwb2['rx_id']

# Filter bidirectional data between nodes 0 and 3 (UWB1)
idx_1 = uwb1[(tx_id_uwb1 == 0) & (rx_id_uwb1 == 3) | (tx_id_uwb1 == 3) & (rx_id_uwb1 == 0)].index
Filtered_data1 = uwb1.loc[idx_1]

# Filter bidirectional data between nodes 3 and 4 (UWB2)
idx_2 = uwb2[(tx_id_uwb2 == 3) & (rx_id_uwb2 == 4) | (tx_id_uwb2 == 4) & (rx_id_uwb2 == 3)].index
Filtered_data2 = uwb2.loc[idx_2]

# Calculate time durations for UWB1
t_uwb1 = Filtered_data1['timestamp']
out = np.diff(t_uwb1.values.astype('datetime64[s]').astype(np.int64))
actualtime1 = np.insert(out, 0, 0)
time_duration = np.cumsum(actualtime1)

# Calculate time durations for UWB2
t_uwb2 = Filtered_data2['timestamp']
out2 = np.diff(t_uwb2.values.astype('datetime64[s]').astype(np.int64))
actualtime2 = np.insert(out2, 0, 0)
time_duration2 = np.cumsum(actualtime2)

# Plot FP Power Level (dBm) for UWB1
plt.figure()
plt.plot(time_duration, Filtered_data1['fp_pow_dbm'])
plt.xlabel('Duration (s)')
plt.ylabel('FP Power Level (dBm)')
plt.title('UWB1 FP Power Level')
plt.show()

# Plot FP Power Level (dBm) for UWB2
plt.figure()
plt.plot(time_duration2, Filtered_data2['fp_pow_dbm'])
plt.xlabel('Duration (s)')
plt.ylabel('FP Power Level (dBm)')
plt.title('UWB2 FP Power Level')
plt.show()
