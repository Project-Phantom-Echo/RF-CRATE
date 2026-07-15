import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pywt

# Load data from .mat files
nuc1 = sio.loadmat('../wificsi1/wificsi1_exp018.mat')
nuc2 = sio.loadmat('../wificsi2/wificsi2_exp018.mat')

# Define start and end times
start_time = datetime.strptime('18:04:44.553', '%H:%M:%S.%f')
end_time = datetime.strptime('18:08:11.553', '%H:%M:%S.%f')

# Convert timestamps to datetime objects
date1 = [datetime.strptime(ts[0], '%H:%M:%S.%f') for ts in nuc1['timestamp']]

# Find indices for the start and end times
idx1 = np.argmin([abs((start_time - t).total_seconds()) for t in date1])
idx11 = np.argmin([abs((end_time - t).total_seconds()) for t in date1])

samples = np.arange(idx1, idx11 + 1)

t_csi1 = np.array([date1[i] for i in samples])
out = np.diff([t.timestamp() for t in t_csi1])
actualtime1 = np.insert(out, 0, 0)
time_duration = np.cumsum(actualtime1)

t_csi2 = np.array([date1[i] for i in samples])
out2 = np.diff([t.timestamp() for t in t_csi2])
actualtime2 = np.insert(out2, 0, 0)
time_duration2 = np.cumsum(actualtime2)

# Plot raw un-filtered data
plt.figure()
plt.plot(time_duration, np.abs(nuc1['tx1rx1_sub10'][samples]), label='NUC1')
plt.plot(time_duration2, np.abs(nuc2['tx1rx1_sub10'][samples]), label='NUC2')
plt.legend()
plt.xlabel('Duration (s)')
plt.ylabel('CSI Amplitude')
plt.show()

# 1D wavelet denoising
scal = 'sln'
dwt_denoised_sig_nuc1 = pywt.wavedec(np.abs(nuc1['tx1rx1_sub10'].flatten()), 'sym3', mode='s', level=4)
dwt_denoised_sig_nuc1 = pywt.waverec(dwt_denoised_sig_nuc1, 'sym3')

dwt_denoised_sig_nuc2 = pywt.wavedec(np.abs(nuc2['tx1rx1_sub10'].flatten()), 'sym3', mode='s', level=4)
dwt_denoised_sig_nuc2 = pywt.waverec(dwt_denoised_sig_nuc2, 'sym3')

# Plot denoised data
plt.figure()
plt.plot(time_duration, np.abs(dwt_denoised_sig_nuc1[samples]), label='NUC1')
plt.plot(time_duration2, np.abs(dwt_denoised_sig_nuc2[samples]), label='NUC2')
plt.legend()
plt.xlabel('Duration (s)')
plt.ylabel('CSI Amplitude')
plt.show()
