import numpy as np
from scipy.signal import butter, lfilter, freqz
# import matplotlib.pyplot as plt
# import os
#
# address = "data" + os.path.sep + "Imu20.npz"
#
# with np.load(address) as data:
#     imu_angular_velocity = data["angular_velocity"]  # angular velocity in rad/sec
#     imu_linear_acceleration = data["linear_acceleration"]  # Accelerations in gs (gravity acceleration scaling)
#     imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
#     time_stamps = data["time_stamps"]
#     yawData = imu_linear_acceleration[2]
# print(yawData)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# # Filter requirements.
# order = 6
# fs = 1000     # sample rate, Hz
# cutoff = 10 # desired cutoff frequency of the filter, Hz
#
#
# # Demonstrate the use of the filter.
# # First make some data to be filtered.
# T = 1000         # seconds
# n = int(T * fs) # total number of samples
# t = np.linspace(0, T, len(yawData), endpoint=False)
# # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
# # data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)
#
# # Filter the data, and plot both the original and filtered signals.
# y = butter_lowpass_filter(yawData, cutoff, fs, order)
#
# plt.subplot(2, 1, 2)
# plt.plot(t, yawData, 'b-', label='data')
# plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
# plt.xlabel('Time [sec]')
# plt.grid()
# plt.legend()
#
# plt.subplots_adjust(hspace=0.35)
# plt.show()