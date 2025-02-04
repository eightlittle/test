import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# Dr. Weimar' lab setting
# X = bilateral axis
# Y = anterior-posterior axis
# Z = polar axis

# use your data file path 
file_path = '/Users/kairenzheng/Desktop/example of code/step7_filtering method/Gait Data KINE 7670_Not Winters.csv'
df = pd.read_csv(file_path, header=None)

# pick up time
time = df.iloc[3:, 0].astype(float)  
sampling_interval = time.iloc[1] - time.iloc[0]  # time difference
fs = 1 / sampling_interval  # frequency = 1/ time difference
cutoff = 6

# pick up point data 
point_x = df.iloc[3:, 3].astype(float).values
point_y = df.iloc[3:, 4].astype(float).values
point_z = df.iloc[3:, 5].astype(float).values

# we have to filter the data to remove the skin movements from the data 
# 4 the zero lag butterworth low pass filtering 
# https://en.wikipedia.org/wiki/Butterworth_filter -> introduction 
def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data)

# example to do filter 
sampling_interval = time.iloc[1] - time.iloc[0]  # time difference
fs = 1 / sampling_interval  # frequency = 1/ time difference
cutoff = 6
filtered_point_x = butterworth_filter(point_x, cutoff, fs, order=4, filter_type='low')
filtered_point_y = butterworth_filter(point_y, cutoff, fs, order=4, filter_type='low')
filtered_point_z = butterworth_filter(point_z, cutoff, fs, order=4, filter_type='low')

#%%
# shoe what is different btw filter and no filter acceleration 
# the reason why we need to filter data -> see the acceleration difference
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# use your data file path 
file_path = '/Users/kairenzheng/Desktop/example of code/step7_filtering method/Gait Data KINE 7670_Not Winters.csv'
df = pd.read_csv(file_path, header=None)
# pick up time
time = df.iloc[3:, 0].astype(float)  
sampling_interval = time.iloc[1] - time.iloc[0]  # time difference
fs = 1 / sampling_interval  # frequency = 1/ time difference

cutoff = 6

# pick up point data 
ankle_x = df.iloc[3:, 3].astype(float).values
ankle_y = df.iloc[3:, 4].astype(float).values
ankle_z = df.iloc[3:, 5].astype(float).values


def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data)

filtered_ankle_x = butterworth_filter(ankle_x, cutoff, fs, order=4, filter_type='low')



def time_d(data, sampling_interval):
    length = len(data)
    velocity = np.zeros(length)
    for i in range(length):
        if i == 0 or i == length - 1:
            velocity[i] = 0
        else:
            velocity[i] = (data[i + 1] - data[i - 1]) / (2 * sampling_interval)
    return velocity

def time_dd(data, sampling_interval):
    length = len(data)
    acceleration = np.zeros(length)
    for i in range(length):
        if i == 0 or i == length - 1:
            acceleration[i] = 0
        else:
            acceleration[i] = (data[i + 1] - 2 * data[i] + data[i - 1]) / (sampling_interval ** 2)
    return acceleration


r_ankle_acc_X = time_dd(ankle_x, sampling_interval)
r_ankle_acc_Y = time_dd(ankle_y, sampling_interval)

def process_motion_data(data, cutoff, fs, sampling_interval):
    """
    對輸入數據進行濾波，計算速度和加速度。

    參數:
    - data: array-like
        輸入的位移數據。
    - cutoff: float
        濾波器的截止頻率 (Hz)。
    - fs: float
        取樣頻率 (Hz)。
    - sampling_interval: float
        時間間隔 (s)。

    返回值:
    - filtered_position: array-like
        濾波後的位移數據。
    - filtered_velocity: array-like
        濾波後的速度數據。
    - filtered_acceleration: array-like
        濾波後的加速度數據。
    """

    # 濾波位移數據
    filtered_position = butterworth_filter(data, cutoff, fs, order=4, filter_type='low')

    # 計算速度和加速度
    raw_velocity = time_d(filtered_position, sampling_interval)
    raw_acceleration = time_dd(filtered_position, sampling_interval)

    # 濾波速度數據
    filtered_velocity = butterworth_filter(raw_velocity[1:-1], cutoff, fs, order=4, filter_type='low')
    filtered_velocity = np.insert(filtered_velocity, 0, 0)  # 開頭加 0
    filtered_velocity = np.append(filtered_velocity, 0)     # 結尾加 0

    # 濾波加速度數據
    filtered_acceleration = butterworth_filter(raw_acceleration[1:-1], cutoff, fs, order=4, filter_type='low')
    filtered_acceleration = np.insert(filtered_acceleration, 0, 0)  # 開頭加 0
    filtered_acceleration = np.append(filtered_acceleration, 0)     # 結尾加 0

    return filtered_position, filtered_velocity, filtered_acceleration


ankle_x, filtered_ankle_velocity_X, ankle_acc_X = process_motion_data(ankle_x, cutoff, fs, sampling_interval)
ankle_y, filtered_ankle_velocity_Y, ankle_acc_Y = process_motion_data(ankle_y, cutoff, fs, sampling_interval)



# 找到事件 Event1 和 Event2
event1 = np.where(filtered_ankle_velocity_X[1:] < 8)[0]
if event1.size > 0:
    event1 = event1[0]  # 取第一個索引
    print(f"Touch down (Event1): {event1}")
    
filtered_ankle_velocity_X_after_event1 = filtered_ankle_velocity_X[event1+1:]
event2 = np.where(filtered_ankle_velocity_X_after_event1 > 20)[0]
if event2.size > 0:
    event2 = event2[0] + event1  # 加回 Event1 的偏移量
    print(f"Take off (Event2): {event2}")

    
# 繪製加速度圖並添加事件垂直線
plt.figure(figsize=(8, 6), dpi=300)

plt.plot(time[1:-1], ankle_acc_X[1:-1], label='Ankle_acc X', color='yellow', antialiased=True)
plt.plot(time[1:-1], ankle_acc_Y[1:-1], label='Ankle_acc Y', color='green', antialiased=True)
plt.axvline(x=time[event1], color='red', linestyle='--', label='Touch Down')
plt.axvline(x=time[event2], color='green', linestyle='--', label='Take Off')
plt.title('Filtered Acceleration Over Time (No First and Last Frames)', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Acceleration (cm/s²)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 繪製加速度圖並添加事件垂直線
plt.figure(figsize=(8, 6), dpi=300)

plt.plot(time[1:-1], r_ankle_acc_X[1:-1], label='Ankle_acc X', color='yellow', antialiased=True)
plt.plot(time[1:-1], r_ankle_acc_Y[1:-1], label='Ankle_acc Y', color='green', antialiased=True)
plt.axvline(x=time[event1], color='red', linestyle='--', label='Touch Down')
plt.axvline(x=time[event2], color='green', linestyle='--', label='Take Off')
plt.title('No filtered Acceleration Over Time (No First and Last Frames)', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Acceleration (cm/s²)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()



