import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 使用者檔案路徑
file_path = "/Users/kairenzheng/Desktop/AU classes/KINE_7670/Motion Capture and Winter's Assignments/Gait Data KINE 7670_Not Winters.csv"
df = pd.read_csv(file_path, header=None)

# 讀取時間欄位
time = df.iloc[3:, 0].astype(float)

#%% - 取得踝關節座標資料
ankle_x = df.iloc[3:, 3].astype(float).values
ankle_y = df.iloc[3:, 4].astype(float).values

# 計算取樣頻率
sampling_interval = time.iloc[1] - time.iloc[0]
fs = 1 / sampling_interval  # 頻率 (Hz)
cutoff = 6  # 截止頻率 (Hz)

# 巴特沃斯濾波器 (Butterworth Filter)
def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data)

# 計算速度 (支援 1D 和 2D 資料)
def time_d(data, sampling_interval):
    data = np.atleast_2d(data).T if data.ndim == 1 else data
    velocity = np.zeros_like(data)
    for i in range(1, data.shape[0] - 1):
        velocity[i, :] = (data[i + 1, :] - data[i - 1, :]) / (2 * sampling_interval)
    return velocity.squeeze()

# 計算加速度 (支援 1D 和 2D 資料)
def time_dd(data, sampling_interval):
    data = np.atleast_2d(data).T if data.ndim == 1 else data
    acceleration = np.zeros_like(data)
    for i in range(1, data.shape[0] - 1):
        acceleration[i, :] = (data[i + 1, :] - 2 * data[i, :] + data[i - 1, :]) / (sampling_interval ** 2)
    return acceleration.squeeze()

# 完整處理過程：濾波、計算速度和加速度
def process_motion_data(data, cutoff, fs, sampling_interval):
    filtered_position = butterworth_filter(data, cutoff, fs)
    raw_velocity = time_d(filtered_position, sampling_interval)
    raw_acceleration = time_dd(filtered_position, sampling_interval)

    filtered_velocity = butterworth_filter(raw_velocity, cutoff, fs)
    filtered_acceleration = butterworth_filter(raw_acceleration, cutoff, fs)

    return filtered_position, filtered_velocity, filtered_acceleration

# 對踝關節的 X 和 Y 座標進行處理
ankle_x_pos, ankle_x_vel, ankle_x_acc = process_motion_data(ankle_x, cutoff, fs, sampling_interval)
ankle_y_pos, ankle_y_vel, ankle_y_acc = process_motion_data(ankle_y, cutoff, fs, sampling_interval)

#%% - 繪製結果以進行檢查
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, ankle_x_pos, label='Filtered Ankle X Position')
plt.ylabel('Position (m)')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, ankle_x_vel, label='Filtered Ankle X Velocity', color='orange')
plt.ylabel('Velocity (m/s)')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, ankle_x_acc, label='Filtered Ankle X Acceleration', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()

plt.tight_layout()
plt.show()


