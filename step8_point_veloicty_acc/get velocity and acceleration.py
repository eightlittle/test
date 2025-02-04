import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# usr your file path 
file_path = "/Users/kairenzheng/Desktop/AU classes/KINE_7670/Motion Capture and Winter's Assignments/Gait Data KINE 7670_Not Winters.csv"
df = pd.read_csv(file_path, header=None)

# pick up time 
time = df.iloc[3:, 0].astype(float) 

#%% - just do the process to get velocity and acceleration ( you can ignore this part)
# pick up point data 
ankle_x = df.iloc[3:, 3].astype(float).values 
ankle_y = df.iloc[3:, 4].astype(float).values

# set up filtering elements (you can ignore here)
# apply point to do filtering (you can ignore here)
sampling_interval = time.iloc[1] - time.iloc[0]  # time difference
fs = 1 / sampling_interval  # frequency = 1/ time difference
cutoff = 6
def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data)

# make velocity (you can ignore here)
def time_d(data, sampling_interval):
    length = len(data)
    velocity = np.zeros(length)
    for i in range(length):
        if i == 0 or i == length - 1:
            velocity[i] = 0
        else:
            velocity[i] = (data[i + 1] - data[i - 1]) / (2 * sampling_interval)
    return velocity

# make acceleration (you can ignore here)
def time_dd(data, sampling_interval):
    length = len(data)
    acceleration = np.zeros(length)
    for i in range(length):
        if i == 0 or i == length - 1:
            acceleration[i] = 0
        else:
            acceleration[i] = (data[i + 1] - 2 * data[i] + data[i - 1]) / (sampling_interval ** 2)
    return acceleration

# do whole process of filtering velocity, and acceleration (you can ignore here)
def process_motion_data(data, cutoff, fs, sampling_interval):
    filtered_position = butterworth_filter(data, cutoff, fs, order=4, filter_type='low')
    raw_velocity = time_d(filtered_position, sampling_interval)
    raw_acceleration = time_dd(filtered_position, sampling_interval)
    filtered_velocity = butterworth_filter(raw_velocity[1:-1], cutoff, fs, order=4, filter_type='low')
    filtered_velocity = np.insert(filtered_velocity, 0, 0)  
    filtered_velocity = np.append(filtered_velocity, 0)
    filtered_acceleration = butterworth_filter(raw_acceleration[1:-1], cutoff, fs, order=4, filter_type='low')
    filtered_acceleration = np.insert(filtered_acceleration, 0, 0)
    filtered_acceleration = np.append(filtered_acceleration, 0)
    return filtered_position, filtered_velocity, filtered_acceleration

ankle_x, filtered_ankle_velocity_X, ankle_acc_X = process_motion_data(ankle_x, cutoff, fs, sampling_interval)
ankle_y, filtered_ankle_velocity_Y, ankle_acc_Y = process_motion_data(ankle_y, cutoff, fs, sampling_interval)

#%%
# assume we already have all data that we need 
# find event 1 and event 2 
# set up if the ankle x veloicty lower than 
# run the code, and you will see the frame of event1 and event2 
event1 = np.where(filtered_ankle_velocity_X[1:] < 8)[0]
if event1.size > 0:
    event1 = event1[0]  
    print(f"Touch down (Event1): {event1}")
    
filtered_ankle_velocity_X_after_event1 = filtered_ankle_velocity_X[event1+1:]
event2 = np.where(filtered_ankle_velocity_X_after_event1 > 20)[0]
if event2.size > 0:
    event2 = event2[0] + event1  # 加回 Event1 的偏移量
    print(f"Take off (Event2): {event2}")
    
# make pictures of position 
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time, ankle_x, label='Ankle X', color='blue', antialiased=True)
plt.plot(time, ankle_y, label='Ankle Y', color='orange', antialiased=True)
plt.axvline(x=time[event1], color='red', linestyle='--', label='Touch Down') # add event1
plt.axvline(x=time[event2], color='green', linestyle='--', label='Take Off') # add event2
plt.title('Marker Position Over Time', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position (cm)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# make pictures of velocity - no first and last frame because those are 0 
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time[1:-1], filtered_ankle_velocity_X[1:-1], label='Ankle_Vel X', color='blue', antialiased=True)
plt.plot(time[1:-1], filtered_ankle_velocity_Y[1:-1], label='Ankle_Vel Y', color='orange', antialiased=True)
plt.axvline(x=time[event1], color='red', linestyle='--', label='Touch Down') # add event1
plt.axvline(x=time[event2], color='green', linestyle='--', label='Take Off') # add event2
plt.title('Filtered Velocity Over Time (No First and Last Frame)', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Velocity (cm/s)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# make pictures of acceleration - no first and last frame because those are 0 
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time[1:-1], ankle_acc_X[1:-1], label='Ankle_acc X', color='yellow', antialiased=True)
plt.plot(time[1:-1], ankle_acc_Y[1:-1], label='Ankle_acc Y', color='green', antialiased=True)
plt.axvline(x=time[event1], color='red', linestyle='--', label='Touch Down') # add event1
plt.axvline(x=time[event2], color='green', linestyle='--', label='Take Off') # add event2
plt.title('Filtered Acceleration Over Time (No First and Last Frames)', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Acceleration (cm/s²)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

