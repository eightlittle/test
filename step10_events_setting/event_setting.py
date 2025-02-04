import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# x is anterio-posterior direction 
# y is bilateral direction
# z is vertical direction 

# usr your file path 
file_path_s1 = '/Users/kairenzheng/Desktop/example of code/step10_events_setting/Gait Data KINE 7670_Not Winters.csv'
df_s1 = pd.read_csv(file_path_s1, header=None)

# pick up time 
time_1 = df_s1.iloc[3:, 0].astype(float) 

file_path_s2 = '/Users/kairenzheng/Desktop/example of code/step10_events_setting/winterdataset.csv'
df_s2 = pd.read_csv(file_path_s2, header=None)

# pick up time 
time_2 = df_s2.iloc[2:, 1].astype(float)

#%% - just do the process to get velocity and acceleration ( you can ignore this part)
# pick up point data 
ankle_x_subject1 = df_s1.iloc[3:, 3].astype(float).values 
ankle_y_subject1  = df_s1.iloc[3:, 4].astype(float).values

ankle_x_subject2 = df_s2.iloc[2:, 8].astype(float).values 
ankle_y_subject2  = df_s2.iloc[2:, 7].astype(float).values
ankle_x_subject2 = ankle_x_subject2 * 100
ankle_y_subject2  = ankle_y_subject2 * 100

# set up filtering elements (you can ignore here)
# apply point to do filtering (you can ignore here)

sampling_interval = time_1.iloc[1] - time_1.iloc[0]  # time difference
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

ankle_x_s1, filtered_ankle_velocity_X_s1, ankle_acc_X_s1 = process_motion_data(ankle_x_subject1, cutoff, fs, sampling_interval)
ankle_y_s1, filtered_ankle_velocity_Y_s1, ankle_acc_Y_s1 = process_motion_data(ankle_y_subject1, cutoff, fs, sampling_interval)


ankle_x_s2, filtered_ankle_velocity_X_s2, ankle_acc_X_s2 = process_motion_data(ankle_x_subject2, cutoff, fs, sampling_interval)
ankle_y_s2, filtered_ankle_velocity_Y_s2, ankle_acc_Y_s2 = process_motion_data(ankle_y_subject2, cutoff, fs, sampling_interval)
#%%
# assume we already have all data that we need 
# find event 1 and event 2 
# set up if the ankle x veloicty lower than 
# run the code, and you will see the frame of event1 and event2 
event1_s1 = np.where(filtered_ankle_velocity_X_s1[1:] < 8)[0]
if event1_s1.size > 0:
    event1_s1 = event1_s1[0]  
    print(f"Touch down (Event1_s1): {event1_s1}")
    
filtered_ankle_velocity_X_after_event1 = filtered_ankle_velocity_X_s1[event1_s1+1:]
event2_s1 = np.where(filtered_ankle_velocity_X_after_event1 > 20)[0]
if event2_s1.size > 0:
    event2_s1 = event2_s1[0] + event1_s1  # 加回 Event1 的偏移量
    print(f"Take off (event2_s1): {event2_s1}")
    
#%%
# assume we already have all data that we need 
# find event 1 and event 2 
# event setting can change (depends on your event setting) 

# run the code, and you will see the frame of event1 and event2 
event1_s2 = np.where(filtered_ankle_velocity_X_s2[1:] < 8)[0] # setting id the velocity is lower than 8
if event1_s2.size > 0:
    event1_s2 = event1_s2[0]  # pick up the frame of the event 
    print(f"Touch down (event1_s2): {event1_s2}")
    
filtered_ankle_velocity_X_after_event1 = filtered_ankle_velocity_X_s2[event1_s2+1:] 
# cut the data before event 1 
# sometime the data before event 1 will influence the event setting 

event2_s2 = np.where(filtered_ankle_velocity_X_after_event1 > 20)[0] # setting id the velocity is higher than 20
if event2_s2.size > 0:
    event2_s2 = event2_s2[0] + event1_s2  # add the frame before the frame -> make the frame number of the entire data
    print(f"Take off (event2_s2): {event2_s2}")
    
#%% - make pictures 
# make pictures of position - S1
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time_1, ankle_x_s1, label='Ankle X', color='blue', antialiased=True)
plt.plot(time_1, ankle_y_s1, label='Ankle Y', color='orange', antialiased=True)
plt.axvline(x=time_1[event1_s1], color='red', linestyle='--', label='Touch Down') # add event1
plt.axvline(x=time_1[event2_s1], color='green', linestyle='--', label='Take Off') # add event2
plt.title('Marker Position S1', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position (cm)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# make pictures of position - S2
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time_2, ankle_x_s2, label='Ankle X', color='blue', antialiased=True)
plt.plot(time_2, ankle_y_s2, label='Ankle Y', color='orange', antialiased=True)
plt.axvline(x=time_2[event1_s2], color='red', linestyle='--', label='Touch Down') # add event1
plt.axvline(x=time_2[event2_s2], color='green', linestyle='--', label='Take Off') # add event2
plt.title('Marker Position S2', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position (cm)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# make pictures of velocity - S1
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time_1[1:-1], filtered_ankle_velocity_X_s1[1:-1], label='Ankle_vel X', color='blue', antialiased=True)
plt.plot(time_1[1:-1], filtered_ankle_velocity_Y_s1[1:-1], label='Ankle_vel Y', color='orange', antialiased=True)
plt.axvline(x=time_1[event1_s1], color='red', linestyle='--', label='Touch Down') # add event1
plt.axvline(x=time_1[event2_s1], color='green', linestyle='--', label='Take Off') # add event2
plt.title('Marker velocity S1', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position (cm)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# make pictures of velocity - S2
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time_2[1:-1], filtered_ankle_velocity_X_s2[1:-1], label='Ankle_vel X', color='blue', antialiased=True)
plt.plot(time_2[1:-1], filtered_ankle_velocity_Y_s2[1:-1], label='Ankle_vel Y', color='orange', antialiased=True)
plt.axvline(x=time_2[event1_s2], color='red', linestyle='--', label='Touch Down') # add event1
plt.axvline(x=time_2[event2_s2], color='green', linestyle='--', label='Take Off') # add event2
plt.title('Marker velocity S2', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position (cm)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# make pictures of acceleration - S1
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time_1[1:-1], ankle_acc_X_s1[1:-1], label='Ankle_vel X', color='blue', antialiased=True)
plt.plot(time_1[1:-1], ankle_acc_Y_s1[1:-1], label='Ankle_vel Y', color='orange', antialiased=True)
plt.axvline(x=time_1[event1_s1], color='red', linestyle='--', label='Touch Down') # add event1
plt.axvline(x=time_1[event2_s1], color='green', linestyle='--', label='Take Off') # add event2
plt.title('Marker acceleration S1', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position (cm)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# make pictures of velocity - S2
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time_2[1:-1], ankle_acc_X_s2[1:-1], label='Ankle_vel X', color='blue', antialiased=True)
plt.plot(time_2[1:-1], ankle_acc_Y_s2[1:-1], label='Ankle_vel Y', color='orange', antialiased=True)
plt.axvline(x=time_2[event1_s2], color='red', linestyle='--', label='Touch Down') # add event1
plt.axvline(x=time_2[event2_s2], color='green', linestyle='--', label='Take Off') # add event2
plt.title('Marker acceleration S2', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position (cm)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()