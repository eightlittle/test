import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d

# x is anterio-posterior direction 
# y is bilateral direction
# z is vertical direction 

# usr your file path 
file_path_s1 = '/Users/kairenzheng/Desktop/example of code/step11_time_normalization_two_stretagies/Gait Data KINE 7670_Not Winters.csv'
df_s1 = pd.read_csv(file_path_s1, header=None)

# pick up time 
time_1 = df_s1.iloc[3:, 0].astype(float) 

file_path_s2 = '/Users/kairenzheng/Desktop/example of code/step11_time_normalization_two_stretagies/winterdataset.csv'
df_s2 = pd.read_csv(file_path_s2, header=None)

# pick up time 
time_2 = df_s2.iloc[2:, 1].astype(float)

#%% - just do the process to get velocity and acceleration (you can ignore this part - see step8)
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
#%% - make event1 event2 of the subject1 (you can ignore this part - see step10)
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
    
#%% - make event1 event2 of the subject1 (you can ignore this part - see step10)
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
    
#%% - first strategy of time normalization ( whole time normlization) - have time characteristics

# example for the entire period time normalization
# trials will have same length after time normalization
# still have time characteristics -> event time is different in each trials 
# Resampling function (time normlization function)
def time_normalize(data, target_length):
    original_length = data.shape[0]
    x_original = np.linspace(0, 1, original_length)
    x_target = np.linspace(0, 1, target_length)
    re_sample_data = interp1d(x_original, data, axis=0, kind='linear')(x_target)
    return re_sample_data

# time normalization entire data length 
target_length = 101 # you will change the number 
re_sample_s1_position_ankle_x = time_normalize(ankle_x_s1, target_length)
re_sample_s1_velocity_ankle_x = time_normalize(filtered_ankle_velocity_X_s1, target_length)
re_sample_s1_acceleration_ankle_x = time_normalize(ankle_acc_X_s1, target_length)
time_s1 = time_normalize(time_1, target_length)

# make events again after re-sampling 
event1_s1_after_resample = np.where(re_sample_s1_velocity_ankle_x[1:] < 8)[0]
if event1_s1_after_resample.size > 0:
    event1_s1_after_resample = event1_s1_after_resample[0]  
    print(f"Touch down (Event1_s1_after_re-sample): {event1_s1_after_resample}")
    
filtered_ankle_velocity_X_after_event1 = re_sample_s1_velocity_ankle_x[event1_s1_after_resample+1:]
event2_s1_after_resample = np.where(filtered_ankle_velocity_X_after_event1 > 20)[0]
if event2_s1_after_resample.size > 0:
    event2_s1_after_resample = event2_s1_after_resample[0] + event1_s1_after_resample  # 加回 Event1 的偏移量
    print(f"Take off (event2_s1_after_resample): {event2_s1_after_resample}")

# time normalization entire data length 
target_length = 101 # you will change the number 
re_sample_s2_position_ankle_x = time_normalize(ankle_x_s2, target_length)
re_sample_s2_velocity_ankle_x = time_normalize(filtered_ankle_velocity_X_s2, target_length)
re_sample_s2_acceleration_ankle_x = time_normalize(ankle_acc_X_s2, target_length)
time_s2 = time_normalize(time_2, target_length)

# make events again after re-sampling 
event1_s2_after_resample = np.where(re_sample_s2_velocity_ankle_x[1:] < 8)[0]
if event1_s2_after_resample.size > 0:
    event1_s2_after_resample = event1_s2_after_resample[0]  
    print(f"Touch down (Event1_s2_after_re-sample): {event1_s2_after_resample}")
    
filtered_ankle_velocity_X_after_event1 = re_sample_s2_velocity_ankle_x[event1_s2_after_resample+1:]
event2_s2_after_resample = np.where(filtered_ankle_velocity_X_after_event1 > 20)[0]
if event2_s2_after_resample.size > 0:
    event2_s2_after_resample = event2_s2_after_resample[0] + event1_s2_after_resample  # 加回 Event1 的偏移量
    print(f"Take off (event2_s2_after_resample): {event2_s2_after_resample}")


# Make a picture
# position in different trials 
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time_s1, re_sample_s1_position_ankle_x, label='Ankle X s1', color='blue', antialiased=True)
plt.plot(time_s2, re_sample_s2_position_ankle_x, label='Ankle X s2', color='red', antialiased=True)

# Label events with corresponding colors
event_points = [
    (time_s1[event1_s1_after_resample], re_sample_s1_position_ankle_x[event1_s1_after_resample], "1", 'blue'),  # Event1_s1
    (time_s1[event2_s1_after_resample], re_sample_s1_position_ankle_x[event2_s1_after_resample], "2", 'blue'),  # Event2_s1
    (time_s2[event1_s2_after_resample], re_sample_s2_position_ankle_x[event1_s2_after_resample], "1", 'red'),   # Event1_s2
    (time_s2[event2_s2_after_resample], re_sample_s2_position_ankle_x[event2_s2_after_resample], "2", 'red')    # Event2_s2
]

for x, y, label, color in event_points:
    plt.scatter(x, y, color=color, s=50, zorder=3)  # 標記點，使用對應顏色
    plt.text(x, y, label, fontsize=12, ha='right', va='bottom', color=color, fontweight='bold')

# Picture setting
plt.title('resample with entire data method', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position (cm)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


#%% - second strategy of time normalization (time normlization: event by event) - have  "no" time characteristics

# example for the entire period time normalization
# trials will have same length after time normalization in each period 
# still have time characteristics -> event time is different in each trials 
# Resampling function (time normlization function)
def time_normalize(data, target_length):
    original_length = data.shape[0]
    x_original = np.linspace(0, 1, original_length)
    x_target = np.linspace(0, 1, target_length)
    re_sample_data = interp1d(x_original, data, axis=0, kind='linear')(x_target)
    return re_sample_data

ankle_x_s1_frame1_to_event1 = ankle_x_s1[:event1_s1+1]  
ankle_x_s1_event1_to_event2 = ankle_x_s1[event1_s1+1:event2_s1+1] 
ankle_x_s1_event2_to_end = ankle_x_s1[event2_s1+1:]  

ankle_x_s2_frame1_to_event1 = ankle_x_s2[:event1_s2+1]  
ankle_x_s2_event1_to_event2 = ankle_x_s2[event1_s2+1:event2_s2+1] 
ankle_x_s2_event2_to_end = ankle_x_s2[event2_s2+1:]  

# time normalization event by event 
target_length = 101 # you will change the number 
data_ankle_x_s1_frame1_to_event1 = time_normalize(ankle_x_s1_frame1_to_event1, target_length)
data_ankle_x_s1_event1_to_event2 = time_normalize(ankle_x_s1_event1_to_event2, target_length)
data_ankle_x_s1_event2_to_end = time_normalize(ankle_x_s1_event2_to_end, target_length)

# time normalization event by event 
target_length = 101 # you will change the number 
data_ankle_x_s2_frame1_to_event1 = time_normalize(ankle_x_s2_frame1_to_event1, target_length)
data_ankle_x_s2_event1_to_event2 = time_normalize(ankle_x_s2_event1_to_event2, target_length)
data_ankle_x_s2_event2_to_end = time_normalize(ankle_x_s2_event2_to_end, target_length)

resample_eventByevent_data_s1 = np.hstack((data_ankle_x_s1_frame1_to_event1, data_ankle_x_s1_event1_to_event2, data_ankle_x_s1_event2_to_end))
resample_eventByevent_data_s2 = np.hstack((data_ankle_x_s2_frame1_to_event1, data_ankle_x_s2_event1_to_event2, data_ankle_x_s2_event2_to_end))
time_total_zone = np.arange(resample_eventByevent_data_s1.shape[0])

# make a picture
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time_total_zone, resample_eventByevent_data_s1, label='Ankle X s1', color='blue', antialiased=True)
plt.plot(time_total_zone, resample_eventByevent_data_s2, label='Ankle X s2', color='red', antialiased=True)
plt.axvline(x= target_length, color='red', linestyle='--', label='Touch Down')
plt.axvline(x= target_length * 2, color='green', linestyle='--', label='Take Off')
plt.title('reample with eventBYevent method)', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Velocity (cm/s)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()