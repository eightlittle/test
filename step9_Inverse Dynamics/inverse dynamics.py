import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

#%% - just pick up data and filter them 
# Load data
file_path = '/Users/kairenzheng/Desktop/example of code/step9_Inverse Dynamics/example data for IV.csv'
df = pd.read_csv(file_path, header=None)
time = df.iloc[2:, 5].astype(float).to_numpy()

# pick up mass of each segment
total_mass = df.iloc[0, 1]
mass_ft = df.iloc[1, 1]
mass_shank = df.iloc[2, 1]
mass_thigh = df.iloc[3, 1]

# pick up MOI of each segment
MOI_ft = df.iloc[7:10, 0:3].astype(float).to_numpy()
MOI_shank = df.iloc[12:15, 0:3].astype(float).to_numpy()
MOI_thigh = df.iloc[17:20, 0:3].astype(float).to_numpy()


# Define filter function
def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    b, a = butter(order, normalized_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data, axis=0)

# Sampling frequency and cutoff frequency
fs = 500
cutoff = 6

# Define data categories and apply filtering
data_variables = {
    "com_Rft": df.iloc[2:, 6:9].astype(float).to_numpy(),
    "com_Lft": df.iloc[2:, 9:12].astype(float).to_numpy(),
    "com_Rshank": df.iloc[2:, 12:15].astype(float).to_numpy(),
    "com_Lshank": df.iloc[2:, 15:18].astype(float).to_numpy(),
    "com_Rthigh": df.iloc[2:, 18:21].astype(float).to_numpy(),
    "com_Lthigh": df.iloc[2:, 21:24].astype(float).to_numpy(),
    "jc_Rank": df.iloc[2:, 24:27].astype(float).to_numpy(),
    "jc_Lank": df.iloc[2:, 27:30].astype(float).to_numpy(),
    "jc_Rknee": df.iloc[2:, 30:33].astype(float).to_numpy(),
    "jc_Lknee": df.iloc[2:, 33:36].astype(float).to_numpy(),
    "jc_Rhip": df.iloc[2:, 36:39].astype(float).to_numpy(),
    "jc_Lhip": df.iloc[2:, 39:42].astype(float).to_numpy(),
    "rleg_GRF": df.iloc[2:, 42:45].astype(float).to_numpy(),
    "lleg_GRF": df.iloc[2:, 45:48].astype(float).to_numpy(),
    "rleg_GRT": df.iloc[2:, 48:51].astype(float).to_numpy(),
    "lleg_GRT": df.iloc[2:, 51:54].astype(float).to_numpy(),
    "rleg_COP": df.iloc[2:, 54:57].astype(float).to_numpy(),
    "lleg_COP": df.iloc[2:, 57:60].astype(float).to_numpy(),
    "angle_Rft": df.iloc[2:, 60:63].astype(float).to_numpy(),
    "angle_Lft": df.iloc[2:, 63:66].astype(float).to_numpy(),
    "angle_Rshank": df.iloc[2:, 66:69].astype(float).to_numpy(),
    "angle_Lshank": df.iloc[2:, 69:72].astype(float).to_numpy(),
    "angle_Rthigh": df.iloc[2:, 72:75].astype(float).to_numpy(),
    "angle_Lthigh": df.iloc[2:, 75:78].astype(float).to_numpy()
}

# Apply filtering to all variables
filtered_data = {key: butterworth_filter(value, cutoff, fs) for key, value in data_variables.items()}

#%%
sampling_interval = 0.002
frequency = 500 

def time_d(data, sampling_interval):
    time_d = np.zeros_like(data)
    time_d[1:-1] = (data[2:] - data[:-2]) / (2 * sampling_interval)
    return time_d

# segment linear velocity 
com_Rft = filtered_data['com_Rft']
vel_Rft = time_d(com_Rft, sampling_interval)

com_Lft = filtered_data['com_Lft']
vel_Lft = time_d(com_Lft, sampling_interval)

com_Rshank = filtered_data['com_Rshank']
vel_Rshank = time_d(com_Rshank, sampling_interval)

com_Lshank = filtered_data['com_Lshank']
vel_Lshank = time_d(com_Lshank, sampling_interval)

com_Rthigh = filtered_data['com_Rthigh']
vel_Rthigh = time_d(com_Rthigh, sampling_interval)

com_Lthigh = filtered_data['com_Lthigh']
vel_Lthigh = time_d(com_Lthigh, sampling_interval)


# concept of force
# total force = inernal force (body generate) + gravity + external force (GRF)
# joint force (internal force) = total force - gravity - external force 

# elements for joint force functions 
def inverse_dy_force(mass, segments_LV, timestep):
    """
    input1  mass of segments (kg)
    input2  segment linear velocity (cm / s)
    inpur3  time steps  = ( 1 / frequency )  ex 1000hz = 0.001
    -------
    outcome1 time derivative of linear momentum of segment
    outcome2 weight of segment ( mass * 9.81 of Z axis)
    """
    # only vertical have weight 
    gravity = np.array((0, 0, 9.81))
    Weight = mass * gravity
    segments_LV = segments_LV / 100
    # momentum = (kg.m/s)
    momentum_segment = mass * segments_LV
    # why do time_d to momentum(kg.m/s) -> it will be internal force (kg.m/s^2 = N)
    momentum_dtime = time_d(momentum_segment[1:-1], timestep)
    zero_row = np.zeros((1, 3))
    # add zero to the first frame and final frame to keep the same frame 
    momentum_dtime = np.vstack([zero_row, momentum_dtime, zero_row])

    return momentum_dtime, Weight

# for joint force, we need momentum with time derivative and weight of each segment 
rft_momentum_dtime, rft_weight = inverse_dy_force(mass_ft, vel_Rft, sampling_interval)
lft_momentum_dtime, lft_weight = inverse_dy_force(mass_ft, vel_Lft, sampling_interval)
rshank_momentum_dtime, rshank_weight = inverse_dy_force(mass_shank, vel_Rshank, sampling_interval)
lshank_momentum_dtime, lshank_weight = inverse_dy_force(mass_shank, vel_Rshank, sampling_interval)
rthigh_momentum_dtime, rthigh_weight = inverse_dy_force(mass_thigh, vel_Rthigh, sampling_interval)
lthigh_momentum_dtime, lthigh_weight = inverse_dy_force(mass_thigh, vel_Lthigh, sampling_interval)

# joint force = sum_momentum_dtime - sum_segment_weight - GRF
# force of ankles
force_Rank = rft_momentum_dtime - rft_weight - filtered_data['rleg_GRF']
force_Lank = lft_momentum_dtime - lft_weight - filtered_data['lleg_GRF']

# force of knees 
force_Rknee = (
    (rft_momentum_dtime + rshank_momentum_dtime)
    - (rft_weight + rshank_weight)
    - filtered_data['rleg_GRF']
)

force_Lknee = (
    (lft_momentum_dtime + lshank_momentum_dtime)
    - (lft_weight + lshank_weight)
    - filtered_data['lleg_GRF']
)

# force of hip
force_Rhip = (
    (rft_momentum_dtime + rshank_momentum_dtime + rthigh_momentum_dtime)
    - (rft_weight + rshank_weight + rthigh_weight)
    - filtered_data['rleg_GRF']
)

force_Lhip = (
    (lft_momentum_dtime + lshank_momentum_dtime + lthigh_momentum_dtime)
    - (lft_weight + lshank_weight + lthigh_weight)
    - filtered_data['lleg_GRF']
)

# segment linear velocity 
angle_Rft = filtered_data['angle_Rft']
a_vel_Rft = time_d(angle_Rft, sampling_interval)

angle_Lft = filtered_data['angle_Lft']
a_vel_Lft = time_d(angle_Lft, sampling_interval)

angle_Rshank = filtered_data['angle_Rshank']
a_vel_Rshank = time_d(angle_Rshank, sampling_interval)

angle_Lshank = filtered_data['angle_Lshank']
a_vel_Lshank = time_d(angle_Lshank, sampling_interval)

angle_Rthigh = filtered_data['angle_Rthigh']
a_vel_Rthigh = time_d(angle_Rthigh, sampling_interval)

angle_Lthigh = filtered_data['angle_Lthigh']
a_vel_Lthigh = time_d(angle_Lthigh, sampling_interval)

# elements for joint torque functions 
def inverse_dy_torque(mois, segments_angular_velocity, joints_position, segment_com_position, timestep):
    """
    input1 = MOI of segment
    input2 = segment angular velocicty
    input3 = joint position
    input4 = com position of segment
    input5 = time steps ex 1000hz = 0.001
    outcome1 = time derivative of linear momentum of segment
    outcome2 = length from joint to segment 
    """
    a = segment_com_position.shape[0]
    angular_momentum_segment = np.zeros((a, 3))
    for i in range(a):
        # angular momentum = kg-m2/sec
        angular_momentum_segment[i, :] = np.dot(mois, segments_angular_velocity[i, :])
    time_derivative_angular_momentum_segment = time_d(angular_momentum_segment[1:-1], timestep)
    zero_row = np.zeros((1, 3))
    # add zero to the first frame and final frame to keep the same frame 
    time_derivative_angular_momentum_segment = np.vstack([zero_row, time_derivative_angular_momentum_segment, zero_row])
    r_joint_to_segment_com = joints_position - segment_com_position
    return time_derivative_angular_momentum_segment, r_joint_to_segment_com

# elements for ankle torque
Rft_angular_dtime, r_Rank_to_Rft = inverse_dy_torque(MOI_ft, a_vel_Rft, filtered_data['jc_Rank'], filtered_data['com_Rft'], sampling_interval)
Lft_angular_dtime, r_Lank_to_Lft = inverse_dy_torque(MOI_ft, a_vel_Lft, filtered_data['jc_Lank'], filtered_data['com_Lft'], sampling_interval)
r_Rank_to_cop1 = filtered_data['jc_Rank'] - com_Rft
r_Lank_to_cop2 = filtered_data['jc_Lank'] - com_Lft

# elements for knee torque
Rsha_angular_dtime, r_Rknee_to_Rsha = inverse_dy_torque(MOI_shank, a_vel_Rshank, filtered_data['jc_Rknee'], filtered_data['com_Rshank'], sampling_interval)
Lsha_angular_dtime, r_Lknee_to_Lsha = inverse_dy_torque(MOI_shank, a_vel_Lshank, filtered_data['jc_Lknee'], filtered_data['com_Rshank'], sampling_interval)
r_Rknee_to_Rft = filtered_data['jc_Rknee'] - filtered_data['com_Rft']
r_Rknee_to_cop1 = filtered_data['jc_Rknee'] - com_Rft
r_Lknee_to_Lft = filtered_data['jc_Lknee'] - filtered_data['com_Lft']
r_Lknee_to_cop2 = filtered_data['jc_Lknee'] - com_Lft


# elements for hip torque
Rthi_angular_dtime, r_Rhip_to_Rthi = inverse_dy_torque(mass_thigh, a_vel_Rthigh, filtered_data['jc_Rhip'], filtered_data["com_Rthigh"], sampling_interval)
Lthi_angular_dtime, r_Lhip_to_Lthi = inverse_dy_torque(mass_thigh, a_vel_Lthigh, filtered_data['jc_Lhip'], filtered_data["com_Rthigh"], sampling_interval)
r_Rhip_to_Rsha = filtered_data['jc_Rhip'] - filtered_data['com_Rshank']
r_Rhip_to_Rft = filtered_data['jc_Rhip'] - filtered_data['com_Rft']
r_Rhip_to_cop1 = filtered_data['jc_Rhip'] - com_Rft
r_Lhip_to_Lsha = filtered_data['jc_Lhip'] - filtered_data['com_Lshank']
r_Lhip_to_Lft = filtered_data['jc_Lhip'] - filtered_data['com_Lft']
r_Lhip_to_cop2 = filtered_data['jc_Lhip'] - com_Lft

# torque of ankle
torque_Rank = (Rft_angular_dtime 
               + (r_Rank_to_Rft * rft_momentum_dtime) 
               - (Rft_angular_dtime * rft_weight) 
               - (r_Rank_to_cop1 * filtered_data['rleg_GRF'] + filtered_data['rleg_GRT'])
               )

rorque_Lank = (Lft_angular_dtime 
               + (r_Lank_to_Lft * lft_momentum_dtime) 
               - (Lft_angular_dtime * lft_weight) 
               - (r_Lank_to_cop2 * filtered_data['lleg_GRF'] + filtered_data['lleg_GRT'])
               )

# torque of knee
torque_Rknee = (( Rsha_angular_dtime + Rft_angular_dtime) 
                + (r_Rknee_to_Rsha * rshank_momentum_dtime + r_Rknee_to_Rft * rft_weight) 
                - (r_Rknee_to_Rsha * rshank_weight + r_Rknee_to_Rft * rft_weight) 
                - (r_Rknee_to_cop1 *  filtered_data['rleg_GRF'] + filtered_data['rleg_GRT'])
                )

torque_Lknee = (( Lsha_angular_dtime + Lft_angular_dtime) 
                + (r_Lknee_to_Lsha * lshank_momentum_dtime + r_Lknee_to_Lft * lft_weight) 
                - (r_Lknee_to_Lsha * lshank_weight + r_Lknee_to_Lft * lft_weight) 
                - (r_Lknee_to_cop2 *  filtered_data['lleg_GRF'] + filtered_data['lleg_GRT']),
                )
# torque of hip
torque_Rhip = (( Rthi_angular_dtime + Rsha_angular_dtime + Rft_angular_dtime ) 
               + (r_Rhip_to_Rthi * rthigh_momentum_dtime 
                  + r_Rhip_to_Rsha * rshank_momentum_dtime 
                  + r_Rhip_to_Rft * rft_momentum_dtime ) 
               - (r_Rhip_to_Rthi * rthigh_weight 
                  + r_Rhip_to_Rsha * rshank_weight 
                  + r_Rhip_to_Rft * rft_weight ) 
               - (r_Rhip_to_cop1 * filtered_data['rleg_GRF'] + filtered_data['rleg_GRT'])
               )

torque_Lhip = (( Lthi_angular_dtime + Lsha_angular_dtime + Lft_angular_dtime ) 
               + (r_Lhip_to_Lthi * lthigh_momentum_dtime 
                  + r_Lhip_to_Lsha * lshank_momentum_dtime 
                  + r_Lhip_to_Lft * lft_momentum_dtime ) 
               - (r_Lhip_to_Lthi * lthigh_weight 
                  + r_Lhip_to_Lsha * lshank_weight 
                  + r_Lhip_to_Lft * lft_weight ) 
               - (r_Lhip_to_cop2 * filtered_data['lleg_GRF'] + filtered_data['lleg_GRT'])
               )

#%%
# kinetics 

# momentum
# linear momentum  = mass * Linear velocity
# Angular momentum = MOI  * Angular velocity

# examples 
linear_momentum_Rft = mass_ft * vel_Rft
range_frame = a_vel_Rft.shape[0]
angular_momentum_segment = np.zeros((range_frame, 3))
for i in range(range_frame):
    # angular momentum = kg-m2/sec
    angular_momentum_segment[i, :] = np.dot(MOI_ft, a_vel_Rft[i, :])

# joint power
# linear power = force * linear velocity
# angular power = torque * angular vleoicty 

# example 
sampling_interval = 0.002
# create joint velocity 
def time_d(data, sampling_interval):
    time_d = np.zeros_like(data)
    time_d[1:-1] = (data[2:] - data[:-2]) / (2 * sampling_interval)
    return time_d
vel_Rankle = time_d(filtered_data['jc_Rank'], sampling_interval)
vel_Rknee = time_d(filtered_data['jc_Rknee'], sampling_interval)

# find linear power of ankle 
linear_power_Rank = force_Rank * vel_Rankle 

# find ankle angle (distal segment angle - proximal segment angle)
angle_Rankle = filtered_data['angle_Rft'] - filtered_data['angle_Rshank'] 
def time_d(data, sampling_interval):
    time_d = np.zeros_like(data)
    time_d[1:-1] = (data[2:] - data[:-2]) / (2 * sampling_interval)
    return time_d
a_vel_rankle = time_d(angle_Rankle, sampling_interval)

# find angular power of ankle
angular_power_Rank = torque_Rank * a_vel_rankle 

# impluse = time_d(power)

# example 
# linear impluse of Rankle 
linear_impluse_Rankle = time_d(linear_power_Rank[1:-1], sampling_interval)
zero_row = np.zeros((1, 3))
# add zero to the first frame and final frame to keep the same frame 
linear_impluse_Rankle = np.vstack([zero_row, linear_impluse_Rankle, zero_row])


# segment power = power_proximal_joint - power_distal_joint 
angle_Rknee = filtered_data['angle_Rshank'] - filtered_data['angle_Rthigh'] 
def time_d(data, sampling_interval):
    time_d = np.zeros_like(data)
    time_d[1:-1] = (data[2:] - data[:-2]) / (2 * sampling_interval)
    return time_d
vel_Rknee = time_d(filtered_data['jc_Rknee'], sampling_interval)
a_vel_Rknee = time_d(angle_Rknee, sampling_interval)

# knee power 
linear_power_Rknee = force_Rknee * vel_Rknee 
angular_power_Rknee = torque_Rknee * a_vel_Rknee 
shank_power = (linear_power_Rknee + angular_power_Rknee) - (linear_power_Rank + angular_power_Rank)











