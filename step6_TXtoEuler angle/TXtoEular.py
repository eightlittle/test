import numpy as np
import pandas as pd

# create transfermation matrix 
# change static secondary points to dynamic data(only if you have static trails)

# use your data file path
file_path = '/Users/kairenzheng/Desktop/example of code/step6_TXtoEuler angle/example data .csv'

df = pd.read_csv(file_path, header=None)

# dynamic data - pick up pelivs markers
RASIS = df.iloc[3:, 1:4].astype(float).to_numpy()
LASIS = df.iloc[3:, 4:7].astype(float).to_numpy()
RPSIS = df.iloc[3:, 7:10].astype(float).to_numpy() 
LPSIS = df.iloc[3:, 10:13].astype(float).to_numpy()
time = df.iloc[3:, 0].astype(float).to_numpy()

# make mid PSIS for creating pelvis transformation matrix 
m_PSIS = (RPSIS + LPSIS) / 2

#############################################################################
# change the global frame (no need to do it, it is just for TWU lab)
""" Dr Kwon method turning z vector of the global frame -90 degree""" 
global_frame = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
trun = -np.pi / 2
z_90 = np.array([[np.cos(trun), -np.sin(trun), 0],
                 [np.sin(trun), np.cos(trun), 0],
                 [0, 0, 1]])

# Apply the rotation to the global frame
new_global_frame = np.dot(z_90, global_frame)

# Apply the transformation to each row of pelvis_TX
RASIS = np.dot(RASIS, new_global_frame)
LASIS = np.dot(LASIS, new_global_frame)
RPSIS = np.dot(RPSIS, new_global_frame)
LPSIS = np.dot(LPSIS, new_global_frame)
m_PSIS = np.dot(m_PSIS, new_global_frame)
#############################################################################


# transformation matrix to Euler angle 
# provide examples for making transformation matrix in the end 

# create dyanmic TX of pelvis - check step4 
def dy_pel_TX(point1, point2, point3):
    x = point1 - point2 # distance from point 1 to point 2 
    f = point3 - point2 # distance from point 3 to point 2 
    z = np.cross(f, x)
    y = np.cross(z , x)

    # make an empty matrix to save the output
    tx_dynamic = np.zeros((x.shape[0], 3, 3))
    
    for i in range(x.shape[0]):
    # loop for each frame in dynamic data
        tx_dynamic[i, :, :] = np.vstack((x[i, :] / np.linalg.norm(x[i, :]),
                                          y[i, :] / np.linalg.norm(y[i, :]),
                                          z[i, :] / np.linalg.norm(z[i, :])))
    return tx_dynamic

# pelvis TX in dynamic trail 
pelvis_TX = dy_pel_TX(RASIS, LASIS, m_PSIS)

#%%
# make 12 sequences 
# https://en.wikipedia.org/wiki/Euler_angles here are 12 sequences from rotational matrix 
# transformation matrix = transfer of rotational matrix 
# OA = Euler angle 
def TXtoAngle(matrix, order):
    """
    input1 = matrix
    input2 = sequence for rotation 
    outcome = euler angle
    """
    OA = np.zeros([len(matrix), 3])
    for i in range(len(matrix)):
        t11, t12, t13 = matrix[i, 0]
        t21, t22, t23 = matrix[i, 1]
        t31, t32, t33 = matrix[i, 2]
        
        if order == 'xyz':
            theta2_1 = np.arcsin(t31)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(-t32/cos_2, t33/cos_2)
                theta3 = np.arctan2(-t21/cos_2, t11/cos_2)
    
            else:
                theta2 = theta2_2
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(-t32/cos_2, t33/cos_2)
                theta3 = np.arctan2(-t21/cos_2, t11/cos_2)

        elif order == 'xzy':
            theta2_1 = np.arcsin(-t21)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(t23/cos_2, t22/cos_2)
                theta3 = np.arctan2(t31/cos_2, t11/cos_2)
            else:
                theta2 = theta2_2
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(-t23/cos_2, t22/cos_2)
                theta3 = np.arctan2(t31/cos_2, t11/cos_2)
                
        elif order == 'xyx':
            theta2_1 = np.arccos(t11)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t12/sin_2, -t13/sin_2)
                theta3 = np.arctan2(t21/sin_2, t31/sin_2)
            else:
                theta2 = theta2_2
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t12/sin_2, -t13/sin_2)
                theta3 = np.arctan2(t21/sin_2, t31/sin_2)
                
        elif order == 'xzx':
            theta2_1 = np.arccos(t11)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = -theta2_1
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t13/sin_2, t12/sin_2)
                theta3 = np.arctan2(t31/sin_2, -t21/sin_2)
            else:
                theta2 = theta2_2
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t13/sin_2, t12/sin_2)
                theta3 = np.arctan2(t31/sin_2, -t21/sin_2)   
                
        elif order == 'yxz':
            theta2_1 = np.arcsin(-t32)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(t31/cos_2, t33/cos_2)
                theta3 = np.arctan2(t12/cos_2, t22/cos_2)
            else:
                theta2 = theta2_2
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(t31/cos_2, t33/cos_2)
                theta3 = np.arctan2(t12/cos_2, t22/cos_2)             
                
        elif order == 'yzx':
            theta2_1 = np.arcsin(t12)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(-t13/cos_2, t11/cos_2)
                theta3 = np.arctan2(-t32/cos_2, t22/cos_2)
            else:
                theta2 = theta2_2
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(-t13/cos_2, t11/cos_2)
                theta3 = np.arctan2(-t32/cos_2, t22/cos_2)  

        elif order == 'yxy':
            theta2_1 = np.arccos(t22)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = -theta2_1
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t21/sin_2, t23/sin_2)
                theta3 = np.arctan2(t12/sin_2, -t32/sin_2)
            else:
                theta2 = theta2_2
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t21/sin_2, t23/sin_2)
                theta3 = np.arctan2(t12/sin_2, -t32/sin_2) 

        elif order == 'yzy':
            theta2_1 = np.arccos(t22)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = -theta2_1
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t23/sin_2, -t21/sin_2)
                theta3 = np.arctan2(t32/sin_2, t12/sin_2)
            else:
                theta2 = theta2_2
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t23/sin_2, -t21/sin_2)
                theta3 = np.arctan2(t32/sin_2, t12/sin_2)
                
        elif order == 'zxy':
            theta2_1 = np.arcsin(t23)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(-t21/cos_2, t22/cos_2)
                theta3 = np.arctan2(-t13/cos_2, t33/cos_2)
            else:
                theta2 = theta2_2
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(-t21/cos_2, t22/cos_2)
                theta3 = np.arctan2(-t13/cos_2, t33/cos_2)
                
        elif order == 'zyx':
            theta2_1 = np.arcsin(-t13)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(t12/cos_2, t11/cos_2)
                theta3 = np.arctan2(t23/cos_2, t33/cos_2)
            else:
                theta2 = theta2_2
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(t12/cos_2, t11/cos_2)
                theta3 = np.arctan2(t23/cos_2, t33/cos_2)
                
        elif order == 'zxz':
            theta2_1 = np.arccos(t33)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = -theta2_1
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t31/sin_2, -t32/sin_2)
                theta3 = np.arctan2(t13/sin_2, t23/sin_2)
            else:
                theta2 = theta2_2
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t31/sin_2, -t32/sin_2)
                theta3 = np.arctan2(t13/sin_2, t23/sin_2)
        elif order == 'zyz':
            theta2_1 = np.arccos(t33)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t32/sin_2, t31/sin_2)
                theta3 = np.arctan2(t23/sin_2, -t13/sin_2)
            else:
                theta2 = theta2_1
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t32/sin_2, t31/sin_2)
                theta3 = np.arctan2(t23/sin_2, -t13/sin_2)
        
        OA[i, :] = np.matrix([theta1, theta2, theta3])
    return OA

# transfer transfermotion matrix to angle (unit: rad )
pelvis_angle = TXtoAngle(pelvis_TX, 'xyz')

# transfer transfermotion matrix to angle (unit: rad )
pelvis_angle2= TXtoAngle(pelvis_TX, 'xzx')


import matplotlib.pyplot as plt
# make a picture (unit changes to degree)
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time, pelvis_angle[:, 0]*(180/np.pi), label='pelvis_x', color='blue', antialiased=True)
plt.plot(time, pelvis_angle[:, 1]*(180/np.pi), label='pelvis_y', color='orange', antialiased=True)
plt.plot(time, pelvis_angle[:, 2]*(180/np.pi), label='pelvis_z', color='red', antialiased=True)
plt.title('Pelvis angle position_xyz order', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position (cm)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# make a picture (unit changes to degree) -> have gimbal lock problem
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time, pelvis_angle2[:, 0]*(180/np.pi), label='pelvis_x', color='blue', antialiased=True)
plt.plot(time, pelvis_angle2[:, 1]*(180/np.pi), label='pelvis_y', color='orange', antialiased=True)
plt.plot(time, pelvis_angle2[:, 2]*(180/np.pi), label='pelvis_z', color='red', antialiased=True)
plt.title('Pelvis angle position_xzx order (problem)', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position (cm)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# fix gimbal lock problem 
# using the diff
def unwrap_deg(data):
    """
    input = data
    outcome = data without gimbal lock problem
    """
    # Calculate the difference between consecutive data points (angle changes)
    dp = np.diff(data)
    # Adjust the differences to be within the range of -π to π
    # First, add π to dp, then take the modulus with 2π, and subtract π to bring the angle change within the range of -π to π
    dps = np.mod(dp + np.pi, 2 * np.pi) - np.pi
    # Handle special case: when the difference is -π, and the original change was positive, fix it to π
    dps[np.logical_and(dps == -np.pi, dp > 0)] = np.pi
    # Calculate the correction needed (difference between the adjusted angle change and the original angle change)
    dp_corr = dps - dp
    # For angle changes that are smaller than π, we set the correction to 0 (no need to fix)
    dp_corr[np.abs(dp) < np.pi] = 0
    # Accumulate the corrections into the original data starting from the second data point
    data[1:] += np.cumsum(dp_corr)
    # Return the corrected data
    return data

# Unwrap pelvis angles (in degrees) 
new_pelvis_angle2_x = unwrap_deg(pelvis_angle2[:, 0])  # Apply unwrap directly to 1D arrays
new_pelvis_angle2_y = unwrap_deg(pelvis_angle2[:, 1])  # Apply unwrap directly to 1D arrays
new_pelvis_angle2_z = unwrap_deg(pelvis_angle2[:, 2])  # Apply unwrap directly to 1D arrays

#Plot the unwrapped angles
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time, new_pelvis_angle2_x, label='pelvis_x', color='blue', antialiased=True)
plt.plot(time, new_pelvis_angle2_y, label='pelvis_y', color='orange', antialiased=True)
plt.plot(time, new_pelvis_angle2_z, label='pelvis_z', color='red', antialiased=True)
plt.title('Pelvis angle position_xzx order fixed problem', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position (degrees)', fontsize=12)  # Corrected to 'degrees'
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

#%%
# example to 12 orders for pelvis rotation from transformation matrix to Euler angle 
# need to pick up one with meaningful ( if no, usually use XYZ sequence )
import numpy as np
import matplotlib.pyplot as plt

def TXtoAngle(matrix, order):
    """
    input1 = matrix
    input2 = sequence for rotation 
    outcome = euler angle
    """
    OA = np.zeros([len(matrix), 3])
    for i in range(len(matrix)):
        t11, t12, t13 = matrix[i, 0]
        t21, t22, t23 = matrix[i, 1]
        t31, t32, t33 = matrix[i, 2]
        
        if order == 'xyz':
            theta2_1 = np.arcsin(t31)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(-t32/cos_2, t33/cos_2)
                theta3 = np.arctan2(-t21/cos_2, t11/cos_2)
    
            else:
                theta2 = theta2_2
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(-t32/cos_2, t33/cos_2)
                theta3 = np.arctan2(-t21/cos_2, t11/cos_2)

        elif order == 'xzy':
            theta2_1 = np.arcsin(-t21)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(t23/cos_2, t22/cos_2)
                theta3 = np.arctan2(t31/cos_2, t11/cos_2)
            else:
                theta2 = theta2_2
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(-t23/cos_2, t22/cos_2)
                theta3 = np.arctan2(t31/cos_2, t11/cos_2)
                
        elif order == 'xyx':
            theta2_1 = np.arccos(t11)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t12/sin_2, -t13/sin_2)
                theta3 = np.arctan2(t21/sin_2, t31/sin_2)
            else:
                theta2 = theta2_2
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t12/sin_2, -t13/sin_2)
                theta3 = np.arctan2(t21/sin_2, t31/sin_2)
                
        elif order == 'xzx':
            theta2_1 = np.arccos(t11)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = -theta2_1
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t13/sin_2, t12/sin_2)
                theta3 = np.arctan2(t31/sin_2, -t21/sin_2)
            else:
                theta2 = theta2_2
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t13/sin_2, t12/sin_2)
                theta3 = np.arctan2(t31/sin_2, -t21/sin_2)   
                
        elif order == 'yxz':
            theta2_1 = np.arcsin(-t32)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(t31/cos_2, t33/cos_2)
                theta3 = np.arctan2(t12/cos_2, t22/cos_2)
            else:
                theta2 = theta2_2
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(t31/cos_2, t33/cos_2)
                theta3 = np.arctan2(t12/cos_2, t22/cos_2)             
                
        elif order == 'yzx':
            theta2_1 = np.arcsin(t12)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(-t13/cos_2, t11/cos_2)
                theta3 = np.arctan2(-t32/cos_2, t22/cos_2)
            else:
                theta2 = theta2_2
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(-t13/cos_2, t11/cos_2)
                theta3 = np.arctan2(-t32/cos_2, t22/cos_2)  

        elif order == 'yxy':
            theta2_1 = np.arccos(t22)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = -theta2_1
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t21/sin_2, t23/sin_2)
                theta3 = np.arctan2(t12/sin_2, -t32/sin_2)
            else:
                theta2 = theta2_2
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t21/sin_2, t23/sin_2)
                theta3 = np.arctan2(t12/sin_2, -t32/sin_2) 

        elif order == 'yzy':
            theta2_1 = np.arccos(t22)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = -theta2_1
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t23/sin_2, -t21/sin_2)
                theta3 = np.arctan2(t32/sin_2, t12/sin_2)
            else:
                theta2 = theta2_2
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t23/sin_2, -t21/sin_2)
                theta3 = np.arctan2(t32/sin_2, t12/sin_2)
                
        elif order == 'zxy':
            theta2_1 = np.arcsin(t23)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(-t21/cos_2, t22/cos_2)
                theta3 = np.arctan2(-t13/cos_2, t33/cos_2)
            else:
                theta2 = theta2_2
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(-t21/cos_2, t22/cos_2)
                theta3 = np.arctan2(-t13/cos_2, t33/cos_2)
                
        elif order == 'zyx':
            theta2_1 = np.arcsin(-t13)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(t12/cos_2, t11/cos_2)
                theta3 = np.arctan2(t23/cos_2, t33/cos_2)
            else:
                theta2 = theta2_2
                cos_2 = np.cos(theta2)
                theta1 = np.arctan2(t12/cos_2, t11/cos_2)
                theta3 = np.arctan2(t23/cos_2, t33/cos_2)
                
        elif order == 'zxz':
            theta2_1 = np.arccos(t33)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = -theta2_1
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t31/sin_2, -t32/sin_2)
                theta3 = np.arctan2(t13/sin_2, t23/sin_2)
            else:
                theta2 = theta2_2
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t31/sin_2, -t32/sin_2)
                theta3 = np.arctan2(t13/sin_2, t23/sin_2)
        elif order == 'zyz':
            theta2_1 = np.arccos(t33)
            theta2_2 = np.pi - theta2_1
            if abs(theta2_1) < np.pi/2 or np.pi/2 < abs(theta2_1) < np.pi:
                theta2 = theta2_1
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t32/sin_2, t31/sin_2)
                theta3 = np.arctan2(t23/sin_2, -t13/sin_2)
            else:
                theta2 = theta2_1
                sin_2 = np.sin(theta2)
                theta1 = np.arctan2(t32/sin_2, t31/sin_2)
                theta3 = np.arctan2(t23/sin_2, -t13/sin_2)
        
        OA[i, :] = np.matrix([theta1, theta2, theta3])
    return OA


# List of all 12 rotation sequences
rotation_orders = ['xyz', 'xzy', 'xyx', 'xzx', 'yxz', 'yzx', 'yxy', 'yzy', 'zxy', 'zyx', 'zxz', 'zyz']

# Create subplots to show all 12 rotations
fig, axes = plt.subplots(4, 3, figsize=(12, 16), dpi=300)

# Flatten the axes array to loop over them
axes = axes.flatten()

# Loop over the rotation orders and plot the corresponding Euler angles
for i, order in enumerate(rotation_orders):
    # Compute the Euler angles for the current rotation order
    pelvis_angle = TXtoAngle(pelvis_TX, order)
    
    # Convert to degrees for better readability
    pelvis_angle_deg = pelvis_angle * (180 / np.pi)
    
    # Plot each angle (x, y, z) in the corresponding subplot
    axes[i].plot(time, pelvis_angle_deg[:, 0], label='pelvis_x', color='blue')
    axes[i].plot(time, pelvis_angle_deg[:, 1], label='pelvis_y', color='orange')
    axes[i].plot(time, pelvis_angle_deg[:, 2], label='pelvis_z', color='red')
    
    # Set title and labels
    axes[i].set_title(f'Rotation Order: {order}', fontsize=12)
    axes[i].set_xlabel('Time (s)', fontsize=10)
    axes[i].set_ylabel('Angle (degrees)', fontsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.7)
    axes[i].legend(loc='upper right')

# Adjust layout to avoid overlap
plt.tight_layout()
plt.show()

#%%
# joint angles = distal segment angle - proximal segment angle 
# example 
# Rhip_angle = Rthigh - pelvis
# Lhip_angle = Lthigh - pelvis
# Rknee_angle = Rsha - Rthigh
# Lknee_angle = Lsha - Lthigh
# Rank_angle = Rft - Rsha
# Lank_angle = Lft - Lsha
# Ltrunk_angle = abdomen - pelvis
# Utrunk_angle = thorax - abdomen
# Rsho_angle = RUA - thorax
# Lsho_angle = LUA - thorax
# Relb_angle = RFA - RUA
# Lelb_angle = LFA - LUA
