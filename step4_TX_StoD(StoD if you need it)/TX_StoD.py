import numpy as np
import pandas as pd

# the lab setting
# x is bilateral direction
# y is anterio-posterior direction
# z is vertical direction 

# create transfermation matrix 
# change static secondary points to dynamic data(only if you have static trails)

# open your file path 
file_path = '/Users/kairenzheng/Desktop/example of code/step4_TX_StoD/example data .csv'

df = pd.read_csv(file_path, header=None)

# dynamic data
RASIS = df.iloc[3:, 1:4].astype(float).to_numpy()
LASIS = df.iloc[3:, 4:7].astype(float).to_numpy()
RPSIS = df.iloc[3:, 7:10].astype(float).to_numpy() 
LPSIS = df.iloc[3:, 10:13].astype(float).to_numpy()
time = df.iloc[2:, 0].astype(float).to_numpy()

# find the original point of pelvis in dyanmic data 
mid_pelvis = (RASIS + LASIS + RPSIS + LPSIS) / 4


# static trial (mean)
static_RASIS= np.mean(df.iloc[3:, 15:18].astype(float), axis=0).to_numpy()
static_LASIS= np.mean(df.iloc[3:, 18:21].astype(float), axis=0).to_numpy()
static_RPSIS= np.mean(df.iloc[3:, 21:24].astype(float), axis=0).to_numpy()
static_LPSIS= np.mean(df.iloc[3:, 24:27].astype(float), axis=0).to_numpy()
static_RGT= np.mean(df.iloc[3:, 27:30].astype(float), axis=0).to_numpy()
static_LGT= np.mean(df.iloc[3:, 30:33].astype(float), axis=0).to_numpy()

# find the original point of pelvis in static data 
static_mid_pelvis = (static_RASIS + static_LASIS + static_RPSIS + static_LPSIS) / 4 

# find hip joints in static trial
# find right hip
w = static_LASIS[1] - static_RASIS[1] # 1 = y
rhip_y = static_RASIS[1] + 0.14 * w # 1 = y
rhip_z = static_RASIS[2] - 0.3 * w # 2 = z
rhip_x = static_RGT[0]  # 0 = x
rhip_point = np.vstack((rhip_x, rhip_y, rhip_z))

# find left hip
lhip_y = static_LASIS[1] - 0.14 * w # 1 = y
lhip_z = static_LASIS[2] - 0.3 * w # 2 = z
lhip_x = static_LGT[0]  # 0 = x
lhip_point = np.vstack((lhip_x, lhip_y, lhip_z))

# build transformation matrix in static data
def sta_TX (point1, point2, point3):
    x = point1 - point2 # distance from point 1 to point 2 
    f = point3 - point2 # distance from point 3 to point 2 
    z = np.cross(f, x) # cross two vector to have a perpendicular vector
    y = np.cross(z , x) # cross two vector to have a perpendicular vector
    s_x, s_y, s_z = x / np.linalg.norm(x), y / np.linalg.norm(y), z / np.linalg.norm(z)
    # the vectors in a transformation matrix are unit vectors
    s_TX = np.vstack((s_x, s_y, s_z))
    # combine three vectors to be a three direction matrix 
    return s_TX

# build transformation matrix in static data (each segment will be different)
m_PSIS_static = (static_RPSIS + static_LPSIS) / 2

# pelvis TX in static trial (input: point1, point2, point3) -> order of points are important
pelvis_TX_static = sta_TX(static_RASIS.T, static_LASIS.T, m_PSIS_static.T)

# provide examples for making transformation matrix in the end 
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

m_PSIS = (RPSIS + LPSIS) / 2
# pelvis TX in dynamic trail 
pelvis_TX = dy_pel_TX(RASIS, LASIS, m_PSIS)




# make a picture of checking if three directions are correct
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


arrow_scale = 100 # adjust array size 
ax.scatter(static_mid_pelvis[0], static_mid_pelvis[1], static_mid_pelvis[2], 
           c='r', marker='x', label='mid pelvis')
ax.quiver(static_mid_pelvis[0], static_mid_pelvis[1], static_mid_pelvis[2], 
          pelvis_TX_static[0 , 0] * arrow_scale, pelvis_TX_static[0 , 1] * arrow_scale, pelvis_TX_static[0, 2] * arrow_scale, 
          color='r', label='pelvis X axis')
ax.quiver(static_mid_pelvis[0], static_mid_pelvis[1], static_mid_pelvis[2], 
          pelvis_TX_static[1 , 0] * arrow_scale, pelvis_TX_static[1 , 1] * arrow_scale, pelvis_TX_static[1, 2] * arrow_scale, 
          color='g', label='pelvis Y axis')
ax.quiver(static_mid_pelvis[0], static_mid_pelvis[1], static_mid_pelvis[2], 
          pelvis_TX_static[2 , 0] * arrow_scale, pelvis_TX_static[2 , 1] * arrow_scale, pelvis_TX_static[2, 2] * arrow_scale,  
          color='b', label='pelvis Z axis')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.set_xlim(-50, 200)
ax.set_ylim(-50, 200)
ax.set_zlim(-50, 200)

ax.set_box_aspect([1, 1, 1])

plt.title('3D Plot of pelvis TX')
plt.legend()
plt.show()


#%%
# change points from static to dyanmic trial (only if you have static trial)
import numpy as np

def StoD(sta_o_SIS, p, s_TX, d_o_SIS, dy_frame):
    """
    Convert static joint center to dynamic joint center.
    
    Parameters:
    sta_o_SIS : ndarray (3,)  -> Static origin in global frame
    p : ndarray (3,1)         -> Joint center in static trial
    s_TX : ndarray (3,3)      -> Static transformation matrix
    d_o_SIS : ndarray (n,3)   -> Dynamic origin in global frame
    dy_frame : ndarray (n,3,3) -> Dynamic transformation matrices per frame
    
    Returns:
    glob_point : ndarray (n,3) -> Joint center in dynamic trial
    """
    # 
    p = p.reshape((3,))  # checking the shape of the data
    distance = p - sta_o_SIS  # distance from the point to the local origianl point

    # change to the static transformation matrix (example: pelvis transformation matrix)
    local_point = np.dot(s_TX, distance).reshape((3,))  
    
    
    frames = len(d_o_SIS)
    glob_point = np.zeros((frames, 3))
    for i in range(frames):
        # change to the global tranformation matrix 
        #  add the distance from local original point to the global original point
        glob_point[i, :] = np.dot(dy_frame[i, :, :].T, local_point) + d_o_SIS[i, :]
    return glob_point

# check the data shape
static_mid_pelvis = np.array(static_mid_pelvis).reshape((3,)) 
rhip_point = np.array(rhip_point).reshape((3,1))  
lhip_point = np.array(lhip_point).reshape((3,1)) 

# apply for the data
d_jc_Rhip_global = StoD(static_mid_pelvis, rhip_point, pelvis_TX_static, mid_pelvis, pelvis_TX)
d_jc_Lhip_global = StoD(static_mid_pelvis, lhip_point, pelvis_TX_static, mid_pelvis, pelvis_TX)

#%%
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(RASIS[1, 0], RASIS[1, 1], RASIS[1, 2], c='m', label='RASIS')
ax.scatter(LASIS[1, 0], LASIS[1, 1], LASIS[1, 2], c='y', label='LASIS')
ax.scatter(RPSIS[1, 0], RPSIS[1, 1], RPSIS[1, 2], c='c', label='RPSIS')
ax.scatter(LPSIS[1, 0], LPSIS[1, 1], LPSIS[1, 2], c='r', label='LPSIS')
ax.scatter(d_jc_Rhip_global[1, 0], d_jc_Rhip_global[1, 1], d_jc_Rhip_global[1, 2], 
           c='b', marker='x', label='Right Hip JC')
ax.scatter(d_jc_Lhip_global[1, 0], d_jc_Lhip_global[1, 1], d_jc_Lhip_global[1, 2], 
           c='g', marker='x', label='Left Hip JC')

pelvis_points = np.array([RASIS[1], LASIS[1], LPSIS[1], RPSIS[1], RASIS[1]])  # 回到RASIS形成封閉結構
ax.plot(pelvis_points[:, 0], pelvis_points[:, 1], pelvis_points[:, 2], c='k', linestyle='-', linewidth=2, label="Pelvis Outline")

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Joint Centers and Pelvis Landmarks')

ax.legend()
plt.show()

#%%
# the example for making transformation matrix for each segment 
def find_JC_and_dy_TX(combined_data_dy, mean_sta, secondary_points_sta): 
    d_jc_Rwri = (combined_data_dy["Golfer:RWrist"]  + combined_data_dy["Golfer:RMWrist"]) / 2
    d_jc_Lwri = (combined_data_dy["Golfer:LWrist"]  + combined_data_dy["Golfer:LMWrist"]) / 2
    d_jc_Rank = (combined_data_dy["Golfer:RAnkle"] + combined_data_dy["Golfer:RMAnkle"]) / 2
    d_jc_Lank = (combined_data_dy["Golfer:LAnkle"] + combined_data_dy["Golfer:LMAnkle"]) / 2
    d_jc_Utrunk = (combined_data_dy["Golfer:C7"] + combined_data_dy["Golfer:SN"]) / 2
    d_jc_Mtrunk = (combined_data_dy["Golfer:T5"] + combined_data_dy["Golfer:XI"]) / 2
    d_mid_head = (combined_data_dy["Golfer:HeadR"] + combined_data_dy["Golfer:HeadL"]) / 2
    
# find static transformation matrix
def sta_TX (point1, point2, point3):
    x = point1 - point2
    f = point3 - point2
    z = np.cross(f, x)
    y = np.cross(z , x)
    s_x, s_y, s_z = x / np.linalg.norm(x), y / np.linalg.norm(y), z / np.linalg.norm(z)
    s_TX = np.vstack((s_x, s_y, s_z))
    return s_TX

# find dynamic transformation matrix (complex)
# make pelvis TX
# point 1 = RASIS
# point 2 = LASIS
# point 3 = mid of PSIS
def dy_pel_TX(point1, point2, point3):
    x = point1 - point2
    f = point3 - point2
    z = np.cross(f, x)
    y = np.cross(z , x)

    pelvis_TX = np.zeros((x.shape[0], 3, 3))
    for i in range(x.shape[0]):
        pelvis_TX[i, :, :] = np.vstack((x[i, :] / np.linalg.norm(x[i, :]),
                                          y[i, :] / np.linalg.norm(y[i, :]),
                                          z[i, :] / np.linalg.norm(z[i, :])))
    return pelvis_TX

# examples
# d_mPSIS = (combined_data_dy["Golfer:RPSIS"] + combined_data_dy["Golfer:LPSIS"]) / 2
# d_pel_TX = dy_pel_TX(combined_data_dy["Golfer:RASIS"], combined_data_dy["Golfer:LASIS"], d_mPSIS)

# find limbs trandformation matrix
# point 1 = z axis (faraway from original point)
# point 2 = original point
# point 3 = point to create fake axis 
def dy_limbs_TX(point1, point2, point3, RorL):
    if RorL == 'R':
        z = point1 - point2
        f = point3 - point2
        y = np.cross(z, f)
        x = np.cross(y , z)

        limb_TX = np.zeros((x.shape[0], 3, 3))
        for i in range(x.shape[0]):
            limb_TX[i, :, :] = np.vstack((x[i, :] / np.linalg.norm(x[i, :]),
                                              y[i, :] / np.linalg.norm(y[i, :]),
                                              z[i, :] / np.linalg.norm(z[i, :])))
    elif RorL == 'L':
        f = point1 - point2
        z = point3 - point2
        y = np.cross(f, z)
        x = np.cross(y , z)

        limb_TX = np.zeros((x.shape[0], 3, 3))
        for i in range(x.shape[0]):
            limb_TX[i, :, :] = np.vstack((x[i, :] / np.linalg.norm(x[i, :]),
                                              y[i, :] / np.linalg.norm(y[i, :]),
                                              z[i, :] / np.linalg.norm(z[i, :])))
    return limb_TX

# examples
# make thigh TX
# # make R and L Thigh dy_TX
# d_Rthigh_TX = dy_limbs_TX(d_jc_Rhip_global, d_jc_Rknee_global, combined_data_dy["Golfer:RThigh"], 'R')
# d_Lthigh_TX = dy_limbs_TX(combined_data_dy["Golfer:LThigh"], d_jc_Lknee_global, d_jc_Lhip_global, 'L')
# make shank TX
# # make R and L shank dy_TX
# d_Rsha_TX = dy_limbs_TX(d_jc_Rknee_global, d_jc_Rank, combined_data_dy["Golfer:RAnkle"], 'R')
# d_Lsha_TX = dy_limbs_TX(combined_data_dy["Golfer:LAnkle"], d_jc_Lank, d_jc_Lknee_global, 'L')
# make upper arm TX
# # make upper arm dy_TX
# d_RUA_TX = dy_limbs_TX(d_jc_Rsho_global, d_jc_Relb_global, combined_data_dy["Golfer:RUA1"], 'R')
# d_LUA_TX = dy_limbs_TX(combined_data_dy["Golfer:LUA1"], d_jc_Lelb_global, d_jc_Lsho_global, 'L')
# make forearm TX
# # forearm arm d TX
# d_RFA_TX = dy_limbs_TX(d_jc_Relb_global, d_jc_Rwri, combined_data_dy["Golfer:RWrist"], 'R')
# d_LFA_TX = dy_limbs_TX(combined_data_dy["Golfer:LWrist"], d_jc_Lwri, d_jc_Lelb_global, 'L')

# find feet transformation matrix 
# point 1 = toe
# point 2 = heel
# point 3 = ankle 
def dy_ft_TX(point1, point2, point3):
    y = point1 - point2
    f = point3 - point2
    x = np.cross(y, f)
    z = np.cross(x, y)

    ft_TX = np.zeros((x.shape[0], 3, 3))
    for i in range(x.shape[0]):
        ft_TX[i, :, :] = np.vstack((x[i, :] / np.linalg.norm(x[i, :]),
                                          y[i, :] / np.linalg.norm(y[i, :]),
                                          z[i, :] / np.linalg.norm(z[i, :])))
    return ft_TX

# example
# # make R and L foot dy_TX
# d_Rft_TX = dy_ft_TX(combined_data_dy["Golfer:RToe"], combined_data_dy["Golfer:RHeel"], d_jc_Rank)
# d_Lft_TX = dy_ft_TX(combined_data_dy["Golfer:LToe"], combined_data_dy["Golfer:LHeel"], d_jc_Lank)

# find trunk transformation matrix
def dy_trunk_TX(point1, point2, point3, point4):
    z = point1 - point2
    f = point3 - point4
    x = np.cross(f, z)
    y = np.cross(z, x)

    trunk_TX = np.zeros((x.shape[0], 3, 3))
    for i in range(x.shape[0]):
        trunk_TX[i, :, :] = np.vstack((x[i, :] / np.linalg.norm(x[i, :]),
                                          y[i, :] / np.linalg.norm(y[i, :]),
                                          z[i, :] / np.linalg.norm(z[i, :])))
    return trunk_TX

# # make abdomen and thorax dy_TX
# mid_sho = (d_jc_Rsho_global + d_jc_Lsho_global) / 2
# d_abdomen_TX = dy_trunk_TX(d_jc_Mtrunk, d_o_mid_pel, combined_data_dy["Golfer:XI"], combined_data_dy["Golfer:T5"])
# mid_sho = (d_jc_Rsho_global + d_jc_Lsho_global) / 2
# d_thorax_TX = dy_trunk_TX(mid_sho, d_jc_Mtrunk, combined_data_dy["Golfer:SN"], combined_data_dy["Golfer:C7"])
    


# make head dy_TX
def dy_head_TX(point1, point2, point3):
    x = point1 - point2
    y = point3 - point2
    z = np.cross(x, y)

    tx_dynamic = np.zeros((x.shape[0], 3, 3))
    for i in range(x.shape[0]):
        tx_dynamic[i, :, :] = np.vstack((x[i, :] / np.linalg.norm(x[i, :]),
                                          y[i, :] / np.linalg.norm(y[i, :]),
                                          z[i, :] / np.linalg.norm(z[i, :])))
    return tx_dynamic

# make head TX
# d_head_TX = dy_head_TX(combined_data_dy["Golfer:HeadR"], d_mid_head, combined_data_dy["Golfer:HeadA"])

#%%
# example: animation -> need to use visual 3D to have animation (spyder is not working)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# 假設 data_dict 已經包含 (n, 3) 的數據
data_dict = {
    "RASIS": RASIS,
    "LASIS": LASIS,
    "RPSIS": RPSIS,
    "LPSIS": LPSIS,
    "mid_pelvis": mid_pelvis,
    "Rhip": d_jc_Rhip_global,
    "Lhip": d_jc_Lhip_global
}

# 確保 frame 數一致
min_frames = min([data_dict[key].shape[0] for key in data_dict])
num_points = len(data_dict)

# 建立 3D 陣列 (num_points, min_frames, 3)
data = np.zeros([num_points, min_frames, 3])
for idx, key in enumerate(data_dict.keys()):
    data[idx, :, :] = data_dict[key][:min_frames, :]

# 創建 3D 圖
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 計算 X、Y、Z 軸的範圍
x_min, x_max = np.min(data[:, :, 0]), np.max(data[:, :, 0])
y_min, y_max = np.min(data[:, :, 1]), np.max(data[:, :, 1])
z_min, z_max = np.min(data[:, :, 2]), np.max(data[:, :, 2])

# 找出最大範圍，確保所有軸的間距相同
max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0

# 計算軸的中心點
x_mid = (x_max + x_min) / 2.0
y_mid = (y_max + y_min) / 2.0
z_mid = (z_max + z_min) / 2.0

# 設定所有軸的範圍，確保間距相同
ax.set_xlim(x_mid - max_range, x_mid + max_range)
ax.set_ylim(y_mid - max_range, y_mid + max_range)
ax.set_zlim(z_mid - max_range, z_mid + max_range)

# 設置 3D 圖的比例為 1:1:1，避免變形
ax.set_box_aspect([1, 1, 1])

# 初始化散點圖和骨盆線
scatter = ax.scatter([], [], [], c='r', marker='o')
lines, = ax.plot([], [], [], 'k-', linewidth=2)  # 連接骨盆四點的線

# 設置骨盆三軸箭頭
arrow_scale = 10  # 調整箭頭大小
quiver_x = None
quiver_y = None
quiver_z = None

def update(frame):
    global quiver_x, quiver_y, quiver_z

    scatter._offsets3d = (data[:, frame, 0], data[:, frame, 1], data[:, frame, 2])

    # 更新骨盆四點連線
    pelvis_points = np.array([
        data[0, frame, :],  # RASIS
        data[1, frame, :],  # LASIS
        data[3, frame, :],  # LPSIS
        data[2, frame, :],  # RPSIS
        data[0, frame, :]   # 回到 RASIS 閉合線
    ])
    lines.set_data(pelvis_points[:, 0], pelvis_points[:, 1])
    lines.set_3d_properties(pelvis_points[:, 2])

    # 更新骨盆旋轉軸箭頭
    mid_pelvis_pos = mid_pelvis[frame]
    pelvis_rot = pelvis_TX[frame]

    # 如果箭頭變數還未初始化，則先初始化
    if quiver_x is None:
        quiver_x = ax.quiver(mid_pelvis_pos[0], mid_pelvis_pos[1], mid_pelvis_pos[2], 
                              pelvis_rot[0, 0] * arrow_scale, pelvis_rot[0, 1] * arrow_scale, pelvis_rot[0, 2] * arrow_scale, 
                              color='r')
        quiver_y = ax.quiver(mid_pelvis_pos[0], mid_pelvis_pos[1], mid_pelvis_pos[2], 
                              pelvis_rot[1, 0] * arrow_scale, pelvis_rot[1, 1] * arrow_scale, pelvis_rot[1, 2] * arrow_scale, 
                              color='g')
        quiver_z = ax.quiver(mid_pelvis_pos[0], mid_pelvis_pos[1], mid_pelvis_pos[2], 
                              pelvis_rot[2, 0] * arrow_scale, pelvis_rot[2, 1] * arrow_scale, pelvis_rot[2, 2] * arrow_scale,  
                              color='b')
    else:
        # 先移除舊的箭頭
        quiver_x.remove()
        quiver_y.remove()
        quiver_z.remove()

        # 更新箭頭
        quiver_x = ax.quiver(mid_pelvis_pos[0], mid_pelvis_pos[1], mid_pelvis_pos[2], 
                              pelvis_rot[0, 0] * arrow_scale, pelvis_rot[0, 1] * arrow_scale, pelvis_rot[0, 2] * arrow_scale, 
                              color='r')
        quiver_y = ax.quiver(mid_pelvis_pos[0], mid_pelvis_pos[1], mid_pelvis_pos[2], 
                              pelvis_rot[1, 0] * arrow_scale, pelvis_rot[1, 1] * arrow_scale, pelvis_rot[1, 2] * arrow_scale, 
                              color='g')
        quiver_z = ax.quiver(mid_pelvis_pos[0], mid_pelvis_pos[1], mid_pelvis_pos[2], 
                              pelvis_rot[2, 0] * arrow_scale, pelvis_rot[2, 1] * arrow_scale, pelvis_rot[2, 2] * arrow_scale,  
                              color='b')

    return scatter, lines, quiver_x, quiver_y, quiver_z


# 創建動畫
ani = FuncAnimation(fig, update, frames=min_frames, interval=0.01, blit=False)

plt.show()




