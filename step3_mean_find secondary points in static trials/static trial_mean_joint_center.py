import numpy as np
import pandas as pd

# the lab setting
#   x = anterio-posterior axis 
#   y = bilateral axis
#   z = polar axis 

# use your data file path
file_path = '/Users/kairenzheng/Desktop/example of code/mean_find secondary points in static trials/example_data.csv'
df = pd.read_csv(file_path, header=None)

# calaulate average of each maker (you can use loop)
static_RASIS_ave = np.mean(df.iloc[3:, 1:4].astype(float), axis=0).to_numpy()
static_ＬASIS_ave = np.mean(df.iloc[3:, 4:7].astype(float), axis=0).to_numpy()
static_RPSIS_ave = np.mean(df.iloc[3:, 7:10].astype(float), axis=0).to_numpy()
static_LＰSIS_ave = np.mean(df.iloc[3:, 10:13].astype(float), axis=0).to_numpy()
static_RGT_ave = np.mean(df.iloc[3:, 13:16].astype(float), axis=0).to_numpy()
static_LGT_ave = np.mean(df.iloc[3:, 16:19].astype(float), axis=0).to_numpy()
static_RAshoulder_ave = np.mean(df.iloc[3:, 19:22].astype(float), axis=0).to_numpy()
static_LAshoulder_ave = np.mean(df.iloc[3:, 22:25].astype(float), axis=0).to_numpy()

# hip point calculation (only use in dynamic data)
# Tylkowski Method (only use dynamic data):
# step 1. = measure W (LASIS - RASIS) (bilateral axis direction)
# step 2. = RASIS y (anterio-posterior axis direction) + 0.14 * W / LASIS y (anterio-posterior axis direction) + 0.14 * W 
# step 3. = RASIS z (polar axis direction) - 0.3 * W              / LASIS z (polar axis direction) - 0.3 * W  
# step 4. = RASIS x (bilater axis direction) - 0.19 * W           / LASIS x (bilater axis direction) - 0.19 * W   

w = static_ＬASIS_ave[1] - static_RASIS_ave[1] # 1 = y
rhip_y = static_RASIS_ave[1] + 0.14 * w # 1 = y
rhip_z = static_RASIS_ave[2] - 0.3 * w # 2 = z
rhip_x = static_RASIS_ave[0] - 0.19 * w # 0 = x

lhip_y = static_LASIS_ave[1] - 0.14 * w # 1 = y
lhip_z = static_LASIS_ave[2] - 0.3 * w # 2 = z
lhip_x = static_LASIS_ave[0] - 0.19 * w # 0 = x


# Tylkowski-Andriacchi (T-A) Method:
# step 1. = measure W (LASIS - RASIS)
# step 2. = RASIS y (anterio-posterior axis direction) + 0.14 * W / LASIS y (anterio-posterior axis direction) + 0.14 * W 
# step 3. = RASIS z (polar axis direction) - 0.3 * W              / LASIS z (polar axis direction) - 0.3 * W  
# step 4. = RASIS x = RGT X         / LASIS = LGT X

w = static_LASIS_ave[1] - static_RASIS_ave[1] # 1 = y
rhip_y = static_RASIS_ave[1] + 0.14 * w # 1 = y
rhip_z = static_RASIS_ave[2] - 0.3 * w # 2 = z
rhip_x = static_RGT_ave[0]  # 0 = x

lhip_y = static_LASIS_ave[1] - 0.14 * w # 1 = y
lhip_z = static_LASIS_ave[2] - 0.3 * w # 2 = z
lhip_x = static_LGT_ave[0]  # 0 = x

# mid point method
# example: shoulder joint center -> you can apply for almost shoulder, elbow, wrist, knee, ankle joints
Rshoulder_joint_center = (static_RAshoulder_ave + static_LAshoulder_ave) / 2

