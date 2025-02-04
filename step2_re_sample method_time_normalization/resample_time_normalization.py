import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Use your data file path 
file_path = '/Users/kairenzheng/Desktop/example of code/step1_gap fill method_interpolation/example data for demo.csv'
df = pd.read_csv(file_path, header=None)

# pick up data and time 
data = df.iloc[3:, 3:6].astype(float).to_numpy()
time = df.iloc[3:, 2].astype(float).to_numpy()  # 提取時間資料

# Resampling function (time normlization function)
def time_normalize(data, target_length):
    original_length = data.shape[0]
    x_original = np.linspace(0, 1, original_length)
    x_target = np.linspace(0, 1, target_length)
    re_sample_data = interp1d(x_original, data, axis=0, kind='linear')(x_target)
    return re_sample_data

# Example: target length = 101 frames (you can decide the target length)
target_length = 101 # you will change the number 
re_sample_data = time_normalize(data, target_length)
re_sample_time = time_normalize(time, target_length)

# see the results = after resampling -> data will have 101 frames 
print(data.shape) # will be (n, 3)
print(time.shape) # will be (n, )
print(re_sample_data.shape) # after re-sample -> will be (101, 3)
print(re_sample_time.shape) # after re-sample -> will be (101, )
