import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# use your data file path
file_path = '/Users/kairenzheng/Desktop/example of code/step1_gap fill method_interpolation/example data for demo.csv'
df = pd.read_csv(file_path, header=None)

# using golf club marker for an example (have some gaps)
club_x = df.iloc[3:, 3].astype(float)  
club_y = df.iloc[3:, 4].astype(float)  
club_z = df.iloc[3:, 5].astype(float)  

# pick up a time 
time = df.iloc[3:, 2].astype(float) 

# show the picture before gap filling 
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time, club_x, label='club_x', color='blue', antialiased=True)
plt.plot(time, club_y, label='club_y', color='orange', antialiased=True)
plt.plot(time, club_z, label='club_z', color='red', antialiased=True)
plt.title('Marker with gaps', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position (cm)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# gap fill method (interpolation)
def interpolate_with_fallback(data):
    data = pd.DataFrame(data)
    data.replace(0, np.nan, inplace=True)
    data = data.interpolate(method='linear', axis=0)
    data.fillna(method='bfill', inplace=True)  
    data.fillna(method='ffill', inplace=True)  
    if data.isnull().values.any() or (data == 0).any().any():
        data = data.interpolate(method='polynomial', order=2, axis=0).fillna(method='bfill').fillna(method='ffill')
    return data.values

# apply for the data
gap_fill_club_x = interpolate_with_fallback(club_x)
gap_fill_club_y = interpolate_with_fallback(club_y)
gap_fill_club_z = interpolate_with_fallback(club_z)

# show the picture after gap filling 
plt.figure(figsize=(8, 6), dpi=300)
plt.plot(time, gap_fill_club_x, label='club_x', color='blue', antialiased=True)
plt.plot(time, gap_fill_club_y, label='club_y', color='orange', antialiased=True)
plt.plot(time, gap_fill_club_z, label='club_z', color='red', antialiased=True)
plt.title('Marker after gap filling', fontsize=16)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Position (cm)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
