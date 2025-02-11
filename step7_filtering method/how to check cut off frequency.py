import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.signal import freqs
from scipy import interpolate

#%%
def read_c3d(path):
    # import library
    import ezc3d
    import numpy as np
    import pandas as pd    
    c = ezc3d.c3d(path)
    motion_information = c['header']['points']
    analog_information = c['header']['analogs']
    motion_axis = ['x', 'y', 'z']
    motion_markers = []
    for marker_name in c['parameters']['POINT']['LABELS']['value']:
        for axis in motion_axis:
            name = marker_name + '_' + axis
            motion_markers.append(name) 
    motion_data = pd.DataFrame(np.zeros([np.shape(c['data']['points'])[-1], # last frame + 1
                                         len(c['parameters']['POINT']['LABELS']['value'])*3]), # marker * 3
                               columns=motion_markers) 
    for i in range(len(c['parameters']['POINT']['LABELS']['value'])):
        motion_data.iloc[:, 1*i*3:1*i*3+3] = np.transpose(c['data']['points'][:3, i, :])
    motion_time = np.linspace(
                                0, # start
                              ((c['header']['points']['last_frame'])/c['header']['points']['frame_rate']), # stop = last_frame/frame_rate
                              num = (np.shape(c['data']['points'])[-1]) # num = last_frame
                              )
    motion_data.insert(0, 'Frame', motion_time)
    FP_channel = c['parameters']['ANALOG']['LABELS']['value']
    FP_data = pd.DataFrame(np.zeros([np.shape(c['data']['analogs'])[-1], # last frame + 1
                                         len(FP_channel)]), 
                               columns=FP_channel)
    FP_data.iloc[:, :] = np.transpose(c['data']['analogs'][0, :, :])
    FP_time = np.linspace(
                                0, # start
                              ((c['header']['analogs']['last_frame'])/c['header']['analogs']['frame_rate']), # stop = last_frame/frame_rate
                              num = (np.shape(c['data']['analogs'])[-1]) # num = last_frame
                              )
    FP_data.insert(0, 'Frame', FP_time)
    return motion_information, motion_data, analog_information, FP_data
dg1_info, dg1_data, analog_info, FP_dg1 = read_c3d(r'/Users/kairenzheng/Desktop/example of code/step7_filtering method/D 2.c3d')
dg1_data = dg1_data/100 # in meter 

# iterpolation
def interpolate(data):
    data = data.interpolate(method ='polynomial',order = 3)
    return data

dg1_data = interpolate(dg1_data)
dg1_data = np.array(dg1_data)

# the point for final exam
data = np.array(dg1_data[:,100:103], dtype=float)


#%%
import numpy as np
import matplotlib.pyplot as plt

def filtering(data, fs, fc_range=30): # can change the number to change the range
    def two_order_for(data, a0, a1, a2, b1, b2):
        forward_f = np.zeros_like(data)
        forward_f[:2] = data[:2]  
        for i in range(2, len(data)):
            forward_f[i] = (a0 * data[i] + a1 * data[i-1] + a2 * data[i-2]
                            - b1 * forward_f[i-1] - b2 * forward_f[i-2])
        return forward_f

    def two_order_back(forward_f, a0, a1, a2, b1, b2):
        backward_f = np.zeros_like(forward_f)
        reversed_f = forward_f[::-1]  
        backward_f[:2] = reversed_f[:2]
        for i in range(2, len(reversed_f)):
            backward_f[i] = (a0 * reversed_f[i] + a1 * reversed_f[i-1] + a2 * reversed_f[i-2]
                             - b1 * backward_f[i-1] - b2 * backward_f[i-2])
        return backward_f[::-1]  

    def total(fs, fc, data):
        fr = fs / fc
        omgc = np.tan(np.pi / fr)
        c = 1 + 2 * np.cos(np.pi / 4) * omgc + omgc**2
        a0 = a2 = omgc**2 / c
        a1 = 2 * a0
        b1 = 2 * (omgc**2 - 1) / c
        b2 = (1 - 2 * np.cos(np.pi / 4) * omgc + omgc**2) / c

        filtered_f = np.column_stack([
            two_order_for(data[:, i], a0, a1, a2, b1, b2) for i in range(3)
        ])
        filtered_b = np.column_stack([
            two_order_back(filtered_f[:, i], a0, a1, a2, b1, b2) for i in range(3)
        ])
        return filtered_b

    def rms(data, filtered):
        return np.sqrt(np.mean(np.sum((data - filtered)**2, axis=1)))

    # RMS analysis for multiple cutoff frequencies
    frequency = np.zeros(fc_range + 1)
    for fc in range(1, fc_range + 1):
        filtered_data = total(fs, fc, data)
        frequency[fc] = rms(data, filtered_data)

    # 產生 RMS 殘差圖
    fig, ax = plt.subplots()
    ax.plot(frequency[1:], label="RMS")
    ax.axhline(frequency[16], color='r', linestyle='--', label='Zero noise pass')

    # 直線擬合（19Hz ~ 29Hz）
    a = (frequency[29] - frequency[19]) / (29 - 19)
    b = frequency[16] - a * 16
    rms_each_frequency = np.array([a * i + b for i in range(1, fc_range + 1)])

    ax.plot(rms_each_frequency, label="Straight line")
    ax.set_title("Residual Plot")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Residual")
    ax.legend()

    return fig  

fig = filtering(data, fs=500) # fs = camera or force plate frequency 
plt.show()


