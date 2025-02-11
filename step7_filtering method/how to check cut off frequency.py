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
Rthumb = np.array(dg1_data[:,100:103], dtype=float)


# 2nd_forward
def two_order_for(data, a0, a1, a2, b1, b2):
    forward_f = np.zeros(data.shape)
    i = 0
    for i in range(len(data)):
        if i == 0:
            forward_f[0] = data[0]
        elif i == 1:
            forward_f[1] = data[1]
        else:
            forward_f[i] = a0* data[i] + a1* data[i-1] + a2* data[i-2] - b1*forward_f[i-1] - b2*forward_f[i-2]
    return forward_f

# 2nd_backforward
def two_order_back(forward_f, a0, a1, a2, b1, b2):
    backward_f_r = np.zeros(forward_f.shape)
    forward_f_r = forward_f[::-1]
    i = 0
    for i in range(len(forward_f_r)):
        if i == 0:
            backward_f_r[0] = forward_f_r[0]
        elif i == 1:
            backward_f_r[1] = forward_f_r[1]
        else:
            backward_f_r[i] = a0* forward_f_r[i] + a1* forward_f_r[i-1] + a2* forward_f_r[i-2] - b1*backward_f_r[i-1] - b2*backward_f_r[i-2]
    backward_f = backward_f_r[::-1]
    return backward_f

def total(fs, fc, data):
    fs = fs
    fc = fc
    fr = fs/fc
    omgc = np.tan(np.pi/fr)
    c = 1 + 2*np.cos(np.pi/4)*omgc + omgc**2
    a0 = a2 = omgc**2/c
    a1 = 2*a0
    b1 = 2*(omgc**2 -1)/c
    b2 = (1 - 2*np.cos(np.pi/4)*omgc + omgc**2)/c
    filter2f_X = two_order_for(data[:, 0], a0, a1, a2, b1, b2)
    filter2f_Y = two_order_for(data[:, 1], a0, a1, a2, b1, b2)
    filter2f_Z = two_order_for(data[:, 2], a0, a1, a2, b1, b2)
    filter2b_X = two_order_back(filter2f_X, a0, a1, a2, b1, b2)
    filter2b_Y = two_order_back(filter2f_Y, a0, a1, a2, b1, b2)
    filter2b_Z = two_order_back(filter2f_Z, a0, a1, a2, b1, b2)
    return filter2f_X, filter2f_Y, filter2f_Z, filter2b_X, filter2b_Y, filter2b_Z, omgc

#find total RMS

# fs = 500 fc = 10
filter2f_X, filter2f_Y, filter2f_Z, filter2b_X, filter2b_Y, filter2b_Z, omgc = total(500, 5, Rthumb)
plt.plot(Rthumb)
plt.plot(filter2f_X)
plt.plot(filter2f_Y)
plt.plot(filter2f_Z)
plt.plot(filter2b_X)
plt.plot(filter2b_Y)
plt.plot(filter2b_Z)
plt.title('marker position(fc = 5)')
plt.xlabel('time')
plt.ylabel('marker position(m)')
plt.legend(('raw_x', 'raw_y', 'raw_z','2nd_f_x', '2nd_f_y','2nd_f_z', '2nd_b_x', '2nd_b_y', '2nd_b_z'), loc = 'upper right')
plt.show()


# find true omgc
true_omgc = omgc*(1/0.802)

#find RMS
def rms(data, fx, fy, fz):
    rms = np.zeros(data.shape[0]) 
    N = data.shape[0]
    aa = 0
    for aa in range(data.shape[0]):
        rms[aa] = np.array((data[aa, 0] - fx[aa])**2 + (data[aa, 1] - fy[aa])**2 + (data[aa, 2] - fz[aa])**2)
        aa = aa + 1  
    sum_rms = np.sqrt(rms.sum()/N)
    return sum_rms

rms_fc_30 = rms(Rthumb, filter2b_X, filter2b_Y, filter2b_Z)
print(rms_fc_30)

#find total RMS
frequency = np.zeros(31)
i = range(1, 30)
for i in range(len(Rthumb)):
    if i == 0:
        continue
    else:
        filter2f_X, filter2f_Y, filter2f_Z, filter2b_X, filter2b_Y, filter2b_Z, omgcc = total(500, i, Rthumb)
        sum_rms = rms(Rthumb, filter2b_X, filter2b_Y, filter2b_Z)
        sum_fc = sum_rms 
        frequency[i] = sum_fc
        if i == 30:
            break
rms = frequency[1:]
plt.plot(rms)

#make straight line function using y = ax + b 
a = (rms[29] - rms[19]) / (29-19)
b = rms[16] - a*16 

rms_each_frequency = np.zeros(30)
for i in range(len(rms)):
    y = a*i + b
    rms_each_frequency[i] = y

plt.plot(rms_each_frequency[:30])

#the RMS of each frequency
plt.plot(rms)
plt.axhline(b, color = 'r', linestyle = '--', label = 'zero noise pass')
plt.plot(rms_each_frequency[:30])
plt.title('residual plot')
plt.xlabel('frequency')
plt.ylabel('residual')
plt.legend(('RMS', 'horizontal_line', 'straight_line'), loc = 'upper right')
plt.show()


filter2f_XX, filter2f_YY, filter2f_ZZ, filter2b_XX, filter2b_YY, filter2b_ZZ, omgcc = total(500, 12, Rthumb)
foward = np.vstack((filter2f_XX, filter2f_YY, filter2f_ZZ))
foward_wrist = foward.T
Wrist_filtered = np.vstack((filter2b_XX, filter2b_YY, filter2b_ZZ))
Wrist_filtered = Wrist_filtered.T
plt.plot(Rthumb)
plt.plot(foward_wrist)
plt.plot(Wrist_filtered)
plt.title('marker position(fc = 12)')
plt.xlabel('frame')
plt.ylabel('filtered_position(m)')
plt.legend(('raw_x', 'raw_y', 'raw_z','2nd_f_x', '2nd_f_y','2nd_f_z', '2nd_b_x', '2nd_b_y', '2nd_b_z'), loc = 'upper right')
plt.show()

