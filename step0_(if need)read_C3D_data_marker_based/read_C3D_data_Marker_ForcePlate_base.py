import ezc3d
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def read_c3d(path, multiple):
# the processes including the interpolation 
    """
    input1 path of the C3D data
    inpu2 the re-sampling times (using motion data frequency to time the number) 
    example marker frequency = 500Hz input2 = 2 -> 500*2 = 1000
    ----------
    outcome1 combine marker and fp data in a dictionary
    outcome2 the description of the data (some variables are mannual)
    """
    # read c3d file
    c = ezc3d.c3d(path, extract_forceplat_data=True)
    # convert c3d motion data to DataFrame format
    # create column's name of motion data
    motion_axis = ['x', 'y', 'z']
    motion_markers = []
    for marker_name in c['parameters']['POINT']['LABELS']['value']:
        for axis in motion_axis:
            name = marker_name + '_' + axis
            motion_markers.append(name)
    # create x, y, z matrix to store motion data
    motion_data = pd.DataFrame(np.zeros([np.shape(c['data']['points'])[-1], # last frame + 1
                                         len(c['parameters']['POINT']['LABELS']['value'])*3]), # marker * 3
                               columns=motion_markers) 
    # key in data into matrix
    for i in range(len(c['parameters']['POINT']['LABELS']['value'])):
        motion_data.iloc[:, 1*i*3:1*i*3+3] = np.transpose(c['data']['points'][:3, i, :])
    # insert time frame
    # create time frame
    motion_time = np.linspace(
                                0, # start
                              ((c['header']['points']['last_frame'])/c['header']['points']['frame_rate']), # stop = last_frame/frame_rate
                              num = (np.shape(c['data']['points'])[-1]) # num = last_frame
                              )
    # insert time frame to motion data
    motion_data.insert(0, 'Frame', motion_time)
    ## 3.1 create force plate channel name (the ori unit Force = N; torque = Nmm; COP = mm in Qualysis C3D)
    GRF_list = []
    GRT_list = []
    COP_list = []
    Number_Forceplates = len(c["data"]["platform"])
    for i in range(Number_Forceplates):
        GRF = c["data"]["platform"][i]['force']
        GRF = GRF.T  
        GRF_list.append(GRF)
        GRT = c["data"]["platform"][i]['moment']
        GRT = GRT.T / 1000  
        GRT_list.append(GRT)
        COP = c["data"]["platform"][i]['center_of_pressure']
        COP = COP.T / 10 
        COP_list.append(COP)
    # Store all force plate data into a dictionary
    FP_data = {}
    for idx in range(Number_Forceplates):
        FP_data[f'FORCE: GRF{idx}'] = GRF_list[idx]
        FP_data[f'FORCE: GRT{idx}'] = GRT_list[idx]
        FP_data[f'FORCE: COP_{idx}'] = COP_list[idx]
    # check the variable type of the force plate data
    if not isinstance(motion_data, np.ndarray):
        motion_data = np.array(motion_data)
    motion_data = motion_data[:, 1:]
    label = c['parameters']['POINT']['LABELS']['value'] 

    # change the variable type from dataframe to dictionary and change unit 
    motion_data_dict = {}
    for i, marker_name in enumerate(label):  #label the name of the data for each variable
        start_col = i * 3
        end_col = start_col + 3
        marker_motion_data = motion_data[:, start_col:end_col] /10 # divide 10 mm -> cm
        motion_data_dict[marker_name] = marker_motion_data  #maker the name of each variable

    # Interpolation: using polynomial method, order = 3 
    def interpolate_with_fallback(data):
        data = pd.DataFrame(data)
        data.replace(0, np.nan, inplace=True)
        data = data.interpolate(method='linear', axis=0)
        data.fillna(method='bfill', inplace=True)  
        data.fillna(method='ffill', inplace=True)  
        if data.isnull().values.any() or (data == 0).any().any():
            data = data.interpolate(method='polynomial', order=2, axis=0).fillna(method='bfill').fillna(method='ffill')
        return data.values
    fillgap_markers = {key: interpolate_with_fallback(value) for key, value in motion_data_dict.items()}
    
    # Resampling function
    def time_normalize(data, target_length):
        original_length = data.shape[0]
        x_original = np.linspace(0, 1, original_length)
        x_target = np.linspace(0, 1, target_length)
        interpolated_data = interp1d(x_original, data, axis=0)(x_target)
        return interpolated_data
    
    # Target length for resampling
    length_of_list = motion_time.shape[0]
    target_length = length_of_list * multiple
    FP_data_normalized = {key: time_normalize(value, target_length) for key, value in FP_data.items()}
    fillgap_markers_normalized = {key: time_normalize(value, target_length) for key, value in fillgap_markers.items()}
    normalized_time_motion = time_normalize(motion_time, target_length)
    time_normalization = {'time(ms)': normalized_time_motion}  
    combine_dict = {**time_normalization, **fillgap_markers_normalized, **FP_data_normalized}
    
    # description: number of markers, frequency, FP numbers, unit of each data
    descriptions = {
        "Number of Markers": c['header']['points']['size'],
        "Camera Rate": c['header']['points']['frame_rate'],
        "Analog Channels": c['header']['analogs']['size'],
        "Analog Sample Rate": c['header']['analogs']['frame_rate'],
        "Number Of Forceplates": len(c["data"]["platform"]),
        "marker_unit": "cm", # should check here 
        "Force_unit": "N",
        "Torque_unit": "Nm",
        "Time_unit": "ms"
    }
    return combine_dict, descriptions

# example: data, descripton = read_c3d(C3D.file, 5( it means: 5 * marker data frequency))
combined_data_dy, description_dy = read_c3d('/Users/kairenzheng/Desktop/example of code/read_C3D_data_marker_based/D 2.c3d', 2)

# pick up data
# example - pick up time 
time_dynamic = combined_data_dy['time(ms)']