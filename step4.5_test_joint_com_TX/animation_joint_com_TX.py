import ezc3d
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

#%% read C3D data

def read_c3d(path, multiple):
# the processes including the interpolation 
    """
    input1 path of the C3D data
    inpu2 the re-sampling times (using motion data frequency to time)
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

# example: data, descripton = read_c3d(C3D.file, 5)
# file_patterns = ["0sta_golfer.c3d", "1Ball-D 1.c3d", "2Club-D 1.c3d"]

combined_data_sta_golfer, description_sta_golfer = read_c3d('/Users/kairenzheng/Desktop/example of code/step4.5_test_joint_com_TX/golfer_sta.c3d', 2)
combined_data_sta_ball, description_Ball = read_c3d('/Users/kairenzheng/Desktop/example of code/step4.5_test_joint_com_TX/Ball-D 1.c3d',2)
combined_data_sta_club, description_club = read_c3d('/Users/kairenzheng/Desktop/example of code/step4.5_test_joint_com_TX/Club-D 1.c3d',2)
combined_data_dy, description_dy = read_c3d('/Users/kairenzheng/Desktop/example of code/step4.5_test_joint_com_TX/D 2.c3d', 2)
time_dynamic = combined_data_dy['time(ms)']


# set up for the sta mean calculation 
def mean_sta(data_dicts, prefixes):
    """
    input1 = sta data (for golf data = golfer, ball, club)
    input2 = sta data -> variables needs to do average
     
    output = mean of the data
    """
    means = {}
    for prefix, data_dict in zip(prefixes, data_dicts):
        sta_data = {}
        # 遍歷每個矩陣計算每列的均值
        for key, value in data_dict.items():
            if key.startswith(prefix):  
                mean_key = key.replace(" ", "")  # 去掉空格
                sta_data[mean_key] = np.mean(value, axis=0)  
        
        means[prefix] = sta_data

    return means
sta_data = [combined_data_sta_golfer, combined_data_sta_ball, combined_data_sta_club]
prefixes = ["Golfer:", "Ball:", "Club:"]
mean_sta = mean_sta(sta_data, prefixes)

# creat secondary points in static data (prepare to transfer to dynamic data)
# It depends one your setting of the secondary points
# Hip using T-A method  (Tylkowski-Andriacchi (T-A) Method)


def calculate_secondary_points_sta(sta_data):
    w = sta_data['Golfer:']["Golfer:LASIS"][1] - sta_data['Golfer:']["Golfer:RASIS"][1]
    
    def avg_point(*points):
        return sum(points) / len(points)
    
    secondary_points = {
        # using mid point method (point1 + point2) / 2
        'sta_jc_Rsho': avg_point(sta_data['Golfer:']["Golfer:RASho"], sta_data['Golfer:']["Golfer:RPSho"]),
        'sta_jc_Lsho': avg_point(sta_data['Golfer:']["Golfer:LASho"], sta_data['Golfer:']["Golfer:LPSho"]),
        'sta_jc_Relb': avg_point(sta_data['Golfer:']["Golfer:RElbow"], sta_data['Golfer:']["Golfer:RMElbow"]),
        'sta_jc_Lelb': avg_point(sta_data['Golfer:']["Golfer:LElbow"], sta_data['Golfer:']["Golfer:LMElbow"]),
        'sta_jc_Rknee': avg_point(sta_data['Golfer:']["Golfer:RKnee"], sta_data['Golfer:']["Golfer:RMKnee"]),
        'sta_jc_Lknee': avg_point(sta_data['Golfer:']["Golfer:LKnee"], sta_data['Golfer:']["Golfer:LMKnee"]),
        'sta_jc_CH': avg_point(sta_data['Club:']["Club:FaceHeel"], sta_data['Club:']["Club:FaceBottom"], 
                               sta_data['Club:']["Club:FaceToe"], sta_data['Club:']["Club:FaceTop"]),
        # 
        'sta_jc_Rhip': np.array([sta_data['Golfer:']["Golfer:RGT"][0], 
                                 sta_data['Golfer:']["Golfer:RASIS"][1] + 0.14 * w, 
                                 sta_data['Golfer:']["Golfer:RASIS"][2] - 0.3 * w]),
        'sta_jc_Lhip': np.array([sta_data['Golfer:']["Golfer:LGT"][0], 
                                 sta_data['Golfer:']["Golfer:LASIS"][1] - 0.14 * w, 
                                 sta_data['Golfer:']["Golfer:LASIS"][2] - 0.3 * w])
    }
    return secondary_points

secondary_points_sta = calculate_secondary_points_sta(mean_sta)

def plot_3d_secondary_points(secondary_points_sta):
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 繪製每個點並添加標籤
    for label, point in secondary_points_sta.items():
        ax.scatter(point[0], point[1], point[2], label=label)

    # 設定標題和軸標籤
    ax.set_title('3D Coordinates of Secondary Points')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # 添加圖例
    ax.legend()

    # 顯示圖形
    plt.show()
plot_3d_secondary_points(secondary_points_sta)

#%% find JC and TX

def find_JC_and_dy_TX(combined_data_dy, mean_sta, secondary_points_sta): 
    d_jc_Rwri = (combined_data_dy["Golfer:RWrist"]  + combined_data_dy["Golfer:RMWrist"]) / 2
    d_jc_Lwri = (combined_data_dy["Golfer:LWrist"]  + combined_data_dy["Golfer:LMWrist"]) / 2
    d_jc_Rank = (combined_data_dy["Golfer:RAnkle"] + combined_data_dy["Golfer:RMAnkle"]) / 2
    d_jc_Lank = (combined_data_dy["Golfer:LAnkle"] + combined_data_dy["Golfer:LMAnkle"]) / 2
    d_jc_Utrunk = (combined_data_dy["Golfer:C7"] + combined_data_dy["Golfer:SN"]) / 2
    d_jc_Mtrunk = (combined_data_dy["Golfer:T5"] + combined_data_dy["Golfer:XI"]) / 2
    d_mid_head = (combined_data_dy["Golfer:HeadR"] + combined_data_dy["Golfer:HeadL"]) / 2
    #make sta frames
    def sta_TX (point1, point2, point3):
        x = point1 - point2
        f = point3 - point2
        z = np.cross(f, x)
        y = np.cross(z , x)
        s_x, s_y, s_z = x / np.linalg.norm(x), y / np.linalg.norm(y), z / np.linalg.norm(z)
        s_TX = np.vstack((s_x, s_y, s_z))
        return s_TX
    
    # make pel dy_TX
    sta_mPSIS = (mean_sta["Golfer:"]["Golfer:RPSIS"] + mean_sta["Golfer:"]["Golfer:LPSIS"]) / 2
    sta_o_SIS = (mean_sta["Golfer:"]["Golfer:RASIS"] + mean_sta["Golfer:"]["Golfer:LASIS"] + mean_sta["Golfer:"]["Golfer:RPSIS"] + mean_sta["Golfer:"]["Golfer:LPSIS"]) / 4
    s_TX = sta_TX(mean_sta["Golfer:"]["Golfer:RASIS"], mean_sta["Golfer:"]["Golfer:LASIS"], sta_mPSIS)
    def dy_pel_TX(point1, point2, point3):
        x = point1 - point2
        f = point3 - point2
        z = np.cross(f, x)
        y = np.cross(z , x)
    
        tx_dynamic = np.zeros((x.shape[0], 3, 3))
        for i in range(x.shape[0]):
            tx_dynamic[i, :, :] = np.vstack((x[i, :] / np.linalg.norm(x[i, :]),
                                              y[i, :] / np.linalg.norm(y[i, :]),
                                              z[i, :] / np.linalg.norm(z[i, :])))
        return tx_dynamic
    d_mPSIS = (combined_data_dy["Golfer:RPSIS"] + combined_data_dy["Golfer:LPSIS"]) / 2
    d_o_mid_pel = (combined_data_dy["Golfer:RASIS"] + combined_data_dy["Golfer:LASIS"] + combined_data_dy["Golfer:RPSIS"] + combined_data_dy["Golfer:LPSIS"]) / 4
    d_pel_TX = dy_pel_TX(combined_data_dy["Golfer:RASIS"], combined_data_dy["Golfer:LASIS"], d_mPSIS)
    
    # Hip points from Sta to Dy trials
    def StoD(sta_o_SIS, p, s_TX, d_o_SIS, dy_frame):
        """
        計算 d_Rhip1 的值
        
        Parameters:
        sta_o (ndarray): origin in static daya
        secondary_points_sta (dict): 包含全局點坐標的字典
        s_TX (ndarray): 框架轉換矩陣
        d_o_SIS (ndarray): (frames, 3) 矩陣
        dy_pel (ndarray): (frames, 3, 3) 矩陣
        
        Returns:
        d_Rhip1 (ndarray): (frames, 3) 矩陣，計算後的結果
        """
        # 提取坐標
        # 計算 Rhip 相對於原點的向量
        distance = p - sta_o_SIS
        # 計算局部 Rhip
        local_point = np.dot(s_TX, distance)
        frames = len(d_o_SIS)
        
        # 初始化 d_Rhip1
        glob_point = np.zeros((frames, 3))
        
        # 迴圈遍歷每一幀
        for i in range(frames):
            # 計算 d_Rhip1
            glob_point[i, :] = np.dot(dy_frame[i, :, :].T, local_point) + d_o_SIS[i, :]
    
        return glob_point
    d_jc_Rhip_global = StoD(sta_o_SIS, secondary_points_sta['sta_jc_Rhip'], s_TX, d_o_mid_pel, d_pel_TX)
    d_jc_Lhip_global = StoD(sta_o_SIS, secondary_points_sta['sta_jc_Lhip'], s_TX, d_o_mid_pel, d_pel_TX)
    
    # Knee points from Sta to Dy trials
    sta_fRleg_TX = sta_TX(mean_sta["Golfer:"]["Golfer:RThigh"], mean_sta["Golfer:"]["Golfer:RKnee"], secondary_points_sta['sta_jc_Rhip'])
    sta_o_Rleg = (secondary_points_sta['sta_jc_Rhip'] + mean_sta["Golfer:"]["Golfer:RKnee"] + mean_sta["Golfer:"]["Golfer:RThigh"]) / 3
    sta_fLleg_TX = sta_TX(mean_sta["Golfer:"]["Golfer:LThigh"], mean_sta["Golfer:"]["Golfer:LKnee"], secondary_points_sta['sta_jc_Lhip'])
    sta_o_Lleg = (secondary_points_sta['sta_jc_Lhip'] + mean_sta["Golfer:"]["Golfer:LKnee"] + mean_sta["Golfer:"]["Golfer:LThigh"]) / 3
    d_fRleg_TX = dy_pel_TX(combined_data_dy["Golfer:RThigh"], combined_data_dy["Golfer:RKnee"], d_jc_Rhip_global)
    d_o_Rleg = (d_jc_Rhip_global + combined_data_dy["Golfer:RKnee"] + combined_data_dy["Golfer:RThigh"]) / 3
    d_fLleg_TX = dy_pel_TX(combined_data_dy["Golfer:LThigh"], combined_data_dy["Golfer:LKnee"], d_jc_Lhip_global)
    d_o_Lleg = (d_jc_Lhip_global + combined_data_dy["Golfer:LKnee"] + combined_data_dy["Golfer:LThigh"]) / 3
    d_jc_Rknee_global = StoD(sta_o_Rleg, secondary_points_sta['sta_jc_Rknee'], sta_fRleg_TX, d_o_Rleg, d_fRleg_TX)
    d_jc_Lknee_global = StoD(sta_o_Lleg, secondary_points_sta['sta_jc_Lknee'], sta_fLleg_TX, d_o_Lleg, d_fLleg_TX)
    
    def dy_limbs_TX(point1, point2, point3, RorL):
        if RorL == 'R':
            z = point1 - point2
            f = point3 - point2
            y = np.cross(z, f)
            x = np.cross(y , z)
    
            tx_dynamic = np.zeros((x.shape[0], 3, 3))
            for i in range(x.shape[0]):
                tx_dynamic[i, :, :] = np.vstack((x[i, :] / np.linalg.norm(x[i, :]),
                                                  y[i, :] / np.linalg.norm(y[i, :]),
                                                  z[i, :] / np.linalg.norm(z[i, :])))
        elif RorL == 'L':
            f = point1 - point2
            z = point3 - point2
            y = np.cross(f, z)
            x = np.cross(y , z)
    
            tx_dynamic = np.zeros((x.shape[0], 3, 3))
            for i in range(x.shape[0]):
                tx_dynamic[i, :, :] = np.vstack((x[i, :] / np.linalg.norm(x[i, :]),
                                                  y[i, :] / np.linalg.norm(y[i, :]),
                                                  z[i, :] / np.linalg.norm(z[i, :])))
        return tx_dynamic
    
    # make R and L Thigh dy_TX
    d_Rthigh_TX = dy_limbs_TX(d_jc_Rhip_global, d_jc_Rknee_global, combined_data_dy["Golfer:RThigh"], 'R')
    d_Lthigh_TX = dy_limbs_TX(combined_data_dy["Golfer:LThigh"], d_jc_Lknee_global, d_jc_Lhip_global, 'L')
    
    # make R and L shank dy_TX
    d_Rsha_TX = dy_limbs_TX(d_jc_Rknee_global, d_jc_Rank, combined_data_dy["Golfer:RAnkle"], 'R')
    d_Lsha_TX = dy_limbs_TX(combined_data_dy["Golfer:LAnkle"], d_jc_Lank, d_jc_Lknee_global, 'L')
    
    def dy_ft_TX(point1, point2, point3):
        y = point1 - point2
        f = point3 - point2
        x = np.cross(y, f)
        z = np.cross(x, y)
    
        tx_dynamic = np.zeros((x.shape[0], 3, 3))
        for i in range(x.shape[0]):
            tx_dynamic[i, :, :] = np.vstack((x[i, :] / np.linalg.norm(x[i, :]),
                                              y[i, :] / np.linalg.norm(y[i, :]),
                                              z[i, :] / np.linalg.norm(z[i, :])))
        return tx_dynamic

    # make R and L foot dy_TX
    d_Rft_TX = dy_ft_TX(combined_data_dy["Golfer:RToe"], combined_data_dy["Golfer:RHeel"], d_jc_Rank)
    d_Lft_TX = dy_ft_TX(combined_data_dy["Golfer:LToe"], combined_data_dy["Golfer:LHeel"], d_jc_Lank)
    
    # Shoulder jc points from Sta to Dy trials
    sta_fRUA_TX = sta_TX(mean_sta["Golfer:"]["Golfer:RUA1"], mean_sta["Golfer:"]["Golfer:RUA2"], mean_sta["Golfer:"]["Golfer:RUA3"])
    sta_o_RUA = (mean_sta["Golfer:"]["Golfer:RUA1"] + mean_sta["Golfer:"]["Golfer:RUA2"] + mean_sta["Golfer:"]["Golfer:RUA3"]) / 3
    sta_fLUA_TX = sta_TX(mean_sta["Golfer:"]["Golfer:LUA1"], mean_sta["Golfer:"]["Golfer:LUA2"], mean_sta["Golfer:"]["Golfer:LUA3"])
    sta_o_LUA = (mean_sta["Golfer:"]["Golfer:LUA1"] + mean_sta["Golfer:"]["Golfer:LUA2"] + mean_sta["Golfer:"]["Golfer:LUA3"]) / 3
    d_fRUA_TX = dy_pel_TX(combined_data_dy["Golfer:RUA1"], combined_data_dy["Golfer:RUA2"], combined_data_dy["Golfer:RUA3"])
    d_o_RUA = (combined_data_dy["Golfer:RUA1"] + combined_data_dy["Golfer:RUA2"] + combined_data_dy["Golfer:RUA3"]) / 3
    d_fLUA_TX = dy_pel_TX(combined_data_dy["Golfer:LUA1"], combined_data_dy["Golfer:LUA2"], combined_data_dy["Golfer:LUA3"])
    d_o_LUA = (combined_data_dy["Golfer:LUA1"] + combined_data_dy["Golfer:LUA2"] + combined_data_dy["Golfer:LUA3"]) / 3
    d_jc_Rsho_global = StoD(sta_o_RUA, secondary_points_sta['sta_jc_Rsho'], sta_fRUA_TX, d_o_RUA, d_fRUA_TX)
    d_jc_Relb_global = StoD(sta_o_RUA, secondary_points_sta['sta_jc_Relb'], sta_fRUA_TX, d_o_RUA, d_fRUA_TX)
    d_jc_Lsho_global = StoD(sta_o_LUA, secondary_points_sta['sta_jc_Lsho'], sta_fLUA_TX, d_o_LUA, d_fLUA_TX)
    d_jc_Lelb_global = StoD(sta_o_LUA, secondary_points_sta['sta_jc_Lelb'], sta_fLUA_TX, d_o_LUA, d_fLUA_TX)
    
    # make upper arm dy_TX
    d_RUA_TX = dy_limbs_TX(d_jc_Rsho_global, d_jc_Relb_global, combined_data_dy["Golfer:RUA1"], 'R')
    d_LUA_TX = dy_limbs_TX(combined_data_dy["Golfer:LUA1"], d_jc_Lelb_global, d_jc_Lsho_global, 'L')
    
    # forearm arm d TX
    d_RFA_TX = dy_limbs_TX(d_jc_Relb_global, d_jc_Rwri, combined_data_dy["Golfer:RWrist"], 'R')
    d_LFA_TX = dy_limbs_TX(combined_data_dy["Golfer:LWrist"], d_jc_Lwri, d_jc_Lelb_global, 'L')
    
    # make abdomen and thorax dy_TX
    def dy_trunk_TX(point1, point2, point3, point4):
        z = point1 - point2
        f = point3 - point4
        x = np.cross(f, z)
        y = np.cross(z, x)
    
        tx_dynamic = np.zeros((x.shape[0], 3, 3))
        for i in range(x.shape[0]):
            tx_dynamic[i, :, :] = np.vstack((x[i, :] / np.linalg.norm(x[i, :]),
                                              y[i, :] / np.linalg.norm(y[i, :]),
                                              z[i, :] / np.linalg.norm(z[i, :])))
        return tx_dynamic
    # make abdomen and thorax dy_TX
    mid_sho = (d_jc_Rsho_global + d_jc_Lsho_global) / 2
    d_abdomen_TX = dy_trunk_TX(d_jc_Mtrunk, d_o_mid_pel, combined_data_dy["Golfer:XI"], combined_data_dy["Golfer:T5"])
    mid_sho = (d_jc_Rsho_global + d_jc_Lsho_global) / 2
    d_thorax_TX = dy_trunk_TX(mid_sho, d_jc_Mtrunk, combined_data_dy["Golfer:SN"], combined_data_dy["Golfer:C7"])
    
    # make head dy_TX
    def sta_head_TX (point1, point2, point3):
        x = point1 - point2
        y = point3 - point2
        z = np.cross(x, y)
        s_x, s_y, s_z = x / np.linalg.norm(x), y / np.linalg.norm(y), z / np.linalg.norm(z)
        s_TX = np.vstack((s_x, s_y, s_z))
        return s_TX
    
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
    d_head_TX = dy_head_TX(combined_data_dy["Golfer:HeadR"], d_mid_head, combined_data_dy["Golfer:HeadA"])
    
    # make CH dy_TX
    s_CH_TX = sta_TX(mean_sta["Club:"]["Club:Head3"], mean_sta["Club:"]["Club:Head1"], mean_sta["Club:"]["Club:Head2"])
    sta_o_CH_mid = (mean_sta["Club:"]["Club:Head1"] + mean_sta["Club:"]["Club:Head2"] + mean_sta["Club:"]["Club:Head3"]) / 3
    d_CH_TX = dy_pel_TX(combined_data_dy["Club:Head3"], combined_data_dy["Club:Head1"], combined_data_dy["Club:Head2"])
    d_o_CH_mid = (combined_data_dy["Club:Head1"] + combined_data_dy["Club:Head2"] + combined_data_dy["Club:Head3"]) / 3
    d_CH_heel = StoD(sta_o_CH_mid, mean_sta["Club:"]["Club:FaceHeel"],s_CH_TX, d_o_CH_mid, d_CH_TX)
    d_CH_Bottom = StoD(sta_o_CH_mid, mean_sta["Club:"]["Club:FaceBottom"],s_CH_TX, d_o_CH_mid, d_CH_TX)
    d_CH_Top = StoD(sta_o_CH_mid, mean_sta["Club:"]["Club:FaceTop"],s_CH_TX, d_o_CH_mid, d_CH_TX)
    d_jc_CH_face = StoD(sta_o_CH_mid, secondary_points_sta['sta_jc_CH'],s_CH_TX, d_o_CH_mid, d_CH_TX)
    d_CH_head_TX = dy_head_TX(d_CH_Bottom, d_CH_heel, d_CH_Top)
    
    
    TX_dy = {
        "pel_TX": d_pel_TX,
        "Rthigh_TX": d_Rthigh_TX,
        "Lthigh_TX": d_Lthigh_TX,
        "Rsha_TX": d_Rsha_TX,
        "Lsha_TX": d_Lsha_TX,
        "Rft_TX": d_Rft_TX,
        "Lft_TX": d_Lft_TX,
        "RUA_TX": d_RUA_TX,
        "LUA_TX": d_LUA_TX,
        "RFA_TX": d_RFA_TX,
        "LFA_TX": d_LFA_TX,
        "abdomen_TX": d_abdomen_TX,
        "thorax_TX": d_thorax_TX,
        "head_TX": d_head_TX,
        "CH_TX": d_CH_head_TX
    }
    
    JC_dy = {
        "M_pel": d_o_mid_pel,
        "Mtrunk": d_jc_Mtrunk,
        "Utrunk": d_jc_Utrunk,
        "Head": d_mid_head,
        "Rhip": d_jc_Rhip_global,
        "Lhip": d_jc_Lhip_global,
        "Rknee": d_jc_Rknee_global,
        "Lknee": d_jc_Lknee_global,
        "Rank": d_jc_Rank,
        "Lank": d_jc_Lank,
        "Rsho": d_jc_Rsho_global,
        "Lsho": d_jc_Lsho_global,
        "Relb": d_jc_Relb_global,
        "Lelb": d_jc_Lelb_global,
        "Rwri": d_jc_Rwri,
        "Lwri": d_jc_Lwri,
        "CH":   d_jc_CH_face
        }
    return TX_dy, JC_dy

dy_TX, joint_center_position = find_JC_and_dy_TX(combined_data_dy, mean_sta, secondary_points_sta)

def plot_3d_scatter(joint_center_position):
    frame1_data = {}

    # 從字典中提取第 0 行並轉換為 NumPy 陣列
    for key in joint_center_position:
        frame1_data[key] = np.array(joint_center_position[key][0, :3])

    # 初始化 x, y, z 列表
    x = []
    y = []
    z = []

    # 從 frame1_data 字典中分別提取 x, y, z 座標
    for key in frame1_data:
        x.append(frame1_data[key][0])
        y.append(frame1_data[key][1])
        z.append(frame1_data[key][2])

    # 將列表轉換為 NumPy 陣列
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # 繪製 3D 圖形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 3D 散點圖
    ax.scatter(x, y, z, c='r', marker='o')

    # 標註坐標軸
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 設置三軸範圍並從 (0, 0, 0) 開始
    ax.set_xlim([0, max(x) * 1.1])  # x 軸範圍
    ax.set_ylim([0, max(y) * 1.1])  # y 軸範圍
    ax.set_zlim([0, max(z) * 1.1])  # z 軸範圍

    plt.title("3D Scatter Plot of Frame 1 from jc_dy Matrices")
    plt.show()
plot_3d_scatter(joint_center_position)

#%% find COM, mass, and MOI of each segment

#make COM, MASS, MOI of each segments
"should input height and weight"
# Segments and their corresponding mass fractions and COM ratios
def segment_mass_CM_moi(joint_center_position, combined_data_dy, mass, gender):
    if gender == 'M':
        segments_Male = [
            ("head", 0.0694),
            ("Utrunk", 0.1596),
            ("Mtrunk", 0.1633),
            ("pel", 0.1117),
            ("RUA", 0.0271),
            ("LUA", 0.0271),
            ("RFA", 0.0162),
            ("LFA", 0.0162),
            ("Rthigh", 0.1416),
            ("Lthigh", 0.1416),
            ("Rsha", 0.0433),
            ("Lsha", 0.0433),
            ("Rft", 0.0137),
            ("Lft", 0.0137)
        ]
        masses = {name: mass * fraction for name, fraction in segments_Male}
        mid_Utrunk = (combined_data_dy["Golfer:SN"] + combined_data_dy["Golfer:C7"]) / 2
        cm_head = (combined_data_dy["Golfer:HeadT"] - mid_Utrunk) * 0.4998 + mid_Utrunk
        cm_Utrunk = ((joint_center_position['Rsho'] + joint_center_position['Lsho']) / 2 - joint_center_position['Mtrunk']) * 0.7001 + joint_center_position['Mtrunk']
        cm_Mtrunk = (joint_center_position['Mtrunk'] - joint_center_position['M_pel']) * 0.5498 + joint_center_position['M_pel']
        cm_pel = (joint_center_position['M_pel'])
        cm_RUA = (joint_center_position['Rsho'] - joint_center_position['Relb']) * 0.4228 + joint_center_position['Relb']
        cm_LUA = (joint_center_position['Lsho'] - joint_center_position['Lelb']) * 0.4228 + joint_center_position['Lelb']
        cm_RFA = (joint_center_position['Relb'] - joint_center_position['Rwri']) * 0.5426 + joint_center_position['Rwri']
        cm_LFA = (joint_center_position['Lelb'] - joint_center_position['Lwri']) * 0.5426 + joint_center_position['Lwri']
        cm_Rthigh = (joint_center_position['Rhip'] - joint_center_position['Rknee']) * 0.5905 + joint_center_position['Rknee']
        cm_Lthigh = (joint_center_position['Lhip'] - joint_center_position['Lknee']) * 0.5905 + joint_center_position['Lknee']
        cm_Rsha = (joint_center_position['Rknee'] - joint_center_position['Rank']) * 0.5541 + joint_center_position['Rank']
        cm_Lsha = (joint_center_position['Lknee'] - joint_center_position['Lank']) * 0.5541 + joint_center_position['Lank']
        cm_Rft = (combined_data_dy["Golfer:RToe"] - combined_data_dy["Golfer:RHeel"]) * 0.4415 + combined_data_dy["Golfer:RHeel"]
        cm_Lft = (combined_data_dy["Golfer:LToe"] - combined_data_dy["Golfer:LHeel"]) * 0.4415 + combined_data_dy["Golfer:LHeel"]
        coms = {
            "cm_head": cm_head,
            "cm_thorax": cm_Utrunk,
            "cm_abdomen": cm_Mtrunk,
            "cm_pel": cm_pel,
            "cm_RUA": cm_RUA,
            "cm_LUA": cm_LUA,
            "cm_RFA": cm_RFA,
            "cm_LFA": cm_LFA,
            "cm_Rthigh": cm_Rthigh,
            "cm_Lthigh": cm_Lthigh,
            "cm_Rsha": cm_Rsha,
            "cm_Lsha": cm_Lsha,
            "cm_Rft": cm_Rft,
            "cm_Lft": cm_Lft
        }
        moi_values_Male = [
            ([73.20, 72.40, 62.40], "head"),
            ([37.55, 59.22, 54.50], "Utrunk"),
            ([38.30, 48.20, 46.80], "Mtrunk"),
            ([55.10, 61.50, 58.70], "pel"),
            ([26.90, 28.50, 15.80], "RUA"),
            ([26.90, 28.50, 15.80], "LUA"),
            ([26.50, 27.60, 12.10], "RFA"),
            ([26.50, 27.60, 12.10], "LFA"),
            ([32.90, 32.90, 14.90], "Rthigh"),
            ([32.90, 32.90, 14.90], "Lthigh"),
            ([24.90, 25.50, 10.30], "Rsha"),
            ([24.90, 25.50, 10.30], "Lsha"),
            ([24.50, 25.70, 12.40], "Rft"),
            ([24.50, 25.70, 12.40], "Lft")
        ]
        mois = {name: np.diag(values) for values, name in moi_values_Male}

        
    elif gender == 'F':
        segments_Female = [
            ("head", 0.0668),
            ("Utrunk", 0.1545),
            ("Mtrunk", 0.1465),
            ("pel", 0.1247),
            ("RUA", 0.0255),
            ("LUA", 0.0255),
            ("RFA", 0.0138),
            ("LFA", 0.0138),
            ("Rthigh", 0.1478),
            ("Lthigh", 0.1478),
            ("Rsha", 0.0481),
            ("Lsha", 0.0481),
            ("Rft", 0.0129),
            ("Lft", 0.0129)
        ]
        masses = {name: mass * fraction for name, fraction in segments_Female}
        mid_Utrunk = (combined_data_dy["Golfer:SN"] + combined_data_dy["Golfer:C7"]) / 2
        cm_head = (combined_data_dy["Golfer:HeadT"] - mid_Utrunk) * 0.4998 + mid_Utrunk
        cm_Utrunk = ((joint_center_position['Rsho'] + joint_center_position['Lsho']) / 2 - joint_center_position['Mtrunk']) * 0.7001 + joint_center_position['Mtrunk']
        cm_Mtrunk = (joint_center_position['Mtrunk'] - joint_center_position['M_pel']) * 0.5498 + joint_center_position['M_pel']
        cm_pel = (joint_center_position['M_pel'])
        cm_RUA = (joint_center_position['Rsho'] - joint_center_position['Relb']) * 0.4228 + joint_center_position['Relb']
        cm_LUA = (joint_center_position['Lsho'] - joint_center_position['Lelb']) * 0.4228 + joint_center_position['Lelb']
        cm_RFA = (joint_center_position['Relb'] - joint_center_position['Rwri']) * 0.5426 + joint_center_position['Rwri']
        cm_LFA = (joint_center_position['Lelb'] - joint_center_position['Lwri']) * 0.5426 + joint_center_position['Lwri']
        cm_Rthigh = (joint_center_position['Rhip'] - joint_center_position['Rknee']) * 0.5905 + joint_center_position['Rknee']
        cm_Lthigh = (joint_center_position['Lhip'] - joint_center_position['Lknee']) * 0.5905 + joint_center_position['Lknee']
        cm_Rsha = (joint_center_position['Rknee'] - joint_center_position['Rank']) * 0.5541 + joint_center_position['Rank']
        cm_Lsha = (joint_center_position['Lknee'] - joint_center_position['Lank']) * 0.5541 + joint_center_position['Lank']
        cm_Rft = (combined_data_dy["Golfer:RToe"] - combined_data_dy["Golfer:RHeel"]) * 0.4415 + combined_data_dy["Golfer:RHeel"]
        cm_Lft = (combined_data_dy["Golfer:LToe"] - combined_data_dy["Golfer:LHeel"]) * 0.4415 + combined_data_dy["Golfer:LHeel"]
        coms = {
            "cm_head": cm_head,
            "cm_thorax": cm_Utrunk,
            "cm_abdomen": cm_Mtrunk,
            "cm_pel": cm_pel,
            "cm_RUA": cm_RUA,
            "cm_LUA": cm_LUA,
            "cm_RFA": cm_RFA,
            "cm_LFA": cm_LFA,
            "cm_Rthigh": cm_Rthigh,
            "cm_Lthigh": cm_Lthigh,
            "cm_Rsha": cm_Rsha,
            "cm_Lsha": cm_Lsha,
            "cm_Rft": cm_Rft,
            "cm_Lft": cm_Lft,
            "cm_CH": joint_center_position["CH"]
        }
        moi_values_Female = [
            ([70.18, 66.00, 64.60], "head"),
            ([57.38, 38.62, 55.23], "Utrunk"),
            ([35.40, 43.30, 41.50], "Mtrunk"),
            ([40.20, 43.40, 44.40], "pel"),
            ([26.00, 27.80, 14.80], "RUA"),
            ([26.00, 27.80, 14.80], "LUA"),
            ([25.70, 26.10, 9.40], "RFA"),
            ([25.70, 26.10, 9.40], "LFA"),
            ([36.40, 36.90, 16.20], "Rthigh"),
            ([36.40, 36.90, 16.20], "Lthigh"),
            ([26.70, 27.10, 9.30], "Rsha"),
            ([26.70, 27.10, 9.30], "Lsha"),
            ([27.90, 29.90, 13.90], "Rft"),
            ([27.90, 29.90, 13.90], "Lft")
        ]

        #find MOI in segment
        mois = {name: np.diag(values) for values, name in moi_values_Female}
        
    return masses, coms, mois
             
mass = (np.mean(combined_data_sta_golfer['FORCE: GRF0'][:, 2]) + np.mean(combined_data_sta_golfer['FORCE: GRF1'][:, 2])) / 9.81
height = np.mean(combined_data_sta_golfer['Golfer:HeadT'][:, 2])
masses_golfer, com_position, moi_golfer = segment_mass_CM_moi(joint_center_position, combined_data_dy, mass, 'M')

total_com = ((com_position['cm_head'] * masses_golfer['head'] + 
             com_position['cm_LFA'] * masses_golfer['LFA'] + 
             com_position['cm_Lft'] * masses_golfer['Lft'] + 
             com_position['cm_Lsha'] * masses_golfer['Lsha'] + 
             com_position['cm_Lthigh'] * masses_golfer['Lthigh'] +
             com_position['cm_LUA'] * masses_golfer['LUA'] +
             com_position['cm_pel'] * masses_golfer['pel'] +
             com_position['cm_RFA'] * masses_golfer['RFA'] +
             com_position['cm_Rft'] * masses_golfer['Rft'] +
             com_position['cm_Rsha'] * masses_golfer['Rsha'] +
             com_position['cm_Rthigh'] * masses_golfer['Rthigh'] +
             com_position['cm_RUA'] * masses_golfer['RUA'] +
             com_position['cm_thorax'] * masses_golfer['Utrunk'] +
             com_position['cm_abdomen'] * masses_golfer['Mtrunk']) / 
            (masses_golfer['head'] + masses_golfer['LFA'] + masses_golfer['Lft'] +
             masses_golfer['Lsha'] + masses_golfer['Lthigh'] + masses_golfer['LUA']+ 
             masses_golfer['pel'] + masses_golfer['RFA'] + masses_golfer['Rft'] +
             masses_golfer['Rsha'] + masses_golfer['Rthigh'] + masses_golfer['RUA'] +
             masses_golfer['Utrunk'] + masses_golfer['Mtrunk']))

#%%
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

#update angel if it is over +-180 degree
def unwrap_deg(data):
    """
    input = data
    cutcome = data without gimblo lock problem
    """
    dp = np.diff(data)
    dps = np.mod(dp + np.pi, 2 * np.pi) - np.pi
    dps[np.logical_and(dps == -np.pi, dp > 0)] = np.pi
    dp_corr = dps - dp
    dp_corr[np.abs(dp) < np.pi] = 0
    data[1:] += np.cumsum(dp_corr)
    return data

""" Dr Kwon method turning z vector of the global frame -90 degree""" 
global_frame = np.matrix([[1, 0 ,0], [0, 1, 0], [0, 0, 1]])
trun = -np.pi/2
z_90 = np.array([[np.cos(trun), -np.sin(trun), 0],
              [np.sin(trun), np.cos(trun), 0],
              [0, 0, 1]])
new_global_frame = np.dot(z_90, global_frame)

dy_TX_n90 = {}
# 遍歷 dy_TX 字典
for key, matrices in dy_TX_n90.items():  
    # 創建一個新的數組來存儲每個計算後的結果
    transformed_matrices = np.empty_like(matrices)

    for i in range(matrices.shape[0]):  # 迭代每一個 frame
        transformed_matrices[i] = np.dot(matrices[i], new_global_frame)

    # 將計算後的結果存回新的字典中
    dy_TX_n90[key] = transformed_matrices

# Function to calculate euler angles for a given key and angle type
def calculate_euler_angle(result_dict, key, angle_type):
    tx = result_dict[key]
    angle = TXtoAngle(tx, angle_type)
    up_angle_x = unwrap_deg(angle[:, 0])
    up_angle_y = unwrap_deg(angle[:, 1])
    up_angle_z = unwrap_deg(angle[:, 2])
    euler_angle = np.vstack((up_angle_x, up_angle_y, up_angle_z)).T
    return euler_angle

# 定義轉換規則
segments_OA = {
    'pelvis': calculate_euler_angle(dy_TX, 'pel_TX', 'xyz')*(180/np.pi),
    'pelvis2': calculate_euler_angle(dy_TX, 'pel_TX', 'zxz')*(180/np.pi),
    'abdomen': calculate_euler_angle(dy_TX, 'abdomen_TX', 'xyz')*(180/np.pi),
    'abdomen2': calculate_euler_angle(dy_TX, 'abdomen_TX', 'zxz')*(180/np.pi),
    'thorax': calculate_euler_angle(dy_TX, 'thorax_TX', 'xyz')*(180/np.pi),
    'thorax2': calculate_euler_angle(dy_TX, 'thorax_TX', 'zxz')*(180/np.pi),
    'RUA': calculate_euler_angle(dy_TX, 'RUA_TX', 'xyz')*(180/np.pi),
    'LUA': calculate_euler_angle(dy_TX, 'LUA_TX', 'xyz')*(180/np.pi),
    'RFA': calculate_euler_angle(dy_TX, 'RFA_TX', 'xyz')*(180/np.pi),
    'LFA': calculate_euler_angle(dy_TX, 'LFA_TX', 'xyz')*(180/np.pi),
    'Rthigh': calculate_euler_angle(dy_TX, 'Rthigh_TX', 'xyz')*(180/np.pi),
    'Lthigh': calculate_euler_angle(dy_TX, 'Lthigh_TX', 'xyz')*(180/np.pi),
    'Rsha': calculate_euler_angle(dy_TX, 'Rsha_TX', 'xyz')*(180/np.pi),
    'Lsha': calculate_euler_angle(dy_TX, 'Lsha_TX', 'xyz')*(180/np.pi),
    'Rft': calculate_euler_angle(dy_TX, 'Rft_TX', 'xyz')*(180/np.pi),
    'Lft': calculate_euler_angle(dy_TX, 'Lft_TX', 'xyz')*(180/np.pi),
    'head': calculate_euler_angle(dy_TX, 'head_TX', 'xyz')*(180/np.pi),
    'CH': calculate_euler_angle(dy_TX, 'CH_TX', 'xyz')*(180/np.pi)
}

# for segment_name, segment_data in segments_OA.items():
#     # Check if segment_data is not empty
#     if segment_data.size == 0:
#         continue
    
#     # Extract frames and rotation angles for x, y, and z axes
#     frames = np.arange(segment_data.shape[0])
#     x_rotation = segment_data[:, 0]  # X-axis rotation angle
#     y_rotation = segment_data[:, 1]  # Y-axis rotation angle
#     z_rotation = segment_data[:, 2]  # Z-axis rotation angle

#     # Plot 2D graph
#     plt.figure(figsize=(8, 6))
#     plt.plot(frames, x_rotation, marker='o', linestyle='-', color='r', label='X-axis rotation (degrees)')
#     plt.plot(frames, y_rotation, marker='o', linestyle='-', color='g', label='Y-axis rotation (degrees)')
#     plt.plot(frames, z_rotation, marker='o', linestyle='-', color='b', label='Z-axis rotation (degrees)')

#     # Set title and axis labels
#     plt.title(f'{segment_name.capitalize()} Rotation over Frames')
#     plt.xlabel('Frames')
#     plt.ylabel('Rotation (degrees)')

#     # Add legend, grid, and display the plot
#     plt.legend(loc="upper right", fontsize='small')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


"joint angel = proximal segment angle - distal segment angle"
Rhip_angle = segments_OA['Rthigh'] - segments_OA['pelvis']
Lhip_angle = segments_OA['Lthigh'] - segments_OA['pelvis']
Rknee_angle = segments_OA['Rsha'] - segments_OA['Rthigh']
Lknee_angle = segments_OA['Lsha'] - segments_OA['Lthigh']
Rank_angle = segments_OA['Rft'] - segments_OA['Rsha']
Lank_angle = segments_OA['Lft'] - segments_OA['Lsha']
Ltrunk_angle = segments_OA['abdomen'] - segments_OA['pelvis']
Utrunk_angle = segments_OA['thorax'] - segments_OA['abdomen']
Rsho_angle = segments_OA['RUA'] - segments_OA['thorax']
Lsho_angle = segments_OA['LUA'] - segments_OA['thorax']
Relb_angle = segments_OA['RFA'] - segments_OA['RUA']
Lelb_angle = segments_OA['LFA'] - segments_OA['LUA']

# Create the joint angular data dictionary
joints_OA = {
    'Rhip': Rhip_angle, 'Lhip': Lhip_angle,'Rknee': Rknee_angle,'Lknee': Lknee_angle,
    'Rank': Rank_angle,'Lank': Lank_angle,'Ltrunk': Ltrunk_angle,'Utrunk': Utrunk_angle,
    'Rsho': Rsho_angle,'Lsho': Lsho_angle,'Relb': Relb_angle,'Lelb': Lelb_angle,
    }


#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

composition = {
    "head": com_position['cm_head'],
    "abdomen": com_position['cm_abdomen'],
    "thorax": com_position['cm_thorax'],
    "pel": com_position['cm_pel'],
    "LUA": com_position['cm_LUA'],
    "LFA": com_position['cm_LFA'],
    "RUA": com_position['cm_RUA'],
    "RFA": com_position['cm_RFA'],
    "Lthigh": com_position['cm_Lthigh'],
    "Lsha": com_position['cm_Lsha'],
    "Lft": com_position['cm_Lft'],
    "Rthigh": com_position['cm_Rthigh'],
    "Rsha": com_position['cm_Rsha'],
    "Rft": com_position['cm_Rft'],
    "CH": joint_center_position['CH']
}

dy_TX = {
    "head": dy_TX['head_TX'],
    "abdomen": dy_TX['abdomen_TX'],
    "thorax": dy_TX['thorax_TX'],
    "pel": dy_TX['pel_TX'],
    "LUA": dy_TX['LUA_TX'],
    "LFA": dy_TX['LFA_TX'],
    "RUA": dy_TX['RUA_TX'],
    "RFA": dy_TX['RFA_TX'],
    "Lthigh": dy_TX['Lthigh_TX'],
    "Lsha": dy_TX['Lsha_TX'],
    "Lft": dy_TX['Lft_TX'],
    "Rthigh": dy_TX['Rthigh_TX'],
    "Rsha": dy_TX['Rsha_TX'],
    "Rft": dy_TX['Rft_TX'],
    "CH": dy_TX['CH_TX'],
}

joint_center_position = {
    "Lank": joint_center_position['Lank'],
    "Lknee": joint_center_position['Lknee'],
    "Lhip": joint_center_position['Lhip'],
    "Rank": joint_center_position['Rank'],
    "Rknee": joint_center_position['Rknee'],
    "Rhip": joint_center_position['Rhip'],
    "M_pel": joint_center_position['M_pel'],
    "Mtrunk": joint_center_position['Mtrunk'],
    "Utrunk": joint_center_position['Utrunk'],
    "Head": joint_center_position['Head'],
    "Lsho": joint_center_position['Lsho'],
    "Lelb": joint_center_position['Lelb'],
    "Lwri": joint_center_position['Lwri'],
    "Rsho": joint_center_position['Rsho'],
    "Relb": joint_center_position['Relb'],
    "Rwri": joint_center_position['Rwri'],
    "mid_wri": (joint_center_position['Rwri'] + joint_center_position['Lwri']) / 2,
    "CH": joint_center_position['CH']
}

# check frames
min_frames = min([composition[key].shape[0] for key in composition])
num_points = len(composition)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# XYZ range
x_min, x_max = np.min([composition[k][:, 0] for k in composition]), np.max([composition[k][:, 0] for k in composition])
y_min, y_max = np.min([composition[k][:, 1] for k in composition]), np.max([composition[k][:, 1] for k in composition])
z_min, z_max = np.min([composition[k][:, 2] for k in composition]), np.max([composition[k][:, 2] for k in composition])

max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
x_mid, y_mid, z_mid = (x_max + x_min) / 2.0, (y_max + y_min) / 2.0, (z_max + z_min) / 2.0
ax.set_xlim(x_mid - max_range, x_mid + max_range)
ax.set_ylim(y_mid - max_range, y_mid + max_range)
ax.set_zlim(z_mid - max_range, z_mid + max_range)
ax.set_box_aspect([1, 1, 1])

# label
scatter_com = ax.scatter([], [], [], c='r', marker='o', label='COM (circle)')
scatter_joint = ax.scatter([], [], [], c='b', marker='x', label='Joint (cross)')
quivers = []
labels = {key: ax.text(0, 0, 0, '', fontsize=8) for key in composition}
labels_joint = {key: ax.text(0, 0, 0, '', fontsize=8) for key in joint_center_position}
arrow_scale = 10  # adjust array size

# create lines (point1 connects to point2)
joint_connections = [('Lank', 'Lknee'), ('Lknee', 'Lhip'), ('Lhip', 'M_pel'),
                     ('Rank', 'Rknee'), ('Rknee', 'Rhip'), ('Rhip', 'M_pel'),
                     ('M_pel', 'Mtrunk'), ('Mtrunk', 'Utrunk'), ('Utrunk', 'Head'),
                     ('Utrunk', 'Lsho'), ('Lsho', 'Lelb'), ('Lelb', 'Lwri'),
                     ('Utrunk', 'Rsho'), ('Rsho', 'Relb'), ('Relb', 'Rwri'),
                     ('mid_wri', 'CH')
 
]

lines = [ax.plot([], [], [], 'b-')[0] for _ in joint_connections]

def update(frame):
    global quivers
    scatter_com._offsets3d = (
        [composition[k][frame, 0] for k in composition],
        [composition[k][frame, 1] for k in composition],
        [composition[k][frame, 2] for k in composition]
    )
    
    scatter_joint._offsets3d = (
        [joint_center_position[k][frame, 0] for k in joint_center_position],
        [joint_center_position[k][frame, 1] for k in joint_center_position],
        [joint_center_position[k][frame, 2] for k in joint_center_position]
    )
        
    for key in composition:
        pos = composition[key][frame]
        labels[key].remove()  # 刪除舊的標籤
        labels[key] = ax.text(pos[0], pos[1], pos[2], key, fontsize=10,
                              verticalalignment='bottom', horizontalalignment='left')
    
    for key in joint_center_position:
        pos = joint_center_position[key][frame]
        labels_joint[key].remove()
        labels_joint[key] = ax.text(pos[0], pos[1], pos[2], key, fontsize=10,
                                    verticalalignment='bottom', horizontalalignment='left')
    
    
    # remove old frame
    for quiver in quivers:
        quiver.remove()
    quivers = []
    
    # update array
    for key in composition:
        origin = composition[key][frame]
        rot_matrix = dy_TX[key][frame]
        
        quivers.append(ax.quiver(*origin, *rot_matrix[0] * arrow_scale, color='r'))
        quivers.append(ax.quiver(*origin, *rot_matrix[1] * arrow_scale, color='g'))
        quivers.append(ax.quiver(*origin, *rot_matrix[2] * arrow_scale, color='b'))
    
    # update connect points
    for line, (p1, p2) in zip(lines, joint_connections):
        pos1 = joint_center_position[p1][frame]
        pos2 = joint_center_position[p2][frame]
        line.set_data([pos1[0], pos2[0]], [pos1[1], pos2[1]])
        line.set_3d_properties([pos1[2], pos2[2]])
    
    return [scatter_com, scatter_joint] + quivers + list(labels.values()) + list(labels_joint.values()) + lines

# create animation
ani = FuncAnimation(fig, update, frames=min_frames, interval=0.00001, blit=False)
arrow_legend = [
    ax.quiver([], [], [], [], [], [], color='r', label='Red = X axis'),
    ax.quiver([], [], [], [], [], [], color='g', label='Green = Y axis'),
    ax.quiver([], [], [], [], [], [], color='b', label='Blue = Z axis')
]
ax.legend()
ax.legend()
plt.show()
