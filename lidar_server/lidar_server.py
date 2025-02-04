from webbrowser import get
import json
import re
import pandas as pd
from pandas import json_normalize
from multiprocessing import Value, Array, Process
import copy
import struct
import pickle
import threading
from multiprocessing import set_start_method
import open3d as o3d
import datetime
import os
import glob
import time
import numpy as np
import base64
import urllib.parse
import requests
import matplotlib.pyplot as plt
from math import dist
import colorsys
import csv
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh import extract_relevant_features
import psutil
from flask import Flask, request
import requests

# lab
ROUGH_DBSCAN_EPS = 0.1
ROUGH_DBSCAN_MIN_POINTS = 10
CLUSTER_MIN_HEIGHT_THERESH = -0.6
CLUSTER_MAX_HEIGHT_THERESH = 1
CLUSTER_POINTS_THERESH = 60
CLUSTER_BOX_SIZE_THERESH = 2
STRICT_DBSCAN_EPS = 0.2
STRICT_DBSCAN_MIN_POINTS_DIV = 10


nw_time = time.time()

# 色数設定 (2 ** color_sep)
color_sep = 10

h_list = [0]

SOURCE_PCD = o3d.io.read_point_cloud("/ws/lidar_data/base/base.pcd")

for i in range(1, color_sep + 1):
    tmp_list = []
    for j in h_list:
        tmp_list.append(j + 1 / (2 ** i))
    h_list += tmp_list

color_list = []
index1 = []
for i in range(len(h_list)):
    rgb = colorsys.hsv_to_rgb(h_list[i], 1, 1)
    rgb_255 = [0, 0, 0]
    for j in range(3):
        rgb_255[j] = rgb[j]
    color_list.append(rgb_255)
    index1.append(i)

columns1 =["x", "y", "pred_x", "pred_y", "height", "time_fpfh_feature", "update", "tsfresh_missing_flag", "tsfresh_appear_flag"]
gallary_df = pd.DataFrame(index=index1, columns=columns1)
gallary_df['time_fpfh_feature'] = gallary_df['time_fpfh_feature'].astype('object')

tsfresh_df = pd.DataFrame()

launch_flag = 0


def cos_sim(v1, v2):
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 1
    else:
        return np.dot(v1, v2) / (norm_v1 * norm_v2)

def preprocess_point_cloud(pcd, voxel_size):

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=50))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_fpfh

def extract_feature(pcd):
    feature_list = []
    axis_aligned_bounding_box = pcd.get_axis_aligned_bounding_box()
    x_y = axis_aligned_bounding_box.get_center()[0:2]
    x = x_y[0]
    y = x_y[1]
    z = axis_aligned_bounding_box.get_max_bound()[2]
    pcd_fpfh = preprocess_point_cloud(pcd, 0.05)
    pcd_fpfh_sum = pcd_fpfh.data.sum(axis=1)
    time_fpfh = pcd_fpfh_sum /np.linalg.norm(pcd_fpfh_sum)
    feature_list.extend([x, y, z])
    feature_list.extend(time_fpfh.tolist())
    return x, y, z, time_fpfh
    return feature_list

def cal_tsfresh_time_fpfh_feature(time_fpfh_feature, id):
    global tsfresh_df
    cols = ['id', 'time']
    for i in range(1, 34):
        cols.append(f'fpfh_feature_{i}')
        
    df = pd.DataFrame(index=[], columns=cols)
    
    for i in range(len(time_fpfh_feature)):
        feature_list = [id, i] + time_fpfh_feature[i].tolist()
        record = pd.Series(feature_list, index=df.columns)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    
    # print(df)
    
    selected_features = [
        'cwt_coefficients',
        'abs_energy'
    ]
    setting = ComprehensiveFCParameters()
    fc_parameters = {
            selected_features[0]: setting[selected_features[0]],
            selected_features[1]: setting[selected_features[1]],
            }
    tsfresh_features = extract_features(
            timeseries_container=df,
            default_fc_parameters=fc_parameters,
            column_id='id',
            column_sort='time',
            column_kind=None,
            column_value=None,
            n_jobs=0
        )
    tsfresh_features = tsfresh_features.dropna(how='any', axis=1)
    tsfresh_features['id'] = id
    
    tsfresh_df = pd.concat([tsfresh_df, tsfresh_features], ignore_index=True)
    

def re_id(time_fpfh_feature, id):
    global tsfresh_df
    global gallary_df
    cols = ['id', 'time']
    for i in range(1, 34):
        cols.append(f'fpfh_feature_{i}')
        
    df = pd.DataFrame(index=[], columns=cols)
    
    for i in range(len(time_fpfh_feature)):
        feature_list = [id, i] + time_fpfh_feature[i].tolist()
        record = pd.Series(feature_list, index=df.columns)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    
    # print(df)
    
    selected_features = [
        'cwt_coefficients',
        'sum_values'
    ]
    setting = ComprehensiveFCParameters()
    fc_parameters = {
            selected_features[0]: setting[selected_features[0]],
            selected_features[1]: setting[selected_features[1]],
            }
    tsfresh_features = extract_features(
            timeseries_container=df,
            default_fc_parameters=fc_parameters,
            column_id='id',
            column_sort='time',
            column_kind=None,
            column_value=None,
            n_jobs=0
        )
    tsfresh_features = tsfresh_features.dropna(how='any', axis=1)
    
    max_cos_score = 0
    max_id = -1
    for index, gallary_tsfresh in tsfresh_df.iterrows():
        tmp_id = gallary_tsfresh['id']

        tmp_gallary_tsfresh = gallary_tsfresh.drop('id')
        print('==============================================')
        print('==============================================')
        print(tmp_gallary_tsfresh.to_numpy())
        print(tsfresh_features.to_numpy())
        cos_sim_score = cos_sim(tmp_gallary_tsfresh.to_numpy(), tsfresh_features.to_numpy()[0])
        if cos_sim_score > max_cos_score:
            max_cos_score = cos_sim_score
            max_id = tmp_id
        print('cos_sim_score', cos_sim_score)
        print('==============================================')
        print('==============================================')
            
    
    if max_cos_score > 0.7:
        gallary_df.iloc[id] = np.nan
        tsfresh_df = tsfresh_df[tsfresh_df['id'] != max_id]
        return max_id
    else:
        gallary_df.loc[id]["tsfresh_appear_flag"] = 0
        return id
    
    
    

def first_regist(pcd_arrays, feature_arrays, now_time):
    global gallary_df
    global index1
    global columns1
    gallary_df = pd.DataFrame(index=index1, columns=columns1)
    cluster_pcd = o3d.geometry.PointCloud()
    for i in range(len(feature_arrays)):
        x, y, z, time_fpfh = feature_arrays[i]
        print(gallary_df)
        id = gallary_df[gallary_df.isna().any(axis=1)].iloc[0].name
        print("register data:", id)
        gallary_df.loc[id] = [x, y, x, y, z, time_fpfh, time.time(), 0, 0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_arrays[i])
        pcd.paint_uniform_color(color_list[index1.index(id)])
        cluster_pcd += pcd
        # with open('/workspace/ws/lidar_server/opencampus_id_data.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([now_time, id, x, y, z])
    return cluster_pcd

def match_id(df):
    # print(df)
    # try:
    min_value = df.min().min()
    min_column = df.min().idxmin()
    min_row = df[min_column].idxmin()
    if min_value <= 1:
        df = df.drop(index=min_row, columns=min_column)
    return min_value, min_row, min_column, df
    # except:
    #     return 0, 0, 0, df


def normal_regist(pcd_arrays, feature_arrays, now_time):
    global gallary_df
    global index1
    global tsfresh_df
    current_val = 0.2
    pred_val = 0.8
    
    gallary_df_copy = gallary_df.copy()
    gallary_df_copy = gallary_df_copy[gallary_df_copy.loc[:]["tsfresh_missing_flag"] == 0]
    gallary_df_copy = gallary_df_copy.dropna(how="all", axis=0)
    gallary_df_copy_id_list = gallary_df_copy.index.values

    gallary_data = gallary_df_copy.to_numpy()
    
    if len(gallary_data) == 0:
        return first_regist(pcd_arrays, feature_arrays, now_time)
    
    
    else:
        exist_gallary_df = pd.DataFrame(gallary_df_copy, columns=gallary_df_copy_id_list)
        objective_array = []
        print(len(gallary_data))
        for i in range(len(feature_arrays)):
            tmp = []
            for j in range(len(gallary_data)):
                objective = (((gallary_data[j][0] - feature_arrays[i][0]) ** 2 + (gallary_data[j][1] - feature_arrays[i][1]) ** 2 + (gallary_data[j][4] - feature_arrays[i][2]) ** 2) ** 0.5) * current_val + (((gallary_data[j][2] - feature_arrays[i][0]) ** 2 + (gallary_data[j][3] - feature_arrays[i][1]) ** 2 + (gallary_data[j][4] - feature_arrays[i][2]) ** 2) ** 0.5) * pred_val
                tmp.append(objective)
            objective_array.append(tmp)
            
        objective_df = pd.DataFrame(objective_array, columns=gallary_df_copy_id_list)

        
        cluster_pcd = o3d.geometry.PointCloud()
        for i in range(len(objective_df)):
            # try:
            if objective_df.empty:
                break
            else:
                min_value, min_row, min_column, objective_df = match_id(objective_df)
                # if min_value == 0 and min_row == 0 and min_column == 0:
                #     break
                # print(min_value, min_row, min_column)
                if min_value > 1.5:
                    break
                else:
                    id = min_column
                    x, y, z, time_fpfh = feature_arrays[min_row]
                    time_fpfh_feature = gallary_df.loc[id]["time_fpfh_feature"]
                    
                    prev_x = gallary_df.loc[id]["x"]
                    prev_y = gallary_df.loc[id]["y"]

                    if len(time_fpfh_feature) == 10:
                        time_fpfh_feature = time_fpfh_feature[1:]
                    
                    if len(time_fpfh_feature) == 0 or len(time_fpfh_feature) == 33:
                        time_fpfh_feature = np.concatenate(([time_fpfh_feature], [time_fpfh]), axis=0)
                    else:
                        time_fpfh_feature = np.concatenate((time_fpfh_feature, [time_fpfh]), axis=0)
                    
                    if len(time_fpfh_feature) == 10 and gallary_df.loc[id]["tsfresh_appear_flag"] == 1:
                        id = re_id(time_fpfh_feature, id)
                        print('re_id', id)
                        print('x', x)
                        print('y', y)
                    
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(pcd_arrays[min_row])
                    if gallary_df.loc[id]["tsfresh_appear_flag"] == 0:
                        gallary_df.loc[id] = [x, y, 2 * x - prev_x, 2 * y - prev_y, z, time_fpfh_feature, time.time(), 0, 0]
                        pcd.paint_uniform_color(color_list[index1.index(id)])
                        cluster_pcd += pcd
                    else:
                        gallary_df.loc[id] = [x, y, 2 * x - prev_x, 2 * y - prev_y, z, time_fpfh_feature, time.time(), 0, 1]
                        pcd.paint_uniform_color([0, 0, 0])
                        cluster_pcd += pcd
                    # with open('/workspace/ws/lidar_server/opencampus_id_data.csv', 'a') as f:
                    #     writer = csv.writer(f)
                    #     writer.writerow([now_time, id, x, y, z])
            # except:
            #     break

        print(objective_df)
                    
                    
        
        if len(objective_df) > 0:
            new_id_list = objective_df.index.values
            tmp_gallary_data_df = gallary_df.copy()
            tmp_gallary_data_df = tmp_gallary_data_df[tmp_gallary_data_df.loc[:]["tsfresh_missing_flag"] == 0]
            tmp_gallary_data = tmp_gallary_data_df.to_numpy()
            for i in new_id_list:
                x, y, z, time_fpfh = feature_arrays[i]
                
                distances = np.sum((tmp_gallary_data[:, :2] - np.array([x, y]))**2, axis=1) ** 0.5
                min_distance_gallary = np.nanmin(distances)
                print('caulurate distance')

                if min_distance_gallary <= 0.5:
                    print('ignore the cluster')
                    print(min_distance_gallary)
                    continue
                
                id = gallary_df[gallary_df.isna().any(axis=1)].iloc[0].name

                gallary_df.loc[id] = [x, y, x, y, z, time_fpfh, time.time(), 0, 1]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcd_arrays[i])
                pcd.paint_uniform_color([0, 0, 0])
                cluster_pcd += pcd
                # with open('/workspace/ws/lidar_server/opencampus_id_data.csv', 'a') as f:
                #     writer = csv.writer(f)
                #     writer.writerow([now_time, id, x, y, z])
        
        diffuse_gallary_id_list = objective_df.columns.values
        
        for i in diffuse_gallary_id_list:
            time_fpfh_feature = gallary_df.loc[i]["time_fpfh_feature"]
            if time.time() - gallary_df['update'][i] >= 6 and gallary_df.loc[i]["tsfresh_missing_flag"] == 0:
                if len(time_fpfh_feature) < 10 or len(time_fpfh_feature) == 33:
                    gallary_df.iloc[i] = np.nan
                    print('delete missing data')
                else:
                    cal_tsfresh_time_fpfh_feature(time_fpfh_feature, i)
                    gallary_df.loc[i]["tsfresh_missing_flag"] = 1
        
        gallary_id_list = gallary_df.index.values
        for i in gallary_id_list:
            if time.time() - gallary_df['update'][i] >= 1800 and gallary_df.loc[i]["tsfresh_missing_flag"] == 1:
                gallary_df.iloc[i] = np.nan
                tsfresh_df = tsfresh_df[tsfresh_df['id'] != i]
                print('delete missing data')
        
        print(gallary_df)
        return cluster_pcd

def divide_cluster(pcd, pcd_arrays, thr_points_num):
    global index1
    device = o3d.core.Device("CUDA:0")
    cpu_device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float64
    x_y_df = pd.DataFrame(pcd.points,
                  columns = ["x","y","z"],)
    x_y_df["z"] = 0
    
    z_df = pd.DataFrame(pcd.points,
                  columns = ["x","y","z"],)
    
    x_y_pcd = o3d.geometry.PointCloud()
    x_y_pcd.points = o3d.utility.Vector3dVector(x_y_df.to_numpy())
    thresh_min_points = int(len(x_y_df.index)/STRICT_DBSCAN_MIN_POINTS_DIV)
    
    tmp_x_y_pcd = o3d.t.geometry.PointCloud(device)
    tmp_x_y_pcd.point.positions = o3d.core.Tensor(np.asarray(x_y_pcd.points), dtype, device)
    
    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        labels_tensor = tmp_x_y_pcd.cluster_dbscan(eps=STRICT_DBSCAN_EPS, min_points=thresh_min_points, print_progress=False)
        labels_tensor_cpu = labels_tensor.to(cpu_device)
        labels = np.array(labels_tensor_cpu.cpu().numpy())
        
    tmp_cluster_pcd = o3d.geometry.PointCloud()
    
    min_cluster_points = 0

    if labels.size != 0: 
        for i in range(labels.max() + 1):
            pc_indices = np.where(labels == i)[0]
            xyz = np.asarray(z_df.to_numpy())[pc_indices, :]
            if CLUSTER_MAX_HEIGHT_THERESH > xyz.T[2].max() > CLUSTER_MIN_HEIGHT_THERESH:
                if len(xyz) >= CLUSTER_POINTS_THERESH:
                    tmp_pcd = o3d.geometry.PointCloud()
                    tmp_pcd.points = o3d.utility.Vector3dVector(xyz)
                    
                    plane_xyz = np.asarray(x_y_df.to_numpy())[pc_indices, :]
                    plane_pcd = o3d.geometry.PointCloud()
                    plane_pcd.points = o3d.utility.Vector3dVector(plane_xyz)
                    if min_cluster_points == 0:
                        min_cluster_points = len(tmp_pcd.points)
                    elif min_cluster_points > len(tmp_pcd.points):
                        min_cluster_points = len(tmp_pcd.points)
                    pcd_arrays.append(xyz)
                    
                    tmp_cluster_pcd += tmp_pcd
        
        dists = pcd.compute_point_cloud_distance(tmp_cluster_pcd)
        dists = np.asarray(dists)
        ind = np.where(dists > 0.03 )[0]
        noise_pcd = pcd.select_by_index(ind)
        if len(noise_pcd.points) >= min_cluster_points:
            pcd_arrays = divide_cluster(noise_pcd, pcd_arrays, min_cluster_points)
    return pcd_arrays

def cluster(pcd_numpy):
    ut = time.time()
    cpu_device = o3d.core.Device("CPU:0")
    device = o3d.core.Device("CUDA:0")
    dtype = o3d.core.float64

    tmp_target = o3d.t.geometry.PointCloud()
    tmp_target.point.positions = o3d.core.Tensor(pcd_numpy, dtype, device)
    tmp_target_numpy = tmp_target.point.positions.cpu().numpy().copy()
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(tmp_target_numpy)
    dists = target.compute_point_cloud_distance(SOURCE_PCD)
    dists = np.asarray(dists)
    ind = np.where(dists > 0.1)[0]
    tmp_object_pcd = target.select_by_index(ind)
    cl, ind = tmp_object_pcd.remove_radius_outlier(nb_points=10, radius=0.5)
    tmp_object_pcd = tmp_object_pcd.select_by_index(ind)
    object_pcd = o3d.t.geometry.PointCloud(device)
    object_pcd.point.positions = o3d.core.Tensor(np.asarray(tmp_object_pcd.points), dtype, device)
    
    back_substruction_time = time.time() - ut
    
    ut = time.time()

    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        labels_tensor = object_pcd.cluster_dbscan(eps=ROUGH_DBSCAN_EPS, min_points=ROUGH_DBSCAN_MIN_POINTS, print_progress=False)
        labels_tensor_cpu = labels_tensor.to(cpu_device)
        labels = np.array(labels_tensor_cpu.cpu().numpy())


    cluster_pcd = o3d.geometry.PointCloud()
    
    pcd_arrays = []
    
    if labels.size != 0: 
        for i in range(labels.max() + 1):
            pc_indices = np.where(labels == i)[0]

            if pc_indices.size > 0:
                xyz = np.asarray(tmp_object_pcd.points)[pc_indices, :]
                
                if CLUSTER_MAX_HEIGHT_THERESH > xyz.T[2].max() > CLUSTER_MIN_HEIGHT_THERESH:
                    if len(xyz) >= CLUSTER_POINTS_THERESH:
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(xyz)
                        bounding_box = pcd.get_oriented_bounding_box()
                        size_bounding_box = bounding_box.get_max_bound() - bounding_box.get_min_bound()
                        ts_size = size_bounding_box[0] * size_bounding_box[1]
                        if ts_size >= CLUSTER_BOX_SIZE_THERESH:
                            pcd_arrays = divide_cluster(pcd, pcd_arrays, 0)
                        else:
                            pcd_arrays.append(xyz)
                        cluster_pcd += pcd
    
    cluster_time = time.time() - ut
    
    # o3d.io.write_point_cloud(f"/work_space/lidar_data/eva/reid/5/{datetime.datetime.now() + datetime.timedelta(hours=9)}.pcd", cluster_pcd)

    return pcd_arrays, back_substruction_time, cluster_time, len(pcd_arrays)
    
def sub_back(pcd_numpy, now_time, flag, back_substruction_time, cluster_time, data_size, mbps, process_time):
    global gallary_df
    global nw_time
    global color_list
    global index1
    global launch_flag
    feature_arrays = []
    cluster_pcd = o3d.geometry.PointCloud()
    after_data_len = data_size
    
    if flag == 1:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_numpy)
        pcd_numpy, back_substruction_time, cluster_time, cluster_size = cluster(pcd_numpy)
        data = pickle.dumps(pcd_numpy)
        after_data_len = len(data)/1024/1024
        print('*******************')
        print(back_substruction_time, cluster_time, after_data_len)
        print('*******************')
    processing_time = [back_substruction_time, cluster_time, after_data_len, mbps, flag]
    
    processing_time = np.array(processing_time)
    data = pickle.dumps(processing_time)
    headers = {
        'Content-Type': 'application/octet-stream'
    }
    try:
        response = requests.post(
            'http://192.168.107.50:49228/processing_time',  # Replace with your server's IP and endpoint
            data=data,
            headers=headers
        )
        print(f"Data sent, response status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred while sending data: {e}")

    
    if len(pcd_numpy) != 0:
        ut = time.time()
        
        for human_pcd_data in pcd_numpy:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(human_pcd_data)
            feature_arrays.append(extract_feature(pcd))

        extract_time = time.time() - ut
        print("extract feature time: ", extract_time)

        ut = time.time()
        cluster_pcd = normal_regist(list(pcd_numpy), feature_arrays, now_time)
        
        launch_flag = 1
        cluster_pcd = cluster_pcd.voxel_down_sample(voxel_size=0.1)
        
        regist_time = time.time() - ut
        print("register data time: ", regist_time)
        
    else:
        launch_flag = 0
        extract_time = 0
        regist_time = 0

    new_p_pcd = cluster_pcd
    
    new_p_pcd_numpy = np.asarray(new_p_pcd.points, dtype='float32')
    points = new_p_pcd_numpy.tobytes("C")
    
    pcd_color = np.asarray(new_p_pcd.colors, dtype='float32')
    colors_np = np.asarray(pcd_color * 255, dtype='uint8')
    colors = colors_np.tobytes("C")
    
    # with open('/workspace/ws/lidar_server/new_server_process_11_13_s.csv', 'a') as f:
    #         writer = csv.writer(f)
    #         writer.writerow([datetime.datetime.now() + datetime.timedelta(hours=9), mbps, flag, back_substruction_time, cluster_time, extract_time, regist_time, process_time])
   
    return points, colors

class VizClient(object):
    def __init__(self, host: str = "192.168.50.32", port: int = 8000) -> None:
        self._url = "http://%s:%d" % (host, port)

    def _encode(self, s: bytes) -> str:
        return base64.b64encode(s).decode("utf-8")

    def post_pointcloud(self, pcd_numpy, flag, name, back_substruction_time, cluster_time, data_size, mbps, process_time) -> requests.models.Response:
        ut = time.time()
        now_time = datetime.datetime.now() + datetime.timedelta(hours=9)
        points, colors = sub_back(pcd_numpy, now_time, flag, back_substruction_time, cluster_time, data_size, mbps, process_time)
        # with open('/workspace/ws/lidar_server/server_process.csv', 'a') as f:
        #         writer = csv.writer(f)
        #         writer.writerow([datetime.datetime.now() + datetime.timedelta(hours=9), len(pcd_numpy), time.time() - ut])
        print("************************************************")
        print("cloud: ", time.time() - ut)
        print("************************************************")

        name = str(id(pcd_numpy)) if name == "" else name
        response = requests.post(
            urllib.parse.urljoin(self._url, "pointcloud/store"),
            json={"name": name, "points": self._encode(points), "colors": self._encode(colors)},
        )
        # print(type(points))
        return response

def put_dummy_on_cuda():
    ut = time.time()
    device = o3d.core.Device("CUDA:0")
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3d.core.Tensor(np.empty((1, 3)), dtype, device)
    print("************************************************")
    print("put_dummy_on_cuda: ", time.time() - ut)
    print("************************************************")
    
def recv_data(sock, addr, q_server_back_substruction_time, q_server_cluster_time, q_server_after_cluster_size):
    device = o3d.core.Device("CUDA:0")
    dtype = o3d.core.float32
    client = VizClient()
    while True:
        ut = time.time()
        while len(sock.data) < 4:
            sock.data += sock.conn.recv(12288)
        packed_flag = sock.data[:4]
        flag = struct.unpack("I", packed_flag)[0]
        sock.data = sock.data[4:]
        
        while len(sock.data) < sock.payload_size:
            sock.data += sock.conn.recv(4)
        packed_msg_size = sock.data[:sock.payload_size]
        msg_size = struct.unpack("I", packed_msg_size)[0]
        sock.data = sock.data[4:]
        
        while len(sock.data) < msg_size:
            sock.data += sock.conn.recv(msg_size)
        
        process_time = time.time()-ut

        frame_data = sock.data[:msg_size]
        sock.data = sock.data[msg_size:]
        data_len = len(frame_data)
        pcd_data = pickle.loads(frame_data)
        
        data_size = (data_len + 4 + 4) / 1024 / 1024
        print('data_size: ', data_size)
        total_bits = (data_len + 4 + 4) * 8
        mbps = total_bits / process_time / (1024 * 1024)
        
        print('mbps: ', mbps)

        try:
            res = client.post_pointcloud(pcd_data, flag, 'test', q_server_back_substruction_time, q_server_cluster_time, q_server_after_cluster_size)
        except:
            print("cannot connect web server")
            pass

        print("save data")


def create_flask_app():
    app = Flask(__name__)
    client = VizClient()

    @app.route('/recv_data', methods=['POST'])
    def recv_data():
        ut = time.time()
        try:
            # Assume 'Flag' is sent as a header
            flag = int(request.headers.get('Flag', 0))

            # Read the pickled data from the request body
            data = request.data
            
            data_received = pickle.loads(data)
            data_len = len(data)
            
            back_substruction_time = data_received.get('back_substruction_time')
            cluster_time = data_received.get('cluster_time')
            pcd_data = data_received.get('point_cloud')
            
            process_time = time.time() - ut

            # Calculate data size and throughput
            data_size = data_len / (1024 * 1024)
            print('data_size: ', data_size)
            total_bits = data_len * 8
            mbps = total_bits / process_time / (1024 * 1024)
            print('mbps: ', mbps)

            res = client.post_pointcloud(pcd_data, flag, 'test', back_substruction_time, cluster_time, data_size, mbps, process_time)

        except Exception as e:
            print(f"Error processing data: {e}")
            return "Error", 500

        return "OK", 200

    app.run(host='192.168.50.32', port=49221)

def main():
    put_dummy_on_cuda()
    create_flask_app()

if __name__ == '__main__':
    main()