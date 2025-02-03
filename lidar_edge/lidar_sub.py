#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import ros_numpy
import datetime
import torch.multiprocessing as mp
import time
import open3d.core as o3c
import os, glob
import json
import pickle
import csv
import os
import pandas as pd
from flask import Flask, request
import requests
from statistics import mean

# lab
ROUGH_DBSCAN_EPS = 0.1
ROUGH_DBSCAN_MIN_POINTS = 10
CLUSTER_MIN_HEIGHT_THERESH = -0.6
CLUSTER_MAX_HEIGHT_THERESH = 1
CLUSTER_POINTS_THERESH = 60
CLUSTER_BOX_SIZE_THERESH = 2
STRICT_DBSCAN_EPS = 0.2
STRICT_DBSCAN_MIN_POINTS_DIV = 10

SOURCE_PCD = o3d.io.read_point_cloud("/ws/lidar_data/base/base.pcd")

SERVER_FLAG = False
CAL_FLAG = True
CAL_TIMES = 0

def put_dummy_on_cuda():
    ut = time.time()
    device = o3d.core.Device("CUDA:0")
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3d.core.Tensor(np.empty((1, 3)), dtype, device)
    print(pcd.is_cuda)
    print("************************************************")
    print("put_dummy_on_cuda: ", time.time() - ut)
    print("************************************************")

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

def callback(point_cloud, args):
    ut = time.time()
    code = args[0]
    q = args[1]
    print(ut)
    
    # デバイスの設定
    device = o3d.core.Device("CUDA:0")
    dtype = o3d.core.float32
    
    # 回転行列を含むJSONファイルの読み取り
    config = {}
    with open("/work_space/lidar_data/matrix_config/matrix_config.json", mode="r") as f:
            config = json.load(f)
    
    # 計測時の時間を取得
    dt_now = datetime.datetime.now()
    dt_now_str = dt_now.strftime('%Y_%m_%d_%H_%M_%S')
    
    # ポイントクラウドオブジュエクトからNumpy配列に変換
    pc = ros_numpy.numpify(point_cloud)
    points=np.zeros((pc.shape[0],3))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']
    
    # GPUのメモリにPCDデータを乗せる
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3d.core.Tensor(points, dtype, device)
    # ボクセル化(1cm)
    voxel_pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # 点群データの回転
    voxel_pcd_rotated = voxel_pcd.transform(np.array(config[code]))
    
    if q.full():
        # print('*******************************************************************')
        # print('*******************************************************************')
        print('full error')
        # print('*******************************************************************')
        # print('*******************************************************************')
    
    if q.qsize() >= 1:
        # print('clear que')
        q.get()
    q.put(voxel_pcd_rotated)
    
    print("************************************************")
    print(code, " que: ", time.time() - ut, time.time())
    print("************************************************")
    
    
    
    for file in glob.glob("/work_space/lidar_data/" + code + "/*.pcd", recursive=True):
        os.remove(file)

    o3d.t.io.write_point_cloud("/work_space/lidar_data/" + code + "/" + dt_now_str + ".pcd", pcd)
    # print("save " + code + " data")

    
def connect_ros(q_3JEDKBS001G9601, q_3JEDKC50014U011, q_3JEDL3N0015X621, q_3JEDL76001L4201):
    print('connect_ros')
    put_dummy_on_cuda()
    print('waaaaai')
    rospy.init_node('lidar_subscriber', anonymous=True)
    rospy.Subscriber("/livox/lidar_3JEDKBS001G9601", PointCloud2, callback, ("3JEDKBS001G9601", q_3JEDKBS001G9601))
    rospy.Subscriber("/livox/lidar_3JEDKC50014U011", PointCloud2, callback, ("3JEDKC50014U011", q_3JEDKC50014U011))
    rospy.Subscriber("/livox/lidar_3JEDL3N0015X621", PointCloud2, callback, ("3JEDL3N0015X621", q_3JEDL3N0015X621))
    rospy.Subscriber("/livox/lidar_3JEDL76001L4201", PointCloud2, callback, ("3JEDL76001L4201", q_3JEDL76001L4201))
    rospy.spin()
    
def combine_pcd(q_3JEDKBS001G9601, q_3JEDKC50014U011, q_3JEDL3N0015X621, q_3JEDL76001L4201, q_mbps, q_flag, q_server_back_substruction_time, q_server_cluster_time, q_server_after_cluster_size):
    global REBOOT_FLAG
    global SERVER_FLAG
    global CAL_FLAG
    global CAL_TIMES
    

    # Server URL for HTTP POST requests
    server_url = "http://192.168.100.120:49221/recv_data"  # Replace with your server's IP and port

    send_data_time = time.time()
    band_width = 0
    server_back_substruction_time = 0
    server_cluster_time = 0
    server_cluster_size = 0
    flag_pre = 0
    
    e_total_back_substruction_time = []
    e_total_cluster_time = []
    s_total_back_substruction_time = []
    s_total_cluster_time = []
    
    back_substruction_coefficient = 3
    cluster_coefficient = 3
    cal_flag_time = time.time()

    while True:
        # Update bandwidth and server processing times from queues
        if q_flag.qsize() >= 1:
            flag_pre = q_flag.get()
        if q_mbps.qsize() >= 1:
            band_width = q_mbps.get()
        if q_server_back_substruction_time.qsize() >= 1:
            tmp_server_back_substruction_time = q_server_back_substruction_time.get()
            if flag_pre == 1:
                server_back_substruction_time = tmp_server_back_substruction_time
        if q_server_cluster_time.qsize() >= 1:
            tmp_server_cluster_time = q_server_cluster_time.get()
            if flag_pre == 1:
                server_cluster_time = tmp_server_cluster_time
        if q_server_after_cluster_size.qsize() >= 1:
            server_cluster_size = q_server_after_cluster_size.get()
        
        print('******************************************')
        print(f"server_back_substruction_time: {server_back_substruction_time}")
        print(f"server_cluster_time: {server_cluster_time}")
        print('******************************************')

        # print('band_width: ', band_width)

        # Get point cloud data from queues
        pcd_3JEDKBS001G9601 = q_3JEDKBS001G9601.get()
        pcd_3JEDKC50014U011 = q_3JEDKC50014U011.get()
        pcd_3JEDL3N0015X621 = q_3JEDL3N0015X621.get()
        pcd_3JEDL76001L4201 = q_3JEDL76001L4201.get()

        ut = time.time()

        # Combine point clouds and perform voxel downsampling
        combined_pcd = pcd_3JEDKBS001G9601 + pcd_3JEDKC50014U011 + pcd_3JEDL3N0015X621 + pcd_3JEDL76001L4201
        combined_voxel_pcd = combined_pcd.voxel_down_sample(voxel_size=0.05)

        # Save the point cloud at a specific time (optional)
        now_dt = datetime.datetime.now()
        if 18 == now_dt.hour and 0 < now_dt.minute < 5:
            o3d.t.io.write_point_cloud("/work_space/lidar_data/base/base.pcd", combined_voxel_pcd)

        combined_voxel_numpy = combined_voxel_pcd.point.positions.cpu().numpy().copy()
        data_to_send = {
            'point_cloud': combined_voxel_numpy,
            'back_substruction_time': 0,
            'cluster_time': 0
        }
        serialized_data = pickle.dumps(data_to_send)
        before_data_len = len(serialized_data) / 1024 / 1024
        
        
        if time.time() - cal_flag_time > 600:
            CAL_FLAG = True

        if CAL_FLAG:
            print(e_total_back_substruction_time)
            print(e_total_cluster_time)
            print(s_total_back_substruction_time)
            print(s_total_cluster_time)
            if CAL_TIMES == 0:
                e_total_back_substruction_time = []
                e_total_cluster_time = []
                s_total_back_substruction_time = []
                s_total_cluster_time = []

            if CAL_TIMES % 2 == 0:
                SERVER_FLAG = True
            else:
                SERVER_FLAG = False
            CAL_TIMES += 1
            if CAL_TIMES == 20:
                CAL_FLAG = False
                e_mean_back_substruction_time = mean(e_total_back_substruction_time)
                e_mean_cluster_time = mean(e_total_cluster_time)
                s_mean_back_substruction_time = mean(s_total_back_substruction_time)
                s_mean_cluster_time = mean(s_total_cluster_time)
                back_substruction_coefficient = e_mean_back_substruction_time / s_mean_back_substruction_time
                cluster_coefficient = e_mean_cluster_time / s_mean_cluster_time
                CAL_TIMES = 0
                cal_flag_time = time.time()

                
                
            

        if SERVER_FLAG:
            be_pack_data = time.time()
            flag = '1'  # Indicate processing is to be done on the server
            print('time: ', time.time())

            headers = {
                'Flag': flag,
                'Content-Type': 'application/octet-stream',
            }
            
              # Data size in MB

            # Send data via HTTP POST
            try:
                response = requests.post(
                    server_url,
                    headers=headers,
                    data=serialized_data
                )
                print('Data sent, response status code:', response.status_code)
            except Exception as e:
                print(f"Error sending data: {e}")

            # print('send data time: ', time.time() - be_pack_data)
            # print('left_time: ', (server_back_substruction_time + server_cluster_time))
            # print('right_time: ', (before_data_len + server_cluster_size) * 8 / band_width)
            
            print('******************************************')
            print(f"server_back_substruction_time: {server_back_substruction_time}")
            print(f"server_cluster_time: {server_cluster_time}")
            print(f"before_data_len: {before_data_len}")
            print(f"server_cluster_size: {server_cluster_size}")
            print(f"band_width: {band_width}")
            print('******************************************')
            
            s_total_back_substruction_time.append(server_back_substruction_time)
            s_total_cluster_time.append(server_cluster_time)

            if all([band_width, server_cluster_size, server_back_substruction_time, server_cluster_time]):
                if (back_substruction_coefficient - 1)*server_back_substruction_time + (cluster_coefficient - 1)*server_cluster_time < ((before_data_len + server_cluster_size) * 8 / band_width):
                    SERVER_FLAG = False

            # with open('/work_space/lidar_data/process_change.csv', 'a') as f:
            #     writer = csv.writer(f)
            #     writer.writerow([
            #         datetime.datetime.now() + datetime.timedelta(hours=9),
            #         band_width,
            #         server_back_substruction_time,
            #         server_cluster_time,
            #         1
            #     ])

        else:
            # Process data locally
            combined_voxel_numpy, back_substruction_time, cluster_time, cluster_size = cluster(combined_voxel_numpy)

            flag = '0'  # Indicate processing has been done locally
            print('time: ', time.time())

            headers = {
                'Flag': flag,
                'Content-Type': 'application/octet-stream',
            }
            
            data_to_send = {
                'point_cloud': combined_voxel_numpy,
                'back_substruction_time': back_substruction_time,
                'cluster_time': cluster_time
            }
            
            serialized_data = pickle.dumps(data_to_send)
            after_data_len = len(serialized_data) / 1024 / 1024  # Data size in MB

            # Send data via HTTP POST
            try:
                response = requests.post(
                    server_url,
                    headers=headers,
                    data=serialized_data
                )
                print('Data sent, response status code:', response.status_code)
            except Exception as e:
                print(f"Error sending data: {e}")
            
            e_total_back_substruction_time.append(back_substruction_time)
            e_total_cluster_time.append(cluster_time)

            if band_width != 0:
                # print('left_time: ', (back_substruction_time + cluster_time))
                # print('right_time: ', (before_data_len + after_data_len) * 8 / band_width)
                print('******************************************')
                print(f"back_substruction_time: {back_substruction_time}")
                print(f"cluster_time: {cluster_time}")
                print(f"before_data_len: {before_data_len}")
                print(f"after_data_len: {after_data_len}")
                print(f"band_width: {band_width}")
                print('******************************************')
                if (back_substruction_coefficient - 1)*back_substruction_time/back_substruction_coefficient + (cluster_coefficient - 1)*cluster_time/cluster_coefficient > ((before_data_len + after_data_len) * 8 / band_width):
                    SERVER_FLAG = True

        with open('/work_space/lidar_data/process_change.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now() + datetime.timedelta(hours=9),
                band_width,
                flag,
                back_substruction_coefficient,
                cluster_coefficient
            ])

        send_data_time = time.time()

        o3d.t.io.write_point_cloud("/work_space/lidar_data/combined_pcd/combined.pcd", combined_voxel_pcd)
        print("save combined data")

        
def share_processing_time_server(q_server_back_substruction_time, q_server_cluster_time, q_server_after_cluster_size, q_mbps, q_flag):
    from multiprocessing import Process, Manager

    app = Flask(__name__)

    @app.route('/processing_time', methods=['POST'])
    def processing_time():
        try:
            data = request.data
            server_processing_time = pickle.loads(data)

            print('server_processing_time: ', server_processing_time)

            q_server_back_substruction_time.put(server_processing_time[0])
            q_server_cluster_time.put(server_processing_time[1])
            q_server_after_cluster_size.put(server_processing_time[2])
            q_mbps.put(server_processing_time[3])
            q_flag.put(server_processing_time[4])

            return 'OK', 200
        except Exception as e:
            print(f"Error processing data: {e}")
            return 'Error', 500

    app.run(host='192.168.100.195', port=49228)  # Adjust port if necessary
        

def main():
    if mp.get_start_method() == 'fork':
        mp.set_start_method('spawn', force=True)
        
    manager = mp.Manager()
        
    q_3JEDKBS001G9601 = manager.Queue()
    q_3JEDKC50014U011 = manager.Queue()
    q_3JEDL3N0015X621 = manager.Queue()
    q_3JEDL76001L4201 = manager.Queue()
    
    q_server_back_substruction_time = mp.Queue()
    q_server_cluster_time = mp.Queue()
    q_server_after_cluster_size = mp.Queue()
    q_mbps = mp.Queue()
    q_flag = mp.Queue()
    
    

    p_connect_ros = mp.Process(target=connect_ros, args=(q_3JEDKBS001G9601, q_3JEDKC50014U011, q_3JEDL3N0015X621, q_3JEDL76001L4201))
    p_connect_ros.start()
    
    p_combine_pcd = mp.Process(target=combine_pcd, args=(q_3JEDKBS001G9601, q_3JEDKC50014U011, q_3JEDL3N0015X621, q_3JEDL76001L4201, q_mbps, q_flag, q_server_back_substruction_time, q_server_cluster_time, q_server_after_cluster_size))
    p_combine_pcd.start()
    
    p_share_processing_time_server = mp.Process(target=share_processing_time_server, args=(q_server_back_substruction_time, q_server_cluster_time, q_server_after_cluster_size, q_mbps, q_flag))
    p_share_processing_time_server.start()
    
    p_connect_ros.join()
    p_combine_pcd.join()
    p_share_processing_time_server.join()
    

if __name__ == '__main__':
    main()