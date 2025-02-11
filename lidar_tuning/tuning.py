from unittest import result
import open3d as o3d
import numpy as np
import copy
import pandas as pd
import sys, os
import glob
import json
import time

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])
def draw_result(source, source_1):
    source_temp = copy.deepcopy(source)
    source_1_temp = copy.deepcopy(source_1)
    source_temp.paint_uniform_color([1, 0.706, 0])
    source_1_temp.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source_temp, source_1_temp],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source_path, target_path, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)





    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.99),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000000, 1000))
    
    return result


def execute_icp_registration(source, target, threshold, trans_init):
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000000000))
    return reg_p2p

def remove_plane(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
    
    return outlier_cloud

def tmp_plane(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    floor_pcd = pcd.select_by_index(inliers)
    
    
    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    an_wall_pcd = outlier_cloud.select_by_index(inliers)
    outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
    
    
    
    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    wall_pcd = outlier_cloud.select_by_index(inliers)
    outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
    
    
    
    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    up_pcd = outlier_cloud.select_by_index(inliers)
    outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
    
    
    target_pcd = outlier_cloud
    
    
    
    return target_pcd
def n_tmp_plane(pcd):
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    floor_pcd = pcd.select_by_index(inliers)
    
    
    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    an_wall_pcd = outlier_cloud.select_by_index(inliers)
    outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
    
    
    
    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    wall_pcd = outlier_cloud.select_by_index(inliers)
    outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
    
    
    
    plane_model, inliers = outlier_cloud.segment_plane(distance_threshold=0.05,
                                         ransac_n=3,
                                         num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    up_pcd = outlier_cloud.select_by_index(inliers)
    outlier_cloud = outlier_cloud.select_by_index(inliers, invert=True)
    
    
    target_pcd = an_wall_pcd + outlier_cloud + up_pcd
    
    
    
    return target_pcd

def prepare_dataset_remove_outlier(source_path, target_path, voxel_size, kind):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    
    inlier_cloud_down, inlier_cloud_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    # o3d.io.write_point_cloud('out_source.pcd', inlier_cloud_down)
    # o3d.io.write_point_cloud('out_target.pcd', target_down)
    return source, target, inlier_cloud_down, target_down, inlier_cloud_fpfh, target_fpfh

def calculate_rotation_matrix(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    axis = np.cross(v1, v2)
    axis_length = np.linalg.norm(axis)
    if axis_length == 0:
        return np.identity(3)

    axis = axis / axis_length

    angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.identity(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return R

def main():
    with open("/ws/lidar_data/matrix_config.json", mode="r") as f:
            config = json.load(f)

    rough_voxel_size = 0.1
    precision_voxel_size = 0.01
    threshold = 0.01
    
    
    # 1
    source_path = glob.glob("/ws/lidar_data/3JEDKBS001G9601/**.pcd")[0]
    base_path = "/ws/lidar_tuning/combine_data/combine_3JEDKBS001G9601.pcd"
    inlier_cloud, target, inlier_cloud_down, target_down, inlier_cloud_fpfh, target_fpfh = prepare_dataset_remove_outlier(source_path, base_path, rough_voxel_size, 1)
    result_ransac = execute_global_registration(inlier_cloud_down, target_down,
                                    inlier_cloud_fpfh, target_fpfh,
                                    rough_voxel_size)
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source_path, base_path, precision_voxel_size)
    result_icp = execute_icp_registration(source_down, target_down, threshold, result_ransac.transformation)
    print(result_icp)
    source.transform(result_icp.transformation)
    target.paint_uniform_color([0, 1, 0])
    source.paint_uniform_color([1, 0, 0])
    pcd = source + target
    o3d.io.write_point_cloud('new_3JEDKBS001G9601.pcd', pcd)
    print(result_icp.transformation.tolist())
    config['3JEDKBS001G9601'] = result_icp.transformation.tolist()
    
    # 2
    source_path = glob.glob("/ws/lidar_data/3JEDKC50014U011/*.pcd")[0]
    base_path = "/ws/lidar_tuning/combine_data/combine_3JEDKC50014U011.pcd"
    inlier_cloud, target, inlier_cloud_down, target_down, inlier_cloud_fpfh, target_fpfh = prepare_dataset_remove_outlier(source_path, base_path, rough_voxel_size, 0)
    result_ransac = execute_global_registration(inlier_cloud_down, target_down,
                                    inlier_cloud_fpfh, target_fpfh,
                                    rough_voxel_size)
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source_path, base_path, precision_voxel_size)
    result_icp = execute_icp_registration(source_down, target_down, threshold, result_ransac.transformation)
    print(result_icp)
    source.transform(result_icp.transformation)
    pcd = source + target
    o3d.io.write_point_cloud('new_3JEDKC50014U011.pcd', pcd)
    print(result_icp.transformation.tolist())
    config['3JEDKC50014U011'] = result_icp.transformation.tolist()
    
    # 3
    source_path = glob.glob("/ws/lidar_data/3JEDL3N0015X621/*.pcd")[0]
    base_path = "/ws/lidar_tuning/combine_data/combine_3JEDL3N0015X621.pcd"
    inlier_cloud, target, inlier_cloud_down, target_down, inlier_cloud_fpfh, target_fpfh = prepare_dataset_remove_outlier(source_path, base_path, rough_voxel_size, 0)
    result_ransac = execute_global_registration(inlier_cloud_down, target_down,
                                    inlier_cloud_fpfh, target_fpfh,
                                    rough_voxel_size)
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source_path, base_path, precision_voxel_size)
    result_icp = execute_icp_registration(source_down, target_down, threshold, result_ransac.transformation)
    print(result_icp)
    source.transform(result_icp.transformation)
    pcd = source + target
    o3d.io.write_point_cloud('new_3JEDL3N0015X621.pcd', pcd)
    print(result_icp.transformation.tolist())
    config['3JEDL3N0015X621'] = result_icp.transformation.tolist()
    
    # 4
    source_path = glob.glob("/ws/lidar_data/3JEDL76001L4201/*.pcd")[0]
    base_path = "/ws/lidar_tuning/combine_data/combine_3JEDL76001L4201.pcd"
    inlier_cloud, target, inlier_cloud_down, target_down, inlier_cloud_fpfh, target_fpfh = prepare_dataset_remove_outlier(source_path, base_path, rough_voxel_size, 0)
    result_ransac = execute_global_registration(inlier_cloud_down, target_down,
                                    inlier_cloud_fpfh, target_fpfh,
                                    rough_voxel_size)
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source_path, base_path, precision_voxel_size)
    result_icp = execute_icp_registration(source_down, target_down, threshold, result_ransac.transformation)
    print(result_icp)
    source.transform(result_icp.transformation)
    pcd = source + target
    o3d.io.write_point_cloud('new_3JEDL76001L4201.pcd', pcd)
    print(result_icp.transformation.tolist())
    config['3JEDL76001L4201'] = result_icp.transformation.tolist()
    
    with open("/ws/lidar_data/matrix_config.json", mode="w") as f:
            config = json.dumps(config)
            f.write(config)



if __name__ == '__main__':
    main()