import os
import sys
import csv
import glob
import argparse
from datetime import datetime
from tqdm import tqdm
import json

import numpy as np
import open3d as o3d
import hdbscan
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from joblib import Parallel, delayed

import imo_pcd_reader

# 0   "barrier",
# 1   "bicycle",
# 2   "bus",
# 3   "car",
# 4   "construction_vehicle",
# 5   "motorcycle",
# 6   "pedestrian",
# 7   "traffic_cone",
# 8   "trailer",
# 9   "truck",
# 10  "driveable_surface",
# 11  "other_flat",
# 12  "sidewalk",
# 13  "terrain",
# 14  "manmade",
# 15  "vegetation",

parser = argparse.ArgumentParser(description='Point Cloud Alignment')

parser.add_argument('--slam_paras_project_dir', type=str, 
                    default='/your/path/to/.../map*')

parser.add_argument('--seg_result_dir', type=str, 
                    default='/your/path/to/.../segmented_pcd')

parser.add_argument('--dataset_name', type=str, 
                    default='Parking_PC_*')

parser.add_argument('--bucket', type=str, 
                    default='test-tm-*')

parser.add_argument("--stack_frames", action="store_true", default=False,
                    help="Stack frames if present (default: false)")

args = parser.parse_args()

if args.stack_frames:
    timestamps = []
    matrices = []
    maps = glob.glob(os.path.join(args.slam_paras_project_dir, "map*/"))
    for map_dir in maps:
        file_list = glob.glob(os.path.join(map_dir, "timestamp_pose.csv"))
        if len(file_list) == 1:
            filename = file_list[0]
            current_matrix_rows = []
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    for idx, num in enumerate(row):
                        if idx == 0:
                            timestamps.append(int(num))
                        else:
                            current_matrix_rows.append(float(num))
                    matrices.append(np.array(current_matrix_rows).reshape((4, 4)))
                    current_matrix_rows = []  # Reset for the next matrix

def filename_2_timestamp(filename):
    parts = filename.split('_')
    query_datetime = parts[0] + '_' + parts[1][:-3]  # Slice to remove the last 3 digits (milliseconds)
    query_dt = datetime.strptime(query_datetime, '%Y%m%d_%H%M%S') 
    seconds = query_dt.timestamp()
    milliseconds = int(parts[1][-3:])
    return int(seconds * 1e9) + milliseconds * 1e6

def find_nearest_pose(timestamps, timestamps_np, matrices, target_timestamp):
    
    closest_index = np.searchsorted(timestamps_np, target_timestamp)

    timestamp_th = 0.01 * 1e9
    # Handle potential out-of-bounds index
    if closest_index == 0:
        if abs(timestamps[0] - target_timestamp) < timestamp_th:
            return matrices[0]
        else:
            return None
    if closest_index == len(timestamps):
        if abs(timestamps[-1] - target_timestamp) < timestamp_th:
            return matrices[-1]
        else:
            return None

    before = timestamps[closest_index - 1]
    after = timestamps[closest_index]
    if abs(before - target_timestamp) < abs(after - target_timestamp):
        if abs(before - target_timestamp) < timestamp_th:
            return matrices[closest_index - 1]
        else:
            return None
    else:
        if abs(after - target_timestamp) < timestamp_th:
            return matrices[closest_index]
        else:
            return None

def objective_function(params, grid_points, sd_noise):
    return imo_pcd_reader.objective_function(params, grid_points, sd_noise)

def objective_function_py(yaw, grid_points, sd_noise):
    # Create rotation matrix
    rotation_2d = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])
    # Rotate the point cloud
    rotated_points = np.dot(grid_points, rotation_2d)
    # Find min and max coordinates of the rotated points
    min_coords = np.min(rotated_points, axis=0)
    max_coords = np.max(rotated_points, axis=0)
    # Calculate eigenvalues and center
    eigenvalues = max_coords - min_coords
    center = (min_coords + max_coords) / 2.0
    # Calculate P (transpose of center * transpose of rotation matrix)
    P = np.dot(center, rotation_2d.T)
    # Calculate theta2
    theta2 = yaw + np.pi / 2
    # Create eigenvectors matrix
    eigenvectors = np.array([
        [np.cos(yaw), np.cos(theta2)],
        [np.sin(yaw), np.sin(theta2)]
    ])

    likelihoods = Parallel(n_jobs=-1)(delayed(imo_pcd_reader.measurement_likelihood)(P, eigenvalues, eigenvectors, grid_points[i, :], sd_noise) 
                                                        for i in range(grid_points.shape[0]))

    return -np.sum(likelihoods)

def project_and_grid(points_3d, grid_resolution):
    # 1. Project onto XY Plane
    points_2d = points_3d[:, :2]  # Discard the Z coordinate

    # 2. Assign to Grid Cells
    grid_indices = np.floor(points_2d / grid_resolution)  # Find grid cell indices

    # 3. Remove Duplicates (efficiently using KDTree)
    tree = cKDTree(grid_indices)
    unique_indices = tree.query(grid_indices, k=1)[1]  # Get unique indices using nearest neighbor
    unique_grid_indices = grid_indices[unique_indices]

    # 4. Convert Back to Coordinates
    grid_points = unique_grid_indices * grid_resolution + grid_resolution / 2  # Center points

    return grid_points

def fit_l_shape_3d_py(point_cloud):
    grid_resolution = 0.1
    grid_points = project_and_grid(point_cloud, grid_resolution)

    points = grid_points
    # Center the data
    points -= np.mean(points, axis=0)
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(points)
    # Get the main axis (first principal component)
    main_axis = pca.components_[0]
    # Calculate the angle between the main axis and the x-axis
    angle_radians = np.arctan2(main_axis[1], main_axis[0])

    p_theta = np.arange(angle_radians-np.pi/4, angle_radians+np.pi/4, np.pi/180)

    results = Parallel(n_jobs=-1)(delayed(objective_function_py)(yaw, point_cloud[:, :2], 0.5) for yaw in p_theta)
    best_idx = np.argmin(results)

    theta1 = p_theta[best_idx]

    rotation_2d = np.array([
                    [np.cos(theta1), -np.sin(theta1)],
                    [np.sin(theta1), np.cos(theta1)]
                ])
    rotation_matrix = np.eye(3)
    rotation_matrix[:2, :2] = rotation_2d

    rotated_point_cloud = np.dot(point_cloud, rotation_matrix)
    min_coords = np.min(rotated_point_cloud, axis=0)
    max_coords = np.max(rotated_point_cloud, axis=0)
    extents = max_coords - min_coords
    center = (min_coords + max_coords) / 2
    center = np.dot(center, rotation_matrix.T)
    return o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extents)

def fit_l_shape_3d(point_cloud):
    grid_resolution = 0.1
    grid_points = project_and_grid(point_cloud, grid_resolution)

    points = grid_points
    # Center the data
    points -= np.mean(points, axis=0)
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(points)
    # Get the main axis (first principal component)
    main_axis = pca.components_[0]
    # Calculate the angle between the main axis and the x-axis
    angle_radians = np.arctan2(main_axis[1], main_axis[0])

    p_theta = np.arange(angle_radians-np.pi/4, angle_radians+np.pi/4, np.pi/180)

    results = imo_pcd_reader.fit_l_shape_3d(point_cloud[:, :2], p_theta, 0.5)
    best_idx = np.argmin(results)

    theta1 = p_theta[best_idx]

    rotation_2d = np.array([
                    [np.cos(theta1), -np.sin(theta1)],
                    [np.sin(theta1), np.cos(theta1)]
                ])
    rotation_matrix = np.eye(3)
    rotation_matrix[:2, :2] = rotation_2d

    rotated_point_cloud = np.dot(point_cloud, rotation_matrix)
    min_coords = np.min(rotated_point_cloud, axis=0)
    max_coords = np.max(rotated_point_cloud, axis=0)
    extents = max_coords - min_coords
    center = (min_coords + max_coords) / 2
    center = np.dot(center, rotation_matrix.T)
    return o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extents)

def compare_heading(point_cloud):
    grid_resolution = 0.1
    grid_points = project_and_grid(point_cloud, grid_resolution)

    points = grid_points
    # Center the data
    points -= np.mean(points, axis=0)
    # Perform PCA
    pca = PCA(n_components=2)
    pca.fit(points)
    # Get the main axis (first principal component)
    main_axis = pca.components_[0]
    # Calculate the angle between the main axis and the x-axis
    angle_radians = np.arctan2(main_axis[1], main_axis[0])

    pca_likely = objective_function(angle_radians, grid_points, 2*grid_resolution)
    axis_likely = objective_function(0, grid_points, 2*grid_resolution)
    if axis_likely < pca_likely:
        min_x, max_x = np.min(point_cloud[:, 0]), np.max(point_cloud[:, 0])
        min_y, max_y = np.min(point_cloud[:, 1]), np.max(point_cloud[:, 1])
        min_z, max_z = np.min(point_cloud[:, 2]), np.max(point_cloud[:, 2])
        center = np.array([(min_x+max_x)/2, (min_y+max_y)/2, (min_z+max_z)/2])
        rotation_matrix = np.eye(3)
        extents = np.array([max_x-min_x, max_y-min_y, max_z-min_z])
        return o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extents)
    else:
        theta1 = angle_radians
        rotation_2d = np.array([
                        [np.cos(theta1), -np.sin(theta1)],
                        [np.sin(theta1), np.cos(theta1)]
                    ])
        rotation_matrix = np.eye(3)
        rotation_matrix[:2, :2] = rotation_2d

        rotated_point_cloud = np.dot(point_cloud, rotation_matrix)
        min_coords = np.min(rotated_point_cloud, axis=0)
        max_coords = np.max(rotated_point_cloud, axis=0)
        extents = max_coords - min_coords
        center = (min_coords + max_coords) / 2
        center = np.dot(center, rotation_matrix.T)
        return o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extents)

pcd_names = [f for f in os.listdir(args.seg_result_dir) if f.endswith(".pcd")]
pcd_names.sort()
data_list = [os.path.join(args.seg_result_dir, name) for name in pcd_names]
ts_delta_th = 5e9
frame_idx_delta_th = 1
excluded_area = np.array([[-1.04, 3.863], [-1, 1]])
ceiling_height = 3
label_area_range = 35
selected_label = [2, 3, 4, 8, 9]
parts = args.dataset_name.split('_')
city = parts[-2]
carId = parts[-3]
projectName = parts[-4]

def proc_data(idx, data_pth):
    selected_pcd = imo_pcd_reader.read_AL_pcd_selected_label(data_pth, excluded_area, ceiling_height, label_area_range, selected_label)
    curr_points = selected_pcd[:, :3]
    curr_ts = filename_2_timestamp(os.path.basename(data_pth))

    bboxes_list = []
    if curr_points.size > 100:
        # 2. Define the grid parameters
        cell_size = 0.02  # Size of each grid cell (adjust as needed)
        # Create x-y grid
        x_min, y_min, _ = np.min(curr_points, axis=0)
        x_max, y_max, _ = np.max(curr_points, axis=0)
        x_grid = np.arange(x_min, x_max, cell_size)
        y_grid = np.arange(y_min, y_max, cell_size)

        # Assign each point to a grid cell
        grid_indices = np.floor((curr_points[:, :2] - [x_min, y_min]) / cell_size).astype(int)

        # Create unique grid cell indices and count points in each cell
        unique_grid_indices, point_counts = np.unique(grid_indices, axis=0, return_counts=True)

        # Use DBSCAN to cluster grid cells based on their point counts
        clustering = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=20).fit(unique_grid_indices)  # Adjust parameters
        labels = clustering.labels_

        # Use a k-d tree for efficient nearest neighbor search
        tree = cKDTree(unique_grid_indices)
        _, closest_grid_cell_indices = tree.query(grid_indices, k=1)

        # Assign cluster labels to original points based on their grid cells
        point_labels = labels[closest_grid_cell_indices]

        max_label = max(point_labels)
        bboxes = []
        for i in range(max_label+1):
            cluster_points = curr_points[point_labels == i]
            bbox = fit_l_shape_3d(cluster_points)
            # bbox = fit_l_shape_3d_py(cluster_points)
            # bbox = compare_heading(cluster_points)
            center = bbox.center
            rotation_matrix = np.copy(bbox.R)
            extent = bbox.extent
            r = R.from_matrix(rotation_matrix)
            yaw, pitch, roll = r.as_euler('zyx', degrees=False)
            box_data = {
                "id": i,
                "lidarPoints": cluster_points.shape[0],
                "rotation": [roll, pitch, yaw],
                "size": extent.tolist(),
                "translation": center.tolist()
            }
            bboxes_list.append(box_data)
            bbox.color = [0, 0, 1]
            bboxes.append(bbox)

        # curr_pcd = imo_pcd_reader.read_AL_pcd_with_excluded_area(data_pth, excluded_area, ceiling_height)
        # curr_coord = curr_pcd[:, :3]
        # source = o3d.geometry.PointCloud()
        # source.points = o3d.utility.Vector3dVector(curr_coord)
        # o3d.visualization.draw_geometries([source, *bboxes])

    base_name, extension = os.path.splitext(os.path.basename(data_pth))
    parts = base_name.split('_')
    scene = parts[-1]
    pointClound = '_'.join(parts[:-2]) + ".pcd"
        
    frame_data = {
        "bucket": args.bucket,
        "carId": carId,
        "city": city,
        "directoryName": args.dataset_name,
        "instances": bboxes_list,
        "lidarTime": curr_ts,
        "pointClound": pointClound,
        "projectName": projectName,
        "scene": scene
    }
    return frame_data

out_data_list = Parallel(n_jobs=-1)(delayed(proc_data)(idx, data_pth) for idx, data_pth in tqdm(enumerate(data_list), total=len(data_list), mininterval=1.0))

parent_dir = os.path.abspath(os.path.join(args.seg_result_dir, os.pardir))
out_pth = os.path.join(parent_dir, "ptv3_bbox_result.txt")
with open(out_pth, "w") as file:
    for data in out_data_list:
        json.dump(data, file)
        file.write("\n")  # Add newline after each dictionary
