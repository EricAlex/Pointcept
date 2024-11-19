import os
import sys
import csv
import glob
import struct
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
from joblib import Parallel, delayed, parallel_backend
import multiprocessing
from concave_hull import concave_hull
from shapely.geometry import Polygon, MultiPolygon
import statistics
import math
import time
import pyvista as pv
# import imageio.v2 as imageio
import subprocess
import copy
import uuid

import imo_pcd_reader

# 0: road
# 1: curb
# 2: car
# 3: bicycle
# 4: traffic_pole
# 5: traffic_board
# 6: vegetation
# 7: construction_sign
# 8: others
# 9: pedestrain
# 10: bus
# 11: truck
# 12: special_vehicle
# 13: tricycle
# 14: traffic_cone
# 15: crash_barrel
# 16: noise
# 17: sidewalks
# 18: fence
# 19: wall
# 20: bump
# 21: architecture
# 22: pillars
# 23: ground_obstacle
# 24: floating_obstacle
# 25: animal
# 26: undefined

bbClass_label_map = {
    "vehicle.car" : [2],
    "static_object.pillar" : [22]
}

bbClass_config_map = {
    "vehicle.car" : "car.ini",
    "static_object.pillar" : "pillars.ini"
}

parser = argparse.ArgumentParser(description='Point Cloud Alignment')

parser.add_argument('--slam_paras_project_dir', type=str, 
                    default='/your/path/to/.../map*')

parser.add_argument('--seg_result_dir', type=str, 
                    default='/your/path/to/.../segmented_pcd')

parser.add_argument('--dataset_name', type=str, 
                    default='Parking_PC_*')

parser.add_argument("--stack_frames", action="store_true", default=False,
                    help="Stack frames if present (default: false)")

args = parser.parse_args()

def filename_2_timestamp(filename):
    parts = filename.split('_')
    query_datetime = parts[0] + '_' + parts[1][:-3]  # Slice to remove the last 3 digits (milliseconds)
    query_dt = datetime.strptime(query_datetime, '%Y%m%d_%H%M%S') 
    seconds = query_dt.timestamp()
    milliseconds = int(parts[1][-3:])
    return int(seconds * 1e9) + int(milliseconds * 1e6)

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

def write_data(pcd_path, measurements, promissing_new_t_idx, p_n_t_eigenvectors, p_n_t_extents, pose4Predict, grid_parameters, scan_time, frame_idx):
    """Writes measurement data and grid parameters to a binary file.

    Args:
        pcd_path (str or os.PathLike): Path to the output file.
        measurements (list): List of measurement objects.
        grid_parameters (dict): Dictionary containing grid parameters.
        scan_time (float): delta time to last frame (seconds).
        frame_idx (int): Frame index.
    """

    if os.path.exists(pcd_path):
        os.remove(pcd_path)

    with open(pcd_path, 'wb') as fbinaryout:
        # Write the number of measurements
        num_measurements = len(measurements)
        fbinaryout.write(struct.pack('Q', num_measurements))  # Unsigned long long (8 bytes)

        # Write each measurement
        for m in measurements:
            fbinaryout.write(struct.pack('d', m[0]))  # Float (4 bytes)
            fbinaryout.write(struct.pack('d', m[1]))  # Float (4 bytes)

        num_new_t = len(promissing_new_t_idx)
        fbinaryout.write(struct.pack('Q', num_new_t))  # Unsigned long long (8 bytes)

        for idx in promissing_new_t_idx:
            fbinaryout.write(struct.pack('Q', idx))
        
        for eigenvectors in p_n_t_eigenvectors:
            fbinaryout.write(struct.pack('d', eigenvectors[0, 0]))
            fbinaryout.write(struct.pack('d', eigenvectors[1, 0]))
            fbinaryout.write(struct.pack('d', eigenvectors[0, 1]))
            fbinaryout.write(struct.pack('d', eigenvectors[1, 1]))
        
        for extents in p_n_t_extents:
            fbinaryout.write(struct.pack('d', extents[0]))
            fbinaryout.write(struct.pack('d', extents[1]))
        
        for row in pose4Predict:
            for element in row:
                fbinaryout.write(struct.pack('d', element))

        # Write grid parameters
        fbinaryout.write(struct.pack('d', grid_parameters["dim1_min"]))
        fbinaryout.write(struct.pack('d', grid_parameters["dim1_max"]))
        fbinaryout.write(struct.pack('d', grid_parameters["dim2_min"]))
        fbinaryout.write(struct.pack('d', grid_parameters["dim2_max"]))
        fbinaryout.write(struct.pack('d', grid_parameters["grid_res"]))

        # Write scan time and frame index
        fbinaryout.write(struct.pack('d', scan_time))    # Double (8 bytes)
        fbinaryout.write(struct.pack('Q', frame_idx))   # Unsigned long long (8 bytes)

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

def align_axis_bb(point_cloud):
    theta1 = 0
    theta2 = theta1 + np.pi / 2
    # Create eigenvectors matrix
    eigenvectors = np.array([
        [np.cos(theta1), np.cos(theta2)],
        [np.sin(theta1), np.sin(theta2)]
    ])
    rotated_point_cloud = point_cloud[:, :2]
    min_coords = np.min(rotated_point_cloud, axis=0)
    max_coords = np.max(rotated_point_cloud, axis=0)
    extents = max_coords - min_coords
    center = (min_coords + max_coords) / 2

    return center, eigenvectors, extents

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

    theta2 = theta1 + np.pi / 2
    # Create eigenvectors matrix
    eigenvectors = np.array([
        [np.cos(theta1), np.cos(theta2)],
        [np.sin(theta1), np.sin(theta2)]
    ])

    rotation_2d = np.array([
                    [np.cos(theta1), -np.sin(theta1)],
                    [np.sin(theta1), np.cos(theta1)]
                ])

    rotated_point_cloud = np.dot(point_cloud[:, :2], rotation_2d)
    min_coords = np.min(rotated_point_cloud, axis=0)
    max_coords = np.max(rotated_point_cloud, axis=0)
    extents = max_coords - min_coords
    center = (min_coords + max_coords) / 2
    center = np.dot(center, rotation_2d.T)
    return center, eigenvectors, extents

ts_delta_th = 2e9
frame_idx_delta_th = 1
ts_one_second = 1e9
excluded_area = np.array([[-1.12, 3.863], [-1.1, 1.1]])
ceiling_height = 4
label_area_range = 35
grid_resolution = 0.1

def proc4track(data_list, idx, data_pth, save_dir, selected_label, bbClass):
    curr_pcd = imo_pcd_reader.read_AL_pcd_selected_label(data_pth, excluded_area, ceiling_height, label_area_range, selected_label)
    curr_points = curr_pcd[:, :3]
    curr_ts = filename_2_timestamp(os.path.basename(data_pth))
    curr_pose = find_nearest_pose(timestamps, timestamps_np, matrices, curr_ts)
    if args.stack_frames and curr_pose is not None:
        curr_pose_inv = np.linalg.inv(curr_pose)
        idx_list = []
        pose_list = []
        wind_s = max(idx-frame_idx_delta_th, 0)
        wind_e = min(idx+frame_idx_delta_th+1, len(data_list))
        for i in range(wind_s, wind_e):
            if i != idx:
                frame_ts = filename_2_timestamp(os.path.basename(data_list[i]))
                frame_pose = find_nearest_pose(timestamps, timestamps_np, matrices, frame_ts)
                if frame_pose is not None and abs(frame_ts-curr_ts) < ts_delta_th:
                    idx_list.append(i)
                    pose_list.append(frame_pose)
        
        for f, f_idx in enumerate(idx_list):
            trans_pose = curr_pose_inv.dot(pose_list[f])
            frame_pcd = imo_pcd_reader.read_AL_pcd_selected_label(data_list[f_idx], excluded_area, ceiling_height, label_area_range, selected_label)
            frame_points = frame_pcd[:, :3]
            new_column = np.ones((frame_points.shape[0], 1))
            aug_coord = np.hstack((frame_points, new_column))
            trans_coord = trans_pose.dot(aug_coord.T)
            out_coord = trans_coord.T[:, :3]
            curr_points = np.vstack((curr_points, out_coord))

    total_vertices_list = []
    promissing_new_t_idx = []
    p_n_t_eigenvectors = []
    p_n_t_extents = []
    x_min, y_min, x_max, y_max = [0, 0, 0, 0]
    total_num_vertices = 0
    if curr_points.size != 0:
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

        if bbClass == "vehicle.car":
            u_g_i_th = 100
            min_cluster_size = 20
            min_samples = 15
        elif bbClass == "static_object.pillar":
            u_g_i_th = 10
            min_cluster_size = 10
            min_samples = 4

        if unique_grid_indices.shape[0] > u_g_i_th:
            # Use DBSCAN to cluster grid cells based on their point counts
            clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit(unique_grid_indices)  # Adjust parameters
            # clustering = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=20).fit(unique_grid_indices)  # Adjust parameters
            labels = clustering.labels_

            # Use a k-d tree for efficient nearest neighbor search
            tree = cKDTree(unique_grid_indices)
            _, closest_grid_cell_indices = tree.query(grid_indices, k=1)

            # Assign cluster labels to original points based on their grid cells
            point_labels = labels[closest_grid_cell_indices]

            tmp_vertices_list = []
            tmp_eigenvectors_list = []
            tmp_extents_list = []
            tmp_select_num_list = []
            max_label = max(point_labels)
            for i in range(max_label+1):
                cluster_points = curr_points[point_labels == i]
                
                center, eigenvectors, extents = fit_l_shape_3d(cluster_points)
                # center, eigenvectors, extents = align_axis_bb(cluster_points)

                # concave hull
                # grid_points = project_and_grid(cluster_points, grid_resolution)
                # vertices = concave_hull(grid_points, length_threshold=5)
                # vertices = np.vstack((vertices, center))
                # tmp_vertices_list.append(vertices)
                # tmp_eigenvectors_list.append(eigenvectors)
                # tmp_extents_list.append(extents)
                # tmp_select_num_list.append(vertices.shape[0]+1)
                # random sampling
                if bbClass == "vehicle.car":
                    areaSize = extents[0]*extents[1]
                elif bbClass == "static_object.pillar":
                    areaSize = 1
                select_num = int(0.1*areaSize/(grid_resolution*grid_resolution))
                grid_points = project_and_grid(cluster_points, grid_resolution)
                if select_num > grid_points.shape[0]:
                    select_num = grid_points.shape[0]
                if select_num > 3:
                    selected_pt = np.random.choice(grid_points.shape[0], select_num, replace=False)
                    vertices = grid_points[selected_pt]
                    vertices = np.vstack((vertices, center))
                    tmp_vertices_list.append(vertices)
                    tmp_eigenvectors_list.append(eigenvectors)
                    tmp_extents_list.append(extents)
                    tmp_select_num_list.append(select_num+1)
            
            tmp_select_num = np.array(tmp_select_num_list)
            sorted_indices = np.argsort(tmp_select_num)
            vertices_to_stack = []
            for i in sorted_indices:
                vertices_to_stack.append(tmp_vertices_list[i])
                total_num_vertices += tmp_vertices_list[i].shape[0]
                promissing_new_t_idx.append(total_num_vertices-1)
                p_n_t_eigenvectors.append(tmp_eigenvectors_list[i])
                p_n_t_extents.append(tmp_extents_list[i])

            if len(vertices_to_stack)>0:
                total_vertices = np.vstack(vertices_to_stack)
                total_vertices_list = total_vertices.tolist()
                x_min, y_min = np.min(total_vertices, axis=0)
                x_max, y_max = np.max(total_vertices, axis=0)

    if idx == 0:
        delta_time = 0
    else:
        prev_ts = filename_2_timestamp(os.path.basename(data_list[idx-1]))
        delta_time = float(curr_ts - prev_ts)/ts_one_second
    
    print(f"total number of data: {total_num_vertices}, delta time: {delta_time} s.")

    pose4Predict = np.eye(4)
    if (curr_pose is not None) and (idx > 0):
        curr_pose_inv = np.linalg.inv(curr_pose)
        prev_ts = filename_2_timestamp(os.path.basename(data_list[idx-1]))
        delta_time = float(curr_ts - prev_ts)/ts_one_second
        prev_pose = find_nearest_pose(timestamps, timestamps_np, matrices, prev_ts)
        if abs(delta_time) < 10 and prev_pose is not None:
            pose4Predict = curr_pose_inv.dot(prev_pose)
    
    grid_parameters = {
        "dim1_min": x_min,
        "dim1_max": x_max,
        "dim2_min": y_min,
        "dim2_max": y_max,
        "grid_res": grid_resolution
    }
    base_name, extension = os.path.splitext(os.path.basename(data_pth))
    out_f_pth = os.path.join(save_dir, base_name+".bin")
    write_data(out_f_pth, total_vertices_list, promissing_new_t_idx, p_n_t_eigenvectors, p_n_t_extents, pose4Predict, grid_parameters, delta_time, curr_ts)

def run_and_stream_output(command, args):
    process = subprocess.Popen([command] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)
    for line in process.stdout:
        print(line, end='')

def read_pos_from_binary(file_path):
    potential_objects = []
    with open(file_path, 'rb') as f:
        # Read the header (number of objects)
        num_objects = struct.unpack('<Q', f.read(8))[0]  # '<Q' for little-endian uint64_t
        for _ in range(num_objects):
            # Read kinematic data
            p1, p2, v1, v2, t, s = struct.unpack('<dddddd', f.read(48))  # 6 doubles
            # Read extent data (assuming Eigen::Matrix2d is stored as 4 doubles)
            e_data = struct.unpack('<dddd', f.read(32))
            eigenvalues_data = struct.unpack('<dd', f.read(16))  # 2 doubles
            eigenvectors_data = struct.unpack('<dddd', f.read(32))  # 4 doubles
            # Construct Eigen matrices from raw data
            e = np.array(e_data).reshape(2, 2)
            eigenvalues = np.array(eigenvalues_data)
            eigenvectors = np.array(eigenvectors_data).reshape(2, 2)
            # Read label data
            timestamp, label_v = struct.unpack('<QQ', f.read(16))  # 2 uint64_t
            # Create PO object
            po = {
                'kinematic': {'p1': p1, 'p2': p2, 'v1': v1, 'v2': v2, 't': t, 's': s},
                'extent': {'e': e, 'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors},
                'label': {'timestamp': timestamp, 'label_v': label_v}
            }
            potential_objects.append(po)

    return potential_objects

def add_pos_into_trajectories(trajectories, potential_objects):
    for po in potential_objects:
        label_v = po['label']['label_v']
        if label_v not in trajectories:
            trajectories[label_v] = []  # Create new trajectory if label_v not seen before
        trajectories[label_v].append(po)

def is_point_in_polygon(points, center, eigen_values, eigen_vectors):
    # Translate points to the polygon's coordinate system
    points_translated = points - np.array(center)  # Broadcasting for efficient subtraction
    # Project the translated points onto the eigenvector axes
    projected_coords = np.dot(points_translated, eigen_vectors)  # Note the change in dot product order
    # Check if the projected coordinates are within the half-lengths of the eigenvalues
    half_lengths = eigen_values / 2
    inside_mask = np.all(np.abs(projected_coords) <= half_lengths, axis=1)
    return inside_mask

def po_iou(box1, box2):
    # Extract relevant data from the boxes
    center1 = np.array([box1['kinematic']['p1'], box1['kinematic']['p2']])
    center2 = np.array([box2['kinematic']['p1'], box2['kinematic']['p2']])
    lengths1 = box1['extent']['eigenvalues']
    lengths2 = box2['extent']['eigenvalues']
    axes1 = box1['extent']['eigenvectors']
    axes2 = box2['extent']['eigenvectors']

    # Construct the four corners of each box
    corners1 = np.vstack([
        center1 - 0.5 * lengths1[0] * axes1[:, 0],
        center1 - 0.5 * lengths1[1] * axes1[:, 1],
        center1 + 0.5 * lengths1[0] * axes1[:, 0],
        center1 + 0.5 * lengths1[1] * axes1[:, 1]
    ])
    corners2 = np.vstack([
        center2 - 0.5 * lengths2[0] * axes2[:, 0],
        center2 - 0.5 * lengths2[1] * axes2[:, 1],
        center2 + 0.5 * lengths2[0] * axes2[:, 0],
        center2 + 0.5 * lengths2[1] * axes2[:, 1]
    ])

    # Create Shapely Polygons for IoU calculation
    poly1 = Polygon(corners1)
    poly2 = Polygon(corners2)
    intersection = poly1.intersection(poly2).area
    # union = poly1.union(poly2).area
    # iou = intersection / union if union > 0 else 0
    area1 = poly1.area
    area2 = poly2.area
    iou = intersection / min(area1, area2)
    
    return iou

parent_dir = os.path.abspath(os.path.join(args.seg_result_dir, os.pardir))
maps_dirs = glob.glob(os.path.join(args.slam_paras_project_dir, "map*"))
maps_names = [os.path.basename(map_dir) for map_dir in maps_dirs if os.path.isdir(map_dir)]
maps_names = sorted(maps_names, key=lambda x: int(x.replace("map", "")))
parts = args.dataset_name.split('_')
city = parts[-2]
carId = parts[-3]
projectName = parts[-4]
out_data_list = []
for map_n in maps_names:
    timestamps = []
    matrices = []
    pc_path_file = glob.glob(os.path.join(args.slam_paras_project_dir, map_n, "*pcs_data_path.csv"))
    pose_file = glob.glob(os.path.join(args.slam_paras_project_dir, map_n, "pose*.csv"))
    if (len(pc_path_file) == 1) and (len(pose_file) == 1):
        pose_file = pose_file[0]
        pc_path_file = pc_path_file[0]
        current_matrix_rows = []
        with open(pose_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                for num in row:
                    current_matrix_rows.append(float(num))
                matrices.append(np.array(current_matrix_rows).reshape((4, 4)))
                current_matrix_rows = []  # Reset for the next matrix
        with open(pc_path_file, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for idx, row in enumerate(reader):
                if idx > 0:
                    timestamps.append(filename_2_timestamp(os.path.basename(row[0])))

    timestamps_np = np.array(timestamps)

    if len(timestamps) > 1:
        ts_b = timestamps[0]
        ts_e = timestamps[-1]
        pcd_names = [f for f in os.listdir(args.seg_result_dir) if (f.endswith(".pcd") and filename_2_timestamp(f)>=ts_b and filename_2_timestamp(f)<=ts_e)]
        pcd_names.sort()
        data_list = [os.path.join(args.seg_result_dir, name) for name in pcd_names]
        bboxes_list_by_frame = []
        for data_pth in data_list:
            bboxes_list = []
            bboxes_list_by_frame.append(bboxes_list)
        for bbClass, selected_label in bbClass_label_map.items():
            save_dir = os.path.join(parent_dir, "M4Track", "M_"+map_n, "M_"+bbClass)
            if not os.path.exists(save_dir): os.makedirs(save_dir)

            # Prepare data for tracking
            
            with parallel_backend('loky'):
                Parallel(n_jobs=max(1, int(multiprocessing.cpu_count()/2)))(delayed(proc4track)(data_list, idx, data_pth, save_dir, selected_label, bbClass) 
                            for idx, data_pth in tqdm(enumerate(data_list), total=len(data_list), mininterval=1.0))

            # Perform Tracking

            command = "./deploy/EOT/build/eot_cpp"
            track_args = ["-i", save_dir, "-c", "deploy/EOT/configs/"+bbClass_config_map[bbClass]]

            run_and_stream_output(command, track_args)

            # Trajectory management

            config_base_name, config_extension = os.path.splitext(bbClass_config_map[bbClass])
            trackedFolder = "trackedPOs_"+config_base_name
            saved_parent_dir = os.path.abspath(os.path.join(save_dir, os.pardir))
            tracked_dir = os.path.join(saved_parent_dir, trackedFolder)
            tracked_names = [f for f in os.listdir(tracked_dir) if f.endswith(".bin")]
            tracked_names.sort()
            tracked_list = [os.path.join(tracked_dir, name) for name in tracked_names]
            pos_by_frames = []
            trajectories = {}
            for tracked_ops in tracked_list:
                tmp_frame_pos = read_pos_from_binary(tracked_ops)
                pos_by_frames.append(tmp_frame_pos)
                add_pos_into_trajectories(trajectories, tmp_frame_pos)
            
            trackID2UUID = {}
            for label_v, po_list in trajectories.items():
                trackID2UUID[label_v] = uuid.uuid4()

            # Merge breaked trajectories

            staticSpeedTh = 1 # m/s
            trajectoryLengthTh = 4
            joinTrajectoriesIOUth = 0.5
            toMergePairList = []
            for label_v, po_list in trajectories.items():
                speed_list = [math.sqrt(po['kinematic']['v1']**2 + po['kinematic']['v2']**2) for po in po_list]
                speed_mean = statistics.mean(speed_list)
                if len(po_list) >= trajectoryLengthTh and speed_mean < staticSpeedTh:
                    for nxt_label_v, nxt_po_list in trajectories.items():
                        if len(nxt_po_list) >= trajectoryLengthTh and nxt_po_list[0]['label']['timestamp'] > po_list[-1]['label']['timestamp']:
                            nxt_speed_list = [math.sqrt(po['kinematic']['v1']**2 + po['kinematic']['v2']**2) for po in nxt_po_list]
                            nxt_speed_mean = statistics.mean(nxt_speed_list)
                            if nxt_speed_mean < staticSpeedTh:
                                curr_idx = int(len(po_list)/2)
                                nxt_idx = int(len(nxt_po_list)/2)
                                # curr_idx = -1
                                # nxt_idx = 0
                                tmp_pose = find_nearest_pose(timestamps, timestamps_np, matrices, po_list[curr_idx]['label']['timestamp'])
                                nxt_tmp_pose = find_nearest_pose(timestamps, timestamps_np, matrices, nxt_po_list[nxt_idx]['label']['timestamp'])
                                if tmp_pose is not None and nxt_tmp_pose is not None:
                                    nxt_tmp_po = copy.deepcopy(nxt_po_list[nxt_idx])
                                    nxt2curr = np.linalg.inv(tmp_pose) @ nxt_tmp_pose
                                    nxt_p = np.array([nxt_tmp_po['kinematic']['p1'], nxt_tmp_po['kinematic']['p2'], 0, 1])
                                    trans_p = nxt2curr @ nxt_p
                                    nxt_tmp_po['kinematic']['p1'] = trans_p[0]
                                    nxt_tmp_po['kinematic']['p2'] = trans_p[1]
                                    nxt_tmp_po['extent']['e'] = nxt2curr[:2, :2] @ nxt_po_list[nxt_idx]['extent']['e'] @ nxt2curr[:2, :2].T
                                    nxt_tmp_po['extent']['eigenvalues'], nxt_tmp_po['extent']['eigenvectors'] = np.linalg.eig(nxt_tmp_po['extent']['e'])
                                    if po_iou(nxt_tmp_po, po_list[curr_idx]) > joinTrajectoriesIOUth:
                                        toMergePairList.append([po_list[curr_idx]['label']['label_v'], nxt_po_list[nxt_idx]['label']['label_v']])
                                        break

            for j_p in reversed(toMergePairList):
                if j_p[0] in trajectories and j_p[1] in trajectories:
                    if len(trajectories[j_p[0]]) >= len(trajectories[j_p[1]]):
                        for po in trajectories[j_p[1]]:
                            po['label']['label_v'] = j_p[0]
                        trajectories[j_p[0]].extend(trajectories[j_p[1]])
                        del trajectories[j_p[1]]
                    else:
                        for po in trajectories[j_p[0]]:
                            po['label']['label_v'] = j_p[1]
                        trajectories[j_p[1]][:0] = trajectories[j_p[0]]
                        del trajectories[j_p[0]]

            print(f"Jointed trajectories: {len(toMergePairList)}")

            # Best box assign

            for label_v, po_list in trajectories.items():
                speed_list = [math.sqrt(po['kinematic']['v1']**2 + po['kinematic']['v2']**2) for po in po_list]
                speed_mean = statistics.mean(speed_list)
                if speed_mean < staticSpeedTh:
                    select_list = []
                    for po in po_list:
                        tmp_speed = math.sqrt(po['kinematic']['v1']**2 + po['kinematic']['v2']**2)
                        tmp_rot_rate = po['kinematic']['t']
                        tmp_bb_size = po['extent']['eigenvalues'][0]*po['extent']['eigenvalues'][1]
                        tmp_pose = find_nearest_pose(timestamps, timestamps_np, matrices, po['label']['timestamp'])
                        if tmp_speed < 0.05 and tmp_pose is not None:
                        # if abs(tmp_rot_rate) > np.pi/45 and tmp_pose is not None:
                        # if tmp_bb_size > 6 and tmp_pose is not None:
                            select_list.append(po)
                    
                    if(len(select_list)>=1):
                        po_ori = copy.deepcopy(select_list[0])
                        # largest size
                        # for sl_po in select_list:
                        #     if(sl_po['extent']['eigenvalues'][0]*sl_po['extent']['eigenvalues'][1] >= po_ori['extent']['eigenvalues'][0]*po_ori['extent']['eigenvalues'][1]):
                        #         po_ori = sl_po
                        # selected_pose = find_nearest_pose(timestamps, timestamps_np, matrices, po_ori['label']['timestamp'])
                        # mean position and extent
                        po_ori['kinematic']['p1'] = 0
                        po_ori['kinematic']['p2'] = 0
                        po_ori['extent']['e'] = np.zeros((2, 2))
                        selected_valid_c = 0
                        for sl_po in select_list:
                            pose = find_nearest_pose(timestamps, timestamps_np, matrices, sl_po['label']['timestamp'])
                            if pose is not None:
                                tmp_pos = np.array([sl_po['kinematic']['p1'], sl_po['kinematic']['p2'], 0, 1])
                                trans_pos = pose @ tmp_pos
                                po_ori['kinematic']['p1'] += trans_pos[0]
                                po_ori['kinematic']['p2'] += trans_pos[1]
                                po_ori['extent']['e'] += pose[:2, :2] @ sl_po['extent']['e'] @ pose[:2, :2].T
                                selected_valid_c += 1
                        if selected_valid_c > 0:
                            po_ori['kinematic']['p1'] /= selected_valid_c
                            po_ori['kinematic']['p2'] /= selected_valid_c
                            po_ori['extent']['e'] /= selected_valid_c
                            selected_pose = np.eye(4)
                        else:
                            selected_pose = None

                        ori_pos = np.array([po_ori['kinematic']['p1'], po_ori['kinematic']['p2'], 0, 1])
                        for po in po_list:
                            curr_pose = find_nearest_pose(timestamps, timestamps_np, matrices, po['label']['timestamp'])
                            if selected_pose is not None and curr_pose is not None:
                                curr_pose_inv = np.linalg.inv(curr_pose)
                                selected2curr = curr_pose_inv.dot(selected_pose)
                                po['extent']['e'] = selected2curr[:2, :2] @ po_ori['extent']['e'] @ selected2curr[:2, :2].T
                                trans_pos = selected2curr.dot(ori_pos)
                                cent_dist = math.sqrt((po['kinematic']['p1']-trans_pos[0])**2 + (po['kinematic']['p2']-trans_pos[1])**2)
                                if cent_dist < 2:
                                    po['kinematic']['p1'] = trans_pos[0]
                                    po['kinematic']['p2'] = trans_pos[1]

            if bbClass == "static_object.pillar":
                duplicateBoxIOUTh = 0.05
            else:
                duplicateBoxIOUTh = 0.1
            for for_idx, (data_pth, pos_in_frame) in tqdm(enumerate(zip(data_list, pos_by_frames)), total=len(data_list), mininterval=1.0):
                label_pcd = imo_pcd_reader.read_AL_pcd_selected_label(data_pth, excluded_area, float('inf'), label_area_range, selected_label)
                label_points = label_pcd[:, :3]
                for i, po in enumerate(pos_in_frame):
                    eigenvalues, eigenvectors = np.linalg.eig(po['extent']['e'])
                    center_2d = np.array([po['kinematic']['p1'], po['kinematic']['p2']])
                    is_unique_addable = True
                    for j, po_j in enumerate(pos_in_frame):
                        if j != i and po_iou(po, po_j) > duplicateBoxIOUTh and len(trajectories[po['label']['label_v']]) < len(trajectories[po_j['label']['label_v']]):
                            is_unique_addable = False
                            break
                    if is_unique_addable:
                        points_in_box = label_points[is_point_in_polygon(label_points[:, :2], center_2d, eigenvalues, eigenvectors)]
                        if points_in_box.shape[0] > 0:
                            min_z, max_z = np.min(points_in_box[:, 2]), np.max(points_in_box[:, 2])
                            center = np.array([po['kinematic']['p1'], po['kinematic']['p2'], (min_z+max_z)/2])
                            rotation_2d = np.array([
                                            [eigenvectors[0,0], -eigenvectors[1,0]],
                                            [eigenvectors[1,0], eigenvectors[0,0]]
                                        ])
                            rotation_matrix = np.eye(3)
                            rotation_matrix[:2, :2] = rotation_2d
                            extents = np.array([eigenvalues[0], eigenvalues[1], (max_z-min_z)])
                            bb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extents)
                            tmp_rotation_matrix = np.copy(bb.R)
                            r = R.from_matrix(tmp_rotation_matrix)
                            yaw, pitch, roll = r.as_euler('zyx', degrees=False)
                            tmp_speed = math.sqrt(po['kinematic']['v1']**2 + po['kinematic']['v2']**2)
                            tmp_rot_rate = po['kinematic']['t']
                            box_data = {
                                "bbClass": bbClass,
                                "id": i,
                                "liarPoints": points_in_box.shape[0],
                                "rotation": [roll, pitch, yaw],
                                "size": bb.extent.tolist(),
                                "translation": bb.center.tolist(),
                                "trackId": str(trackID2UUID[po['label']['label_v']]),
                                "speed": tmp_speed,
                                "rotationSpeed": tmp_rot_rate
                            }
                            bboxes_list_by_frame[for_idx].append(box_data)
        
        # Save to file

        for for_idx, data_pth in enumerate(data_list):
            base_name, extension = os.path.splitext(os.path.basename(data_pth))
            parts = base_name.split('_')
            scene = parts[-1]
            pointCloud = '_'.join(parts[:-2]) + ".pcd"
            curr_ts = filename_2_timestamp(os.path.basename(data_pth))
            egoPose = find_nearest_pose(timestamps, timestamps_np, matrices, curr_ts)
            if egoPose is None:
                egoPose = []
            else:
                egoPose = egoPose.flatten().tolist()
                
            frame_data = {
                "directoryName": args.dataset_name,
                "instances": bboxes_list_by_frame[for_idx],
                "lidarTime": curr_ts,
                "pointClound": pointCloud,
                "projectName": projectName,
                "scene": scene,
                "egoPose": egoPose
            }
            out_data_list.append(frame_data)

        # To 3d boxes per frame (visualization)

        # def create_obb(bbox):
        #     center = bbox.center
        #     extents = bbox.extent  # half-extents
        #     rotation_matrix = np.copy(bbox.R)

        #     obb = pv.Cube(center=(0, 0, 0), x_length=extents[0], y_length=extents[1], z_length=extents[2])
        #     obb.points = (rotation_matrix @ obb.points.T).T
        #     obb.points = obb.points + center

        #     return obb

        # streamFolder = "trackedStream"
        # stream_dir = os.path.join(saved_parent_dir, streamFolder)
        # if not os.path.exists(stream_dir): os.makedirs(stream_dir)

        # for data_pth, pos_in_frame in zip(data_list, pos_by_frames):
        #     label_pcd = imo_pcd_reader.read_AL_pcd_selected_label(data_pth, excluded_area, float('inf'), label_area_range, selected_label)
        #     label_points = label_pcd[:, :3]
        #     bboxes = []
        #     for po in pos_in_frame:
        #         eigenvalues, eigenvectors = np.linalg.eig(po['extent']['e'])
        #         center_2d = np.array([po['kinematic']['p1'], po['kinematic']['p2']])
        #         points_in_box = label_points[is_point_in_polygon(label_points[:, :2], center_2d, eigenvalues, eigenvectors)]
        #         if points_in_box.shape[0] > 0:
        #             min_z, max_z = np.min(points_in_box[:, 2]), np.max(points_in_box[:, 2])
        #             center = np.array([po['kinematic']['p1'], po['kinematic']['p2'], (min_z+max_z)/2])
        #             rotation_2d = np.array([
        #                             [eigenvectors[0,0], -eigenvectors[1,0]],
        #                             [eigenvectors[1,0], eigenvectors[0,0]]
        #                         ])
        #             rotation_matrix = np.eye(3)
        #             rotation_matrix[:2, :2] = rotation_2d
        #             extents = np.array([eigenvalues[0], eigenvalues[1], (max_z-min_z)])
        #             bb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, extents)
        #             bb.color = [0, 0, 1]
        #             bboxes.append(bb)

        #     curr_pcd = imo_pcd_reader.read_AL_pcd_with_excluded_area(data_pth, excluded_area, ceiling_height)
        #     curr_coord = curr_pcd[:, :3]
        #     trackIDs = [po['label']['label_v'] for po in pos_in_frame]
            
        #     point_cloud = pv.PolyData(curr_coord)
        #     pl = pv.Plotter(off_screen=True)
        #     # pl = pv.Plotter()
        #     camera_height = 3
        #     focal_point_distance = camera_height * np.tan(np.deg2rad(60))
        #     pl.camera_position = [(0, 0, 120), (0, 0, -1), (1, 0, 0)] # BEV
        #     # pl.camera_position = [(0, -10, camera_height), (0, focal_point_distance, 0), (0, 0, 1)]
        #     pl.add_mesh(
        #         point_cloud,
        #         # render_points_as_spheres=True,
        #         point_size=2,
        #         # color="black",
        #         # cmap="jet",  # Optional: Use colormap for scalar data visualization
        #         # scalars="intensity"  # Uncomment if your point cloud has intensity data
        #     )
        #     pl.show_axes()

        #     for i, bbox in enumerate(bboxes):
        #         obb = create_obb(bbox)
        #         pl.add_mesh(obb, color="blue", show_edges=True, opacity=0.5)
        #         # Add label as 3D text
        #         pl.add_point_labels(
        #             bbox.center,  # Convert center to a numpy array
        #             [str(trackIDs[i])],  # Pass label as a list
        #             point_size=10,
        #             text_color="white",
        #             font_size=12,
        #             always_visible=True,
        #             name="label_"+str(trackIDs[i]),
        #         )

        #     # pl.show()

        #     pl.render()

        #     data_base_name, data_extension = os.path.splitext(os.path.basename(data_pth))
        #     filename = os.path.join(stream_dir, data_base_name+".png")
        #     pl.screenshot(filename=filename)

        #     # # Close the plotter
        #     pl.close()


        # png_names = [f for f in os.listdir(stream_dir) if f.endswith(".png")]
        # png_names.sort()
        # image_files = [os.path.join(stream_dir, name) for name in png_names]

        # # Read the images
        # images = [imageio.imread(filename) for filename in image_files]

        # # Create the GIF
        # imageio.mimsave(os.path.join(saved_parent_dir, "preview.gif"), images, fps=2)

parent_dir = os.path.abspath(os.path.join(args.seg_result_dir, os.pardir))
out_pth = os.path.join(parent_dir, "ptv3_bbox_result.txt")
with open(out_pth, "w") as file:
    for data in out_data_list:
        json.dump(data, file)
        file.write("\n")  # Add newline after each dictionary