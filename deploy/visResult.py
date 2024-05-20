import os
import sys
import argparse

import numpy as np
import open3d as o3d
import imo_pcd_reader

# params
parser = argparse.ArgumentParser(description='Point Cloud Segmentation Result Visulization')

parser.add_argument('--result_base_dir', type=str, 
                    default='/your/path/.../result')

args = parser.parse_args()


# bin_file_path = '/home/xin/Downloads/nuscene3d/sweeps/LIDAR_TOP/n008-2018-08-27-11-48-51-0400__LIDAR_TOP__1535385032199407.pcd.bin'
# data = np.fromfile(bin_file_path, dtype=np.float32)
# points = data.reshape(-1, 5)

# coord_intensities = np.copy(points[:, :4])

# imo_pcd_reader.save_imo_pcd(coord_intensities, "nuscenes_pointcloud.pcd")



# Load the segmentation results
segmentation_data = np.load('/home/xin/Downloads/test_result/result/20240419_161208300_idchigh_0002_1_0047_pred.npy')

scan = imo_pcd_reader.read_pcd('/home/xin/Documents/Driving_PC_240419_idchigh_0002_Test/sweeps/scene10/lidarTop/20240419_161208300_idchigh_0002_1_0047.pcd')

# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(coord)

save_pcd_path = '/home/xin/Downloads/test_result/result/20240419_161208300_idchigh_0002_1_0047_pred.pcd'

imo_pcd_reader.save_pcd(scan, segmentation_data, save_pcd_path)




# # Extract XYZ and labels
# xyz_points = segmentation_data[:, :3]
# labels = segmentation_data[:, 3]  # If labels are present

# # Create the PointCloud object
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(xyz_points)

# # Assign colors based on labels
# colors = get_colors_for_labels(labels)
# pcd.colors = o3d.utility.Vector3dVector(colors)

# # Visualize
# o3d.visualization.draw_geometries([pcd])

# # Save to PCD
# o3d.io.write_point_cloud("segmented_pointcloud.pcd", pcd)