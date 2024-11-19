import os
import sys
import argparse
import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

import numpy as np
import imo_pcd_reader

# params
parser = argparse.ArgumentParser(description='Point Cloud Segmentation Result Visulization')

parser.add_argument('--result_base_dir', type=str, 
                    default='/your/path/.../result')

parser.add_argument('--data_base_dir', type=str, 
                    default='/your/path/.../dataset/sweeps')

parser.add_argument('--pcd_channel_name', type=str, 
                    default='lidarFusion_pcd')

args = parser.parse_args()

def findLabelSave(pcd_pth, save_pth, scene_n="scene"):
    pcd_name = os.path.basename(pcd_pth)
    base_name, extension = os.path.splitext(pcd_name)
    label_file_list = glob.glob(os.path.join(args.result_base_dir, base_name+"*pred.npy"))
    if len(label_file_list) == 1:
        segmentation_data = np.load(label_file_list[0])
        scan = imo_pcd_reader.read_pcd(pcd_pth)
        save_pcd_path = os.path.join(save_pth, base_name+"_seged_"+scene_n+".pcd")
        imo_pcd_reader.save_pcd(scan, segmentation_data, save_pcd_path)
    else:
        print(f"Found {len(label_file_list)} seg label file for {pcd_pth}")

folder_paths = glob.glob(os.path.join(args.data_base_dir, "*scene*"))
scene_names = [os.path.basename(path) for path in folder_paths if os.path.isdir(path)]

parent_dir = os.path.abspath(os.path.join(args.result_base_dir, os.pardir))
save_dir = os.path.join(parent_dir, "seged_pcds")
if not os.path.exists(save_dir): os.makedirs(save_dir)

pcd_channel_name = args.pcd_channel_name
pc_extension = ".pcd"
if not scene_names:
    pcd_channel_name = "sensor_data/LIDAR_TOP_LIDAR_BLINDZONE_MERGED"
    this_dir = os.path.join(args.data_base_dir, pcd_channel_name)
    pcd_names = [f for f in os.listdir(this_dir) if f.endswith(pc_extension)]
    pcd_names.sort()
    for pcd_n in pcd_names:
        findLabelSave(os.path.join(this_dir, pcd_n), save_dir)
else:
    if len(scene_names) == 1:
        scene_n = scene_names[0]
        this_dir = os.path.join(args.data_base_dir, scene_n, pcd_channel_name)
        pcd_names = [f for f in os.listdir(this_dir) if f.endswith(pc_extension)]
        pcd_names.sort()
        for pcd_n in tqdm(pcd_names, total=len(pcd_names), mininterval=1.0):
            findLabelSave(os.path.join(this_dir, pcd_n), save_dir, scene_n)
    else:
        for scene_n in scene_names:
            this_dir = os.path.join(args.data_base_dir, scene_n, pcd_channel_name)
            pcd_names = [f for f in os.listdir(this_dir) if f.endswith(pc_extension)]
            pcd_names.sort()
            with Parallel(n_jobs=max(1, int(multiprocessing.cpu_count()/len(scene_names))), backend='loky') as parallel:
                parallel(delayed(findLabelSave)(os.path.join(this_dir, pcd_n), save_dir, scene_n) 
                            for pcd_n in tqdm(pcd_names, total=len(pcd_names), mininterval=1.0))
