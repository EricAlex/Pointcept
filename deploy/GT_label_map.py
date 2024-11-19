import os
import sys
import argparse
import glob
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

import numpy as np
import imo_pcd_reader

#### train label standards ####
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

label_map43 = {
    1: 2, # car to car
    2: 2, # van to car
    3: 10, # bus to bus
    4: 11, # mini_truck to truck
    5: 11, # truck to truck
    6: 12, # special_vehicle to special_vehicle
    7: 13, # tricycle to tricycle
    8: 3, # bicycle to bicycle
    9: 13, # tricyclist to tricycle
    10: 3, # bicyclist to bicycle
    11: 9, # pedestrain to pedestrain
    12: 14, # traffic_cone to traffic_cone
    13: 15, # crash_barrel to crash_barrel
    14: 7, # construction_sign to construction_sign
    15: 5, # high_intensity_caution to traffic_board
    16: 16, # splash_noise to noise
    17: 16, # dust_noise to noise
    18: 16, # fumes_noise to noise
    19: 16, # ghost_noise to noise
    20: 16, # dilation_noise to noise
    21: 23, # movable_obstacle to ground_obstacle
    22: 23, # unmovable_obstacle to ground_obstacle
    23: 0, # road to road
    24: 17, # sidewalks to sidewalks
    25: 6, # terrain to vegetation
    26: 6, # bosk to vegetation
    27: 6, # tree to vegetation
    28: 1, # curb_stone to curb
    29: 18, # fence to fence
    30: 1, # water_horse to curb
    31: 19, # wall to wall
    32: 0, # lane_line to road
    33: 19, # tunnel to wall
    34: 24, # portal to floating_obstacle
    35: 24, # lift_rod to floating_obstacle
    36: 20, # bump to bump
    37: 21, # architecture to architecture
    38: 4, # traffic_pole to traffic_pole
    39: 5, # traffic_board to traffic_board
    40: 24, # ceiling to floating_obstacle
    41: 22, # pillars to pillars
    42: 8, # others to others
    43: 26, # undefined to undefined
}

label_map27 = {
    1: 0, # road to road
    2: 1, # curb to curb
    3: 2, # car to car
    4: 3, # bicycle to bicycle
    5: 4, # traffic_pole to traffic_pole
    6: 5, # traffic_board to traffic_board
    7: 6, # vegetation to vegetation
    8: 7, # construction_sign to construction_sign
    9: 8, # others to others
    10: 9, # pedestrain to pedestrain
    11: 10, # bus to bus
    12: 11, # truck to truck
    13: 12, # special_vehicle to special_vehicle
    14: 13, # tricycle to tricycle
    15: 14, # traffic_cone to traffic_cone
    16: 15, # crash_barrel to crash_barrel
    17: 16, # noise to noise
    18: 17, # sidewalks to sidewalks
    19: 18, # fence to fence
    20: 19, # wall to wall
    21: 24, # portal to floating_obstacle
    22: 20, # bump to bump
    23: 21, # architecture to architecture
    24: 24, # ceiling to floating_obstacle
    25: 22, # pillars to pillars
    26: 23, # ground_obstacle to ground_obstacle
    27: 24, # floating_obstacle to floating_obstacle
    28: 25, # animal to animal
    43: 26, # undefined to undefined
}

# params
parser = argparse.ArgumentParser(description='Point Cloud GT label mapping')

parser.add_argument('--GT_label_pcds_dir', type=str, 
                    default='/your/path/.../GT_label_pcds')

parser.add_argument('--save_dir', type=str, 
                    default='/your/path/.../GT_label_pcds_4train')

parser.add_argument('--num_classes', type=int, default=27)

args = parser.parse_args()

valid_range = 100 # meters

def mapping_save(gtlabel_pcd_pth, negative_selected_label):
    pcd_name = os.path.basename(gtlabel_pcd_pth)
    pcd = imo_pcd_reader.read_IMOGTL_pcd_negative_selected_label(gtlabel_pcd_pth, negative_selected_label, valid_range)
    scan = pcd[:, :4]
    gt_label = pcd[:, -1]

    max_label_value = np.max(gt_label)
#    print(f"pcd_name: {pcd_name}, num_classes: {args.num_classes}, max_label_value: {max_label_value}")
    if args.num_classes == 43 and max_label_value < 30:
        print(f"pcd_name: {pcd_name}, num_classes: {args.num_classes}, max_label_value: {max_label_value}")
    if args.num_classes == 27 and max_label_value > 28:
        print(f"pcd_name: {pcd_name}, num_classes: {args.num_classes}, max_label_value: {max_label_value}")

    if args.num_classes == 43:
        mapped_gt_label = np.vectorize(label_map43.get)(gt_label)
    if args.num_classes == 27:
        mapped_gt_label = np.vectorize(label_map27.get)(gt_label)
        
    save_pcd_path = os.path.join(args.save_dir, pcd_name)
    imo_pcd_reader.save_pcd(scan, mapped_gt_label, save_pcd_path)

if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

pc_extension = ".pcd"
negative_selected_label = [43]
pcd_names = [f for f in os.listdir(args.GT_label_pcds_dir) if f.endswith(pc_extension)]
pcd_names.sort()
data_list = [os.path.join(args.GT_label_pcds_dir, name) for name in pcd_names]
Parallel(n_jobs=max(1, multiprocessing.cpu_count()/2))(delayed(mapping_save)(pcd_pth, negative_selected_label) for pcd_pth in tqdm(data_list, total=len(data_list), mininterval=1.0))

#for pcd_pth in data_list:
#    mapping_save(pcd_pth, negative_selected_label)
