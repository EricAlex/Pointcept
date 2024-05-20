"""
imotion Dataset

Author: Xin Wang (xin.wang@imotion.ai)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
from collections.abc import Sequence
import glob

from .builder import DATASETS
from .defaults import DefaultDataset

import imo_pcd_reader


@DATASETS.register_module()
class ImotionDataset(DefaultDataset):
    def __init__(
        self,
        split=(),
        data_root="data/imotion",
        sweeps=10,
        transform=None,
        test_mode=False,
        test_cfg=None,
        loop=1,
        ignore_index=-1,
    ):
        self.string_to_add = "lidarTop"
        self.data_root = data_root
        self.sweeps = sweeps
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        super().__init__(
            split=split,
            data_root=data_root,
            transform=transform,
            test_mode=test_mode,
            test_cfg=test_cfg,
            loop=loop,
        )

    def get_data_list(self):
        if isinstance(self.data_root, str):
            scene_paths = glob.glob(os.path.join(self.data_root, "scene*"))
            scene_names = [os.path.basename(path) for path in scene_paths if os.path.isdir(path)]
            scene_names = sorted(scene_names, key=lambda x: int(x.replace("scene", "")))
        split_list = [os.path.join(scene, self.string_to_add) for scene in scene_names]
        pc_extension = ".pcd"
        if isinstance(split_list, Sequence):
            data_list = []
            for split in split_list:
                this_dir = os.path.join(self.data_root, split)
                pcd_names = [f for f in os.listdir(this_dir) if f.endswith(pc_extension)]
                pcd_names.sort()
                data_list += [os.path.join(this_dir, name) for name in pcd_names]
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        # Read the PCD file
        scan = imo_pcd_reader.read_pcd(self.data_list[idx % len(self.data_list)])
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1]) / 255
        segment = np.ones((scan.shape[0],), dtype=np.int64) * self.ignore_index
        data_dict = dict(coord=coord, strength=strength, segment=segment)
        return data_dict

    def get_data_name(self, idx):
        file_name, extension = os.path.splitext(os.path.basename(self.data_list[idx % len(self.data_list)]))
        return file_name
    
    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,
            1: ignore_index,
            2: 6,
            3: 6,
            4: 6,
            5: ignore_index,
            6: 6,
            7: ignore_index,
            8: ignore_index,
            9: 0,
            10: ignore_index,
            11: ignore_index,
            12: 7,
            13: ignore_index,
            14: 1,
            15: 2,
            16: 2,
            17: 3,
            18: 4,
            19: ignore_index,
            20: ignore_index,
            21: 5,
            22: 8,
            23: 9,
            24: 10,
            25: 11,
            26: 12,
            27: 13,
            28: 14,
            29: ignore_index,
            30: 15,
            31: ignore_index,
        }
        return learning_map
