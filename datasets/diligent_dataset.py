from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import glob
import torch
import torchvision
from scipy.io import loadmat
from data_processing.depth_to_depth_z import *
from datasets.synthetic_dataset_utils import *
import data_processing.optical_flow_funs as OF
import scipy.ndimage
import random
import json
import cv2
from collections import defaultdict
import copy


class DiligentDataset(Dataset):

    def __init__(self, dataset_root, mode='train', domains_to_load=['albedo', 'roughness', 'normal', 'depth', 'mask'],
                 transform=None):

        self.dataset_root = dataset_root
        self.scene_dirs = glob.glob(os.path.join(dataset_root, '*'))
        self.scene_dirs = sorted([x for x in self.scene_dirs if os.path.isdir(x)])
        self.mode = mode
        self.domains_to_load = domains_to_load
        self.transform = transform

        print('Found %d DiLiGenT scenes' % len(self.scene_dirs))

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene_idx = idx
        scene_path = self.scene_dirs[scene_idx]


        sample = {}
        sample['name'] = os.path.basename(scene_path)
        normal_path = os.path.join(scene_path, "normal_map.png")
        mask_path = os.path.join(scene_path, "mask.png")
        K = np.loadtxt(os.path.join(scene_path, 'K.txt'))
        intrinsics = np.eye(3)
        intrinsics[0, 0] = K[1, 1]
        intrinsics[1, 1] = K[0, 0]
        intrinsics[0, 2] = K[1, 2]
        intrinsics[1, 2] = K[0, 2]
        scene_name = scene_path.split('/')[-1]
        depth_gt_path = os.path.join(scene_path, f"{scene_name}_gt.mat")
        n_intrinsics = OF.pixel_intrinsics_to_normalized_intrinsics(torch.from_numpy(intrinsics).unsqueeze(0).float(),
                                                                    (512, 612)).squeeze()
        # sample['intrinsics'] = n_intrinsics
        sample['n_intrinsics'] = n_intrinsics
        normal_map = cv2.cvtColor(cv2.imread(normal_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR)
        if normal_map.dtype is np.dtype(np.uint16):
             normal_map = normal_map / 65535 * 2 - 1
        else:
             normal_map = normal_map / 255 * 2 - 1

        mask = cv2.imread(os.path.join(mask_path), cv2.IMREAD_GRAYSCALE).astype(bool)

        depth_gt = loadmat(depth_gt_path)["depth_gt"]
        depth_gt[np.isnan(depth_gt)] = 0
        nx = normal_map[:, :, 0]
        ny = -normal_map[:, :, 1]
        nz = -normal_map[:, :, 2]
        nmag = np.sqrt(nx * nx + ny * ny + nz * nz)
        nmag[nmag == 0] = 1
        normal_map = np.stack((nx / nmag, ny / nmag, nz / nmag), axis=2)

        sample['normal'] = normal_map
        sample['mask'] = mask > 0.1
        sample['depth'] = depth_gt #- 1300
        sample['m'] = np.mean(depth_gt.squeeze()[mask.squeeze()>0.1] ).astype(np.float32)
        sample['f'] = n_intrinsics[0, 0]

        if self.transform:
            self.transform(sample)


        return sample


