import torch
import torch.utils.data
import os
import pickle
import cv2
from utils import *
from scipy.io import loadmat
from tqdm import tqdm
import torchvision
import torch.nn as nn
from PIL import Image
import random
import numpy as np
import numpy.ma as ma
import torchvision.transforms as transforms
opj = os.path.join

class YCB_Dataset(torch.utils.data.Dataset):
    def __init__(self, root, imageset_path, syn_data_path=None, use_real_img = True, num_syn_images=200000 ,target_h=76, target_w=76
                 , bg_path = None, kp_path="data/YCB-Video/YCB_bbox.npy", data_cfg="data/data-YCB.cfg",
                 use_bg_img = True):
        self.root = root
        data_options = read_data_cfg(data_cfg)
        self.input_width = int(data_options['width']) # 608, width of CNN input
        self.input_height = int(data_options['height'])
        self.original_width = 640 # width of original img
        self.original_height = 480
        self.target_h = target_h # after network
        self.target_w = target_w
        self.num_classes = int(data_options['classes'])
        self.train_paths = []
        #self.gen_train_list(imageset_path)

        self.use_real_img = use_real_img
        self.syn_data_path = syn_data_path
        self.syn_range = 80000 # YCB has 80000 syn images in total, continuously indexed
        self.syn_bg_image_paths = get_img_list_from(bg_path) if bg_path is not None else []
        self.use_bg_img = use_bg_img
        self.num_syn_images = num_syn_images # syn images for training

        self.weight_cross_entropy = None
        #self.set_balancing_weight()
        self.object_names_ycbvideo = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can',
                                 '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box',
                                 '010_potted_meat_can', '011_banana', '019_pitcher_base', '021_bleach_cleanser',
                                 '024_bowl', '025_mug', '035_power_drill',
                                 '036_wood_block', '037_scissors', '040_large_marker', '051_large_clamp',
                                 '052_extra_large_clamp',
                                 '061_foam_brick']
        self.ycb_class_to_idx = {}
        for i, item in enumerate(self.object_names_ycbvideo):
            self.ycb_class_to_idx[item] = i

        self.kp3d = np.load(kp_path)
        self.n_kp = 8



    def __getitem__(self, index):
        if not self.use_real_img:
            return self.gen_synthetic()
        if index > len(self.train_paths) - 1:
            return self.gen_synthetic()
        else:
            prefix = self.train_paths[index]
            # get raw image
            raw = cv2.imread(prefix + "-color.png")
            img = cv2.resize(raw, (self.input_height, self.input_width))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # load class info
            meta = loadmat(prefix + '-meta.mat')
            class_ids = meta['cls_indexes']

            # get segmentation gt, note 0 is for background
            label_img = cv2.imread(prefix + "-label.png")[: , : , 0]
            label_img = cv2.resize(label_img, (self.target_h, self.target_w), interpolation=cv2.INTER_NEAREST)

            # generate kp gt map of (nH, nW, nV)
            kp_gt_map_x = np.zeros((self.target_h, self.target_w, self.n_kp))
            kp_gt_map_y = np.zeros((self.target_h, self.target_w, self.n_kp))
            in_pkl = prefix + '-bb8_2d.pkl'
            with open(in_pkl, 'rb') as f:
                bb8_2d = pickle.load(f)
            for i, cid in enumerate(class_ids):
                class_mask = np.where(label_img == cid[0])
                kp_gt_map_x[class_mask] = bb8_2d[:,:,0][i]
                kp_gt_map_y[class_mask] = bb8_2d[:,:,1][i]

            mask_front = ma.getmaskarray(ma.masked_not_equal(label_img, 0)).astype(int)
            #TODO: get mask weighted by class
            return (torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0),
                    torch.from_numpy(label_img).long(),
                    torch.from_numpy(kp_gt_map_x).float(), torch.from_numpy(kp_gt_map_y).float(),
                    torch.from_numpy(mask_front).float())

    def __len__(self):
        if self.use_real_img:
            return len(self.train_paths)+self.num_syn_images
        else:
            return self.num_syn_images

