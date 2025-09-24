'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''

import random

import torch
import numpy as np
import torch.utils.data as data

import yaml
from easydict import EasyDict

from utils.io import IO
from utils.build import DATASETS
from utils.logger import *
from utils.build import build_dataset_from_cfg
import json
from tqdm import tqdm
import pickle
from PIL import Image
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

from timm.data import create_transform

from data.utils import pc_normalize, offread_uniformed
import data.data_transforms as pcl_transforms
from data.DataAugmentation import DataAugmentation
import torchvision.transforms.functional as TF

synonym_map = {
    # 浴室相关
    "bath": "bathtub",
    "bathing tub": "bathtub",
    "tub": "bathtub",
    # 沙发相关
    "lounge": "sofa",
    "couch": "sofa",
    # 麦克风相关
    "mike": "microphone",
    # 飞机相关
    "plane": "airplane",
    "aeroplane": "airplane",
    # 火车相关
    "railroad train": "train",
    # 水龙头相关
    "spigot": "faucet",
    # 汽车相关
    "machine": "car",
    "motorcar": "car",
    "automobile": "car",
    "auto": "car",
    # 船只相关
    "watercraft": "boat",
    # 手机相关
    "cellular phone": "telephone",
    "cell": "telephone",
    "mobile phone": "telephone",
    "cellular telephone": "telephone",
    "cellphont": "telephone",
    # 显示器相关
    "video display": "monitor",
    # 公交车相关
    "jitney": "bus",
    "motorbus": "bus",
    "charabanc": "bus",
    "autobus": "bus",
    "double-decker": "bus",
    "motorcoach": "bus",
    "coach": "bus",
    "passenger vehi": "bus",  # 补充完整语义为"passenger vehicle"，此处统一映射为bus
    "omnibus": "bus",
    # 扬声器相关
    "speaker system": "loudspeaker",
    "speaker unit": "loudspeaker",
    "loudspeaker system": "loudspeaker", # 补充常见同义词
    # 遥控器相关
    "remote": "remote control",
    # 花盆相关
    "flowerpot": "potted plant",
    # 电话相关
    "telephone set": "telephone",
    "phone": "telephone",  # 注意：此处"phone"指固定电话，与手机"cellphone"区分
    # 耳机相关
    "earpiece": "headphone",
    # 文件柜相关
    "filing cabinet": "file cabinet",
    "file": "file cabinet",
    # 笔记本电脑相关
    "laptop computer": "laptop",
    # 钢琴相关
    "forte-piano": "piano",
    "pianoforte": "piano",
    # 打印机相关
    "printing machine": "printer",
    # 自行车相关
    "bike": "bicycle",
    # 洗衣机相关
    "automatic washer": "washing machine",
    # 洗碗机相关
    "dish washer": "dishwasher",
    "dishwashing machine": "dishwasher",
    # 手枪相关
    "handgun": "gun",
    "shooting iron": "gun",
    "side arm": "gun",
    # 手提箱相关
    "grip": "suitcase",
    "traveling bag": "suitcase",
    "travelling bag": "suitcase",  # 英式拼写统一映射
    # 垃圾桶相关
    "ash-bin": "trash can",
    "ashbin": "trash can",
    "wastebin": "trash can",
    "ash bin": "trash can",
    "trash bin": "trash can",
    "garbage can": "trash can",
    "trash barrel": "trash can",
    "dustbin": "trash can",
    # 易拉罐相关
    "tin can": "can",
    "tin": "can",
    # 微波炉相关
    "microwave oven": "microwave",
    # 手提篮相关
    "handbasket": "basket",
    # 键盘
    "computer keyboard": "keyboard"
}

cats = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8,
        'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14, 'flower_pot': 15, 'glass_box': 16,
        'guitar': 17, 'keyboard': 18, 'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23,
        'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29, 'sofa': 30, 'stairs': 31,
        'stool': 32, 'table': 33, 'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}

cats_medium = {'cone': 0, 'cup': 1, 'curtain': 2, 'door': 3, 'dresser': 4, 'glass_box': 5, 'mantel': 6, 'monitor': 7, 'night_stand': 8,
        'person': 9, 'plant': 10, 'radio': 11, 'range_hood': 12, 'sink': 13, 'stairs': 14, 'stool': 15, 'tent': 16, 'toilet': 17, 'tv_stand': 18, 'vase': 19,
        'wardrobe': 20, 'xbox': 21}

cats_hard = {'cone': 0, 'curtain': 1, 'door': 2, 'dresser': 3, 'glass_box': 4, 'mantel': 5, 'night_stand': 6,
        'person': 7, 'plant': 8, 'radio': 9, 'range_hood': 10, 'sink': 11, 'stairs': 12, 'tent': 13, 'toilet': 14, 'tv_stand': 15, 'xbox': 16}

cats_modelnet10 = {'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4, 'monitor': 5, 'night_stand': 6, 'sofa': 7, 'table':8, 'toilet': 9}

cats_scanobject = {'bag': 0, 'bin': 1, 'box': 2, 'cabinet': 3, 'chair': 4, 'desk': 5, 'display': 6, 'door': 7, 'shelf': 8,
        'table': 9, 'bed': 10, 'pillow': 11, 'sink': 12, 'sofa': 13, 'toilet': 14}

cats_redwood = {'chair': 0, 'table': 1, 'trash container': 2, 'bench': 3, 'plant': 4, 'sign': 5, 'bicycle': 6, 'motorcycle': 7, 'sofa': 8}

cats_pix3d = {'bed': 0, 'bookcase': 1, 'chair': 2, 'desk': 3, 'sofa': 4, 'table': 5, 'tool': 6, 'wardrobe': 7}

# cats_co3d = {'frisbee':0 , 'bench':1 , 'skateboard':2 , 'backpack':3, 'laptop':4, 'cup':5, 'umbrella':6, 'teddybear':7,
#              'bowl':8, 'cake':9, 'toyplane':10, 'remote':11, 'orange':12, 'toaster':13, 'couch':14, 'apple':15, 'bottle':16,
#              'donut':17, 'hairdryer':18, 'tv':19, 'hydrant':20, 'toytrain':21, 'cellphone':22, 'toybus':23, 'car':24, 'pizza':25,
#              'motorcycle':26, 'suitcase':27, 'banana':28, 'toilet':29, 'sandwich':30, 'keyboard':31, 'mouse':32, 'parkingmeter':33,
#              'toytruck':34, 'hotdog':35, 'handbag':36, 'kite':37, 'baseballbat':38, 'broccoli':39, 'wineglass':40, 'microwave':41,
#              'baseballglove':42, 'book':43, 'carrot':44, 'ball':45, 'bicycle':46, 'chair':47, 'stopsign':48, 'vase':49, 'plant':50}

cats_co3d = {'bench':0 , 'skateboard':1 , 'backpack':2, 'laptop':3, 'cup':4, 'umbrella':5, 'teddybear':6,
             'bowl':7, 'cake':8, 'toyplane':9, 'remote':10, 'orange':11, 'toaster':12, 'couch':13, 'apple':14, 'bottle':15,
             'donut':16, 'hairdryer':17, 'hydrant':18, 'toytrain':19, 'pizza':20,
             'motorcycle':21, 'suitcase':22, 'banana':23, 'toilet':24, 'sandwich':25, 'keyboard':26, 'mouse':27,
             'toytruck':28, 'handbag':29, 'broccoli':30, 'wineglass':31, 'microwave':32,
             'baseballglove':33, 'book':34, 'carrot':35, 'ball':36, 'bicycle':37, 'chair':38, 'stopsign':39, 'vase':40, 'plant':41}


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
    return batch_pc


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


import os, sys, h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


@DATASETS.register_module()
class ModelNet(data.Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.npoints
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        self.generate_from_raw_data = False
        split = config.subset
        self.subset = config.subset

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test' or split == 'val')
        if split == 'val': split = 'test'
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print_log('The size of %s data is %d' % (split, len(self.datapath)), logger='ModelNet')

        if self.uniform:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root,
                                          'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                # make sure you have raw data in the path before you enable generate_from_raw_data=True.
                if self.generate_from_raw_data:
                    print_log('Processing data %s (only running in the first time)...' % self.save_path,
                              logger='ModelNet')
                    self.list_of_points = [None] * len(self.datapath)
                    self.list_of_labels = [None] * len(self.datapath)

                    for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                        fn = self.datapath[index]
                        cls = self.classes[self.datapath[index][0]]
                        cls = np.array([cls]).astype(np.int32)
                        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                        if self.uniform:
                            point_set = farthest_point_sample(point_set, self.npoints)
                            print_log("uniformly sampled out {} points".format(self.npoints))
                        else:
                            point_set = point_set[0:self.npoints, :]

                        self.list_of_points[index] = point_set
                        self.list_of_labels[index] = cls

                    with open(self.save_path, 'wb') as f:
                        pickle.dump([self.list_of_points, self.list_of_labels], f)
                else:
                    # no pre-processed dataset found and no raw data found, then load 8192 points dataset then do fps after.
                    self.save_path = os.path.join(self.root,
                                                  'modelnet%d_%s_%dpts_fps.dat' % (
                                                      self.num_category, split, 8192))
                    print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
                    print_log(
                        'since no exact points pre-processed dataset found and no raw data found, load 8192 pointd dataset first, then do fps to {} after, the speed is excepted to be slower due to fps...'.format(
                            self.npoints), logger='ModelNet')
                    with open(self.save_path, 'rb') as f:
                        self.list_of_points, self.list_of_labels = pickle.load(f)

            else:
                print_log('Load processed data from %s...' % self.save_path, logger='ModelNet')
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

        self.shape_names_addr = os.path.join(self.root, 'modelnet40_shape_names.txt')
        with open(self.shape_names_addr) as file:
            lines = file.readlines()
            lines = [line.rstrip() for line in lines]
        self.shape_names = lines

        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height

    def __len__(self):
        return len(self.list_of_labels)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

        if self.npoints < point_set.shape[0]:
            point_set = farthest_point_sample(point_set, self.npoints)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.use_height:
            self.gravity_dim = 1
            height_array = point_set[:, self.gravity_dim:self.gravity_dim + 1] - point_set[:,
                                                                                 self.gravity_dim:self.gravity_dim + 1].min()
            point_set = np.concatenate((point_set, height_array), axis=1)

        return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        label_name = self.shape_names[int(label)]

        # return current_points, label, label_name

        return current_points, current_points, label


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):

        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer
        # self.train_transform = config.train_transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize
        ])
        self.id_map_addr = os.path.join(config.DATA_PATH, 'taxonomy.json')
        self.rendered_image_addr = config.IMAGE_PATH

        f = open(os.path.join(config.DATA_PATH, "shape_names.txt"))
        self.classes = f.readlines()
        for i in range(len(self.classes)):
            self.classes[i] = self.classes[i][:-1]
        f.close()

        classes_dict = {}
        for i in range(55):
            classes_dict[self.classes[i]] = i
        self.classes = classes_dict

        self.picked_image_type = ['', '_depth0001']
        # self.picked_image_type = ['']

        self.picked_rotation_degrees = list(range(0, 360, 12))
        self.picked_rotation_degrees = [(3 - len(str(degree))) * '0' + str(degree) if len(str(degree)) < 3 else str(degree) for degree in self.picked_rotation_degrees]
        # self.picked_rotation_degrees = list(range(0, 12, 1))

        with open(self.id_map_addr, 'r') as f:
            self.id_map = json.load(f)

        self.prompt_template_addr = os.path.join('./data/templates.json')
        with open(self.prompt_template_addr) as f:
            self.templates = json.load(f)[config.pretrain_dataset_prompt]

        self.synset_id_map = {}
        for id_dict in self.id_map:
            synset_id = id_dict["synsetId"]
            self.synset_id_map[synset_id] = id_dict

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')

        self.sample_points_num = self.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger='ShapeNet-55')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line[len(taxonomy_id) + 1:].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet-55')

        self.permutation = np.arange(self.npoints)

        self.uniform = True
        self.augment = True
        self.use_caption_templates = False
        # =================================================
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height
        # =================================================

        if self.augment:
            print("using augmented point clouds.")

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        if self.uniform and self.sample_points_num < data.shape[0]:
            data = farthest_point_sample(data, self.sample_points_num)
        else:
            data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)

        if self.augment:
            data = random_point_dropout(data[None, ...])
            data = random_scale_point_cloud(data)
            data = shift_point_cloud(data)
            data = rotate_perturbation_point_cloud(data)
            data = rotate_point_cloud(data)
            data = data.squeeze()

        if self.use_height:
            self.gravity_dim = 1
            height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
                                                                            self.gravity_dim:self.gravity_dim + 1].min()
            data = np.concatenate((data, height_array), axis=1)
            data = torch.from_numpy(data).float()
        else:
            data = torch.from_numpy(data).float()

        captions = self.synset_id_map[sample['taxonomy_id']]['name']
        captions = [caption.strip() for caption in captions.split(',') if caption.strip()]
        
        # missing = set()
        # for caption in captions:   
        #     if caption not in self.classes:
        #         missing.add(caption)
        # print(f"missing:{missing}")
        
        caption = captions[0] #random.choice(captions)
        # 如果 caption 在 synonym_map 里，就替换成标准类别
        if caption in synonym_map:
            caption = synonym_map[caption]

        # 如果 caption 仍然不在 self.classes，就提示一下
        if caption not in self.classes:
            raise KeyError(f"Unknown class name: {caption}")

        label = self.classes[caption]

        captions = []
        tokenized_captions = []
        if self.use_caption_templates:
            for template in self.templates:
                caption = template.format(caption)
                captions.append(caption)
                tokenized_captions.append(self.tokenizer(caption))
        else:
            tokenized_captions.append(self.tokenizer(caption))

        tokenized_captions = torch.stack(tokenized_captions)

        # picked_model_rendered_image_addr = self.rendered_image_addr + '/img/'
        # picked = random.randint(0, 11)
        # picked_image_name = sample['taxonomy_id'] + '/' + sample['model_id'] + '/' + '0' * (3 - len(str(picked))) + str(
        #     picked) + '.png'
        # picked_image_addr = picked_model_rendered_image_addr + picked_image_name


        picked_model_rendered_image_addr = self.rendered_image_addr + '/' +\
                                           sample['taxonomy_id'] + '-' + sample['model_id'] + '/'
        picked_image_name = sample['model_id'] + '_r_' +\
                            str(random.choice(self.picked_rotation_degrees)) +\
                            random.choice(self.picked_image_type) + '.png'
        picked_image_addr = picked_model_rendered_image_addr + picked_image_name
        
        # print("=========================================")
        # print(f"label:{label},caption:{caption}")
        # print(f"picked_image_addr:{picked_image_addr}")
        # print("=========================================")

        try:
            image = pil_loader(picked_image_addr)
            image = self.train_transform(image)
        except:
            raise ValueError("image is corrupted: {}".format(picked_image_addr))

        # return sample['taxonomy_id'], sample['model_id'], tokenized_captions, data, image
        # label = self.classes.index(caption)
        return (image, data, tokenized_captions), label

    def __len__(self):
        return len(self.file_list)
    
@DATASETS.register_module()
class ShapeNet_4modal(data.Dataset):
    def __init__(self, config):

        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.depth_path = config.DEPTH_PATH
        self.subset = config.subset
        self.npoints = config.npoints
        self.tokenizer = config.tokenizer
        
        self.input_size = config.get('input_size', 224)  # 默认值设为224
        
        # self.train_transform = config.train_transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            normalize
        ])
        self.id_map_addr = os.path.join(config.DATA_PATH, 'taxonomy.json')
        self.rendered_image_addr = config.IMAGE_PATH
        self.rendered_depth_addr = config.DEPTH_PATH

        f = open(os.path.join(config.DATA_PATH, "shape_names.txt"))
        self.classes = f.readlines()
        for i in range(len(self.classes)):
            self.classes[i] = self.classes[i][:-1]
        f.close()

        classes_dict = {}
        for i in range(55):
            classes_dict[self.classes[i]] = i
        self.classes = classes_dict

        self.picked_image_type = ['', '_depth0001']
        # self.picked_image_type = ['']

        self.picked_rotation_degrees = list(range(0, 360, 12))
        self.picked_rotation_degrees = [(3 - len(str(degree))) * '0' + str(degree) if len(str(degree)) < 3 else str(degree) for degree in self.picked_rotation_degrees]
        # self.picked_rotation_degrees = list(range(0, 12, 1))

        with open(self.id_map_addr, 'r') as f:
            self.id_map = json.load(f)

        self.prompt_template_addr = os.path.join('./data/templates.json')
        with open(self.prompt_template_addr) as f:
            self.templates = json.load(f)[config.pretrain_dataset_prompt]

        self.synset_id_map = {}
        for id_dict in self.id_map:
            synset_id = id_dict["synsetId"]
            self.synset_id_map[synset_id] = id_dict

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')

        self.sample_points_num = self.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger='ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger='ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger='ShapeNet-55')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line[len(taxonomy_id) + 1:].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='ShapeNet-55')

        self.permutation = np.arange(self.npoints)

        self.uniform = True
        self.augment = True
        self.use_caption_templates = False
        # =================================================
        # TODO: disable for backbones except for PointNEXT!!!
        self.use_height = config.use_height
        # =================================================

        if self.augment:
            print("using augmented point clouds.")

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc

    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        if self.uniform and self.sample_points_num < data.shape[0]:
            data = farthest_point_sample(data, self.sample_points_num)
        else:
            data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)

        if self.augment:
            # TODO
            data = random_point_dropout(data[None, ...])
            data = random_scale_point_cloud(data)
            data = shift_point_cloud(data)
            data = rotate_perturbation_point_cloud(data)
            data = rotate_point_cloud(data)
            data = data.squeeze()

        if self.use_height:
            self.gravity_dim = 1
            height_array = data[:, self.gravity_dim:self.gravity_dim + 1] - data[:,
                                                                            self.gravity_dim:self.gravity_dim + 1].min()
            data = np.concatenate((data, height_array), axis=1)
            data = torch.from_numpy(data).float()
        else:
            data = torch.from_numpy(data).float()

        captions = self.synset_id_map[sample['taxonomy_id']]['name']
        captions = [caption.strip() for caption in captions.split(',') if caption.strip()]
        
        # missing = set()
        # for caption in captions:   
        #     if caption not in self.classes:
        #         missing.add(caption)
        # print(f"missing:{missing}")
        
        # TODO
        caption = captions[0] #random.choice(captions)
        # 如果 caption 在 synonym_map 里，就替换成标准类别
        if caption in synonym_map:
            caption = synonym_map[caption]

        # 如果 caption 仍然不在 self.classes，就提示一下
        if caption not in self.classes:
            raise KeyError(f"Unknown class name: {caption}")

        label = self.classes[caption]

        captions = []
        tokenized_captions = []
        if self.use_caption_templates:
            for template in self.templates:
                caption = template.format(caption)
                captions.append(caption)
                tokenized_captions.append(self.tokenizer(caption))
        else:
            tokenized_captions.append(self.tokenizer(caption))

        tokenized_captions = torch.stack(tokenized_captions)

        # picked_model_rendered_image_addr = self.rendered_image_addr + '/img/'
        # picked = random.randint(0, 11)
        # picked_image_name = sample['taxonomy_id'] + '/' + sample['model_id'] + '/' + '0' * (3 - len(str(picked))) + str(
        #     picked) + '.png'
        # picked_image_addr = picked_model_rendered_image_addr + picked_image_name


        picked_model_rendered_image_addr = self.rendered_image_addr + '/' +\
                                           sample['taxonomy_id'] + '-' + sample['model_id'] + '/'
        picked_image_name = sample['model_id'] + '_r_' +\
                            str(random.choice(self.picked_rotation_degrees)) +\
                            random.choice(self.picked_image_type) + '.png'
        picked_image_addr = picked_model_rendered_image_addr + picked_image_name
        
        picked_model_rendered_depth_addr = self.rendered_depth_addr + '/' +\
                                           sample['taxonomy_id'] + '-' + sample['model_id'] + '/'
        picked_depth_name = sample['model_id'] + '_r_' +\
                            str(random.choice(self.picked_rotation_degrees)) + '.npy'
        picked_depth_addr = picked_model_rendered_depth_addr + picked_depth_name
        
        depth = np.load(picked_depth_addr)
        
        # print("=========================================")
        # print(f"label:{label},caption:{caption}")
        # print(f"picked_image_addr:{picked_image_addr}")
        # print("=========================================")

        # try:
        #     image = pil_loader(picked_image_addr)
        #     image = self.train_transform(image)
        # except:
        #     raise ValueError("image is corrupted: {}".format(picked_image_addr))
        
        # return sample['taxonomy_id'], sample['model_id'], tokenized_captions, data, image
        # label = self.classes.index(caption)
        # return (image, data, tokenized_captions, depth = np.load(picked_depth_addr)
        print(f"DEBUG: Initial depth shape: {depth.shape}, type: {depth.dtype}")

        try:
            image = pil_loader(picked_image_addr)
        except Exception as e:
            raise ValueError(f"image broken: {picked_image_addr}") from e

        flip = random.random() > 0.5
        angle = random.uniform(-10, 10)
        crop_params = transforms.RandomCrop.get_params(
            image, output_size=(self.input_size, self.input_size))
        depth = torch.from_numpy(depth).float()  # 保持原量纲
        # 先确保为 [1,H,W]，再做几何变换
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)  # [1,H,W]
        image, depth = self.apply_shared_transform(image, depth, flip, angle, crop_params)
        # image, depth = self.apply_shared_transform(
        #     image, TF.to_pil_image(torch.from_numpy(depth).float()),
        #     flip, angle, crop_params)
        print(f"DEBUG: Depth shape after apply_shared_transform: {depth.size}, type: {type(depth)}") 

        image = TF.to_tensor(image)                      # 2. 转 Tensor（图像）
        # 深度已是Tensor且形状为[1,H,W]，无需再调用to_tensor
        if not isinstance(depth, torch.Tensor):
            depth = TF.to_tensor(depth)
        # 确保类型为float32
        # depth = depth.float()
        # 逐样本 min-max 归一化到[0,1]
        d_min = torch.nan_to_num(depth.amin(dim=[1,2], keepdim=True), nan=0.0)
        d_max = torch.nan_to_num(depth.amax(dim=[1,2], keepdim=True), nan=1.0)
        depth = (depth - d_min) / (d_max - d_min + 1e-6)
        # print(f"DEBUG: Depth shape after tensorization: {depth.shape}, dtype: {depth.dtype}")

        return (image, data, tokenized_captions, depth), label
        
    def apply_shared_transform(self, image, depth, flip, angle, crop_params):
        # 图像增强
        image = TF.resize(image, self.input_size, interpolation=TF.InterpolationMode.BICUBIC)
        if isinstance(depth, torch.Tensor):
            # 期望形状 [C,H,W]
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)
            # 只缩放空间维度
            depth = TF.resize(depth, [self.input_size, self.input_size], interpolation=TF.InterpolationMode.NEAREST)
        else:
            depth = TF.resize(depth, self.input_size, interpolation=TF.InterpolationMode.NEAREST)

        i, j, h, w = crop_params
        image = TF.crop(image, i, j, h, w)
        depth = TF.crop(depth, i, j, h, w)

        if flip:
            image = TF.hflip(image)
            depth = TF.hflip(depth)

        image = TF.rotate(image, angle)
        depth = TF.rotate(depth, angle, interpolation=TF.InterpolationMode.NEAREST)

        return image, depth

    def __len__(self):
        return len(self.file_list)

@DATASETS.register_module()
class co3d_img_pcl(data.Dataset):
    '''

    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 8192

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats_co3d, range(len(cats_co3d))))
        self.DATA_DIR = config.DATA_PATH
        self.IMAGE_PATH = config.IMAGE_PATH
        self._load_data()

    def _load_data(self):
        self.paths = []
        self.imgpaths = []
        self.labels = []

        # all_meshes = os.listdir(self.DATA_DIR)
        # all_meshes = [x.replace('.npy', '') for x in all_meshes]

        # filename = '/'.join(self.DATA_DIR.split('/')[:-1]) + '/' + str(self.partition)+'_set_balanced.txt'
        filename = '/'.join(self.DATA_DIR.split('/')[:-1]) + '/' + str(self.partition) + '_set_80.txt'
        with open(filename) as file:
            all_meshes = [line.rstrip() for line in file]

        for mesh in all_meshes:

            images_path = self.IMAGE_PATH + '/' + mesh + '/images'
            images = os.listdir(images_path)
            if len(images) == 0:
                pass
            else:
                self.imgpaths.append([self.IMAGE_PATH + '/' + mesh + '/images' + '/' + i for i in images])

                self.paths.append(self.DATA_DIR + '/' + mesh + '/pointcloud1.npy')

                cat = mesh.split('/')[0]
                self.labels.append(cats_co3d[cat])

    def __getitem__(self, index):
        # point = offread_uniformed(self.paths[index], self.num_points)
        # point = pc_normalize(point)

        point1 = torch.from_numpy(np.load(self.paths[index]))
        point1 = torch.index_select(point1, 1, torch.LongTensor([1, 0, 2]))

        # picked = random.randint(0,9)
        # frames = os.listdir(self.imgpaths[index])
        img = pil_loader(random.choice(self.imgpaths[index]))
        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            return self.transform(img, point1), label
        (timg, tpoint) = self.transform(img, point1)
        # point = torch.from_numpy(point).to(torch.float32)
        return (timg, tpoint), label

    def __len__(self):
        return len(self.labels)

@DATASETS.register_module()
class redwood_img_pcl(data.Dataset):
    '''
        points are randomly sampled from .off file, so the results of this dataset may be better or wrose than our claim results
    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 8192

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats_redwood, range(len(cats_redwood))))
        self.DATA_DIR = config.DATA_PATH
        self.IMAGE_PATH = config.IMAGE_PATH
        self.categories = config.CATEGORIES
        self._load_data()

    def _load_data(self):
        self.paths = []
        self.imgpaths = []
        self.labels = []

        # all_meshes = os.listdir(self.DATA_DIR)
        # all_meshes = [x.replace('.npy', '') for x in all_meshes]

        # filename = '/'.join(self.DATA_DIR.split('/')[:-1]) + '/' + str(self.partition)+'_set_balanced.txt'
        filename = '/'.join(self.DATA_DIR.split('/')[:-1]) + '/' + str(self.partition) + '_set80.txt'
        with open(filename) as file:
            all_meshes = [line.rstrip() for line in file]

        with open(self.categories) as json_file:
            categories_dic = json.load(json_file)

        for cat in self.classes:
            for mesh in categories_dic[cat]:
                if mesh in all_meshes:
                    self.paths.append(self.DATA_DIR+'/'+mesh+'.npy')
                    self.imgpaths.append(self.IMAGE_PATH+'/'+mesh)
                    self.labels.append(cats_redwood[cat])

    def __getitem__(self, index):
        # point = offread_uniformed(self.paths[index], self.num_points)
        # point = pc_normalize(point)

        point1 = torch.from_numpy(np.load(self.paths[index]))
        # point1 = torch.index_select(point1, 1, torch.LongTensor([1, 2, 0]))

        picked = random.randint(0,9)
        frames = os.listdir(self.imgpaths[index])
        img = pil_loader(os.path.join(self.imgpaths[index], frames[picked]))
        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            return self.transform(img, point1), label
        (timg, tpoint) = self.transform(img, point1)
        # point = torch.from_numpy(point).to(torch.float32)
        return (timg, tpoint), label

    def __len__(self):
        return len(self.labels)

@DATASETS.register_module()
class redwood_img_pcl_multiview(data.Dataset):
    '''
        points are randomly sampled from .off file, so the results of this dataset may be better or wrose than our claim results
    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 8192

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats_redwood, range(len(cats_redwood))))
        self.DATA_DIR = config.DATA_PATH
        self.IMAGE_PATH = config.IMAGE_PATH
        self.categories = config.CATEGORIES
        self._load_data()

    def _load_data(self):
        self.paths = []
        self.imgpaths = []
        self.labels = []

        # all_meshes = os.listdir(self.DATA_DIR)
        # all_meshes = [x.replace('.npy', '') for x in all_meshes]

        # filename = '/'.join(self.DATA_DIR.split('/')[:-1]) + '/' + str(self.partition)+'_set_balanced.txt'
        filename = '/'.join(self.DATA_DIR.split('/')[:-1]) + '/' + str(self.partition) + '_set80.txt'
        with open(filename) as file:
            all_meshes = [line.rstrip() for line in file]

        with open(self.categories) as json_file:
            categories_dic = json.load(json_file)

        for cat in self.classes:
            for mesh in categories_dic[cat]:
                if mesh in all_meshes:
                    self.paths.append(self.DATA_DIR+'/'+mesh+'.npy')
                    self.imgpaths.append(self.IMAGE_PATH+'/'+mesh)
                    self.labels.append(cats_redwood[cat])

    def __getitem__(self, index):
        # point = offread_uniformed(self.paths[index], self.num_points)
        # point = pc_normalize(point)

        point1 = torch.from_numpy(np.load(self.paths[index]))
        # point1 = torch.index_select(point1, 1, torch.LongTensor([1, 2, 0]))

        imgs = []
        frames = os.listdir(self.imgpaths[index])
        for i in range(10):
            imgs.append(pil_loader(os.path.join(self.imgpaths[index], frames[i])))

        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            imgstw = []
            imgsts = []
            for i in range(10):
                (imgtw, imgts, mask, pc_weak, pc_strong, pc_mask) = self.transform(imgs[i], point1)
                imgstw.append(imgtw)
                imgsts.append(imgts)
            return (torch.stack(imgstw), torch.stack(imgsts), mask, pc_weak, pc_strong, pc_mask), label
        imgst = []
        for i in range(10):
            (imgt, tpoint1) = self.transform(imgs[i], point1)
            imgst.append(imgt)
        return (torch.stack(imgst), tpoint1), label
    def __len__(self):
        return len(self.labels)

@DATASETS.register_module()
class pix3d_img_pcl(data.Dataset):
    '''

    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 8192

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats_pix3d, range(len(cats_pix3d))))
        self.DATA_DIR = config.DATA_PATH
        self.IMAGE_PATH = config.IMAGE_PATH
        self._load_data()

    def _load_data(self):
        self.paths = []
        self.imgpaths = []
        self.labels = []

        # all_meshes = os.listdir(self.DATA_DIR)
        # all_meshes = [x.replace('.npy', '') for x in all_meshes]

        # filename = '/'.join(self.DATA_DIR.split('/')[:-1]) + '/' + str(self.partition)+'_set_balanced.txt'
        filename = '/'.join(self.DATA_DIR.split('/')[:-1]) + '/' + str(self.partition) + '_set80.txt'
        with open(filename) as file:
            all_meshes = [line.rstrip() for line in file]

        for mesh in all_meshes:
            self.paths.append(self.DATA_DIR + '/' + mesh + '/model.npy')
            images_path = self.IMAGE_PATH + '/' + mesh
            images = os.listdir(images_path)
            images.remove('model.npy')
            self.imgpaths.append([self.IMAGE_PATH + '/' + mesh +'/'+ i for i in images])
            cat = mesh.split('/')[0]
            self.labels.append(cats_pix3d[cat])

    def __getitem__(self, index):
        # point = offread_uniformed(self.paths[index], self.num_points)
        # point = pc_normalize(point)

        point1 = torch.from_numpy(np.load(self.paths[index]))
        # point1 = torch.index_select(point1, 1, torch.LongTensor([1, 2, 0]))

        # picked = random.randint(0,9)
        # frames = os.listdir(self.imgpaths[index])
        img = pil_loader(random.choice(self.imgpaths[index]))
        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            return self.transform(img, point1), label
        (timg, tpoint) = self.transform(img, point1)
        # point = torch.from_numpy(point).to(torch.float32)
        return (timg, tpoint), label

    def __len__(self):
        return len(self.labels)

@DATASETS.register_module()
class pix3d_img_pcl_multiview(data.Dataset):
    '''

    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 8192

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats_pix3d, range(len(cats_pix3d))))
        self.DATA_DIR = config.DATA_PATH
        self.IMAGE_PATH = config.IMAGE_PATH
        self._load_data()

    def _load_data(self):
        self.paths = []
        self.imgpaths = []
        self.labels = []

        # all_meshes = os.listdir(self.DATA_DIR)
        # all_meshes = [x.replace('.npy', '') for x in all_meshes]

        # filename = '/'.join(self.DATA_DIR.split('/')[:-1]) + '/' + str(self.partition)+'_set_balanced.txt'
        filename = '/'.join(self.DATA_DIR.split('/')[:-1]) + '/' + str(self.partition) + '_set80.txt'
        with open(filename) as file:
            all_meshes = [line.rstrip() for line in file]

        for mesh in all_meshes:
            self.paths.append(self.DATA_DIR + '/' + mesh + '/model.npy')
            images_path = self.IMAGE_PATH + '/' + mesh
            images = os.listdir(images_path)
            images.remove('model.npy')
            self.imgpaths.append([self.IMAGE_PATH + '/' + mesh +'/'+ i for i in images])
            cat = mesh.split('/')[0]
            self.labels.append(cats_pix3d[cat])

    def __getitem__(self, index):
        point1 = torch.from_numpy(np.load(self.paths[index]))

        label = self.labels[index]

        imgs = []
        shuffled_frames = self.imgpaths[index]
        random.shuffle(shuffled_frames)
        imnum= min(10, len(shuffled_frames))
        for i in range(imnum):
            imgs.append(pil_loader(shuffled_frames[i]))

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            imgstw = []
            imgsts = []
            for i in range(imnum):
                (imgtw, imgts, mask, pc_weak, pc_strong, pc_mask) = self.transform(imgs[i], point1)
                imgstw.append(imgtw)
                imgsts.append(imgts)
            return (torch.stack(imgstw), torch.stack(imgsts), mask, pc_weak, pc_strong, pc_mask), label

        imgst = []
        for i in range(imnum):
            (imgt, tpoint1) = self.transform(imgs[i], point1)
            imgst.append(imgt)
        return (torch.stack(imgst), tpoint1), label

    def __len__(self):
        return len(self.labels)

@DATASETS.register_module()
class ModelNet10_img_pcl(data.Dataset):
    '''
        points are randomly sampled from .off file, so the results of this dataset may be better or wrose than our claim results
    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 8192

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats_modelnet10, range(len(cats_modelnet10))))
        self.DATA_DIR = config.DATA_PATH
        # self.DATA_DIR2 = config.DATA_PATH2
        self.IMAGE_PATH = config.IMAGE_PATH

        self._load_data()
        self.loaded = True

    def _load_data(self):
        self.paths = []
        self.imgpaths = []
        self.labels = []
        # for cat in os.listdir(self.DATA_DIR):
        for cat in self.classes:
            cat_path = os.path.join(self.DATA_DIR, cat, self.partition)
            cat_path_img = os.path.join(self.IMAGE_PATH, self.partition, cat)
            for case in os.listdir(cat_path):
                if case.endswith('.off'):
                    self.paths.append(os.path.join(cat_path, case))
                    self.imgpaths.append(cat_path_img + '/' + case[:-4])
                    self.labels.append(cats_modelnet10[cat])

    def _few(self):
        num_dict = {i: 0 for i in range(40)}
        few_paths = []
        few_labels = []
        random_list = [k for k in range(len(self.labels))]
        random.shuffle(random_list)
        for i in random_list:
            label = self.labels[i]
            if num_dict[label] == self.few_num:
                continue
            few_paths.append(self.paths[i])
            few_labels.append(self.labels[i])
            num_dict[label] += 1
        return few_paths, few_labels

    def __getitem__(self, index):
        # point = offread_uniformed(self.paths[index], self.num_points)
        # point = pc_normalize(point)

        point1 = torch.from_numpy(offread_uniformed(self.paths[index], self.num_points))
        point1 = torch.index_select(point1, 1, torch.LongTensor([1, 2, 0]))

        picked = random.randint(1, 12)
        picked = '0' * (3 - len(str(picked))) + str(picked)
        img = pil_loader(self.imgpaths[index] + '_' + picked + '.png')
        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            return self.transform(img, point1), label
        (timg, tpoint) = self.transform(img, point1)
        # point = torch.from_numpy(point).to(torch.float32)
        return (timg, tpoint), label

    def __len__(self):
        return len(self.labels)
    
# TODO: 4modal
@DATASETS.register_module()
class ModelNet40_4modal(data.Dataset):
    '''
    ModelNet40 dataset with RGB images, point clouds and depth images
    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 8192

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats, range(len(cats))))
        self.DATA_DIR = config.DATA_PATH
        self.IMAGE_PATH = config.IMAGE_PATH
        self.DEPTH_PATH = config.DEPTH_PATH

        try:
            self.nviews = config.NVIEWS
        except:
            self.nviews = 12

        print("Selected number of views ", self.nviews)

        self._load_data()

    def _load_data(self):
        self.paths = []
        self.imgpaths = []
        self.depthpaths = []
        self.labels = []
        for cat in os.listdir(self.DATA_DIR):
            cat_path = os.path.join(self.DATA_DIR, cat, self.partition)
            cat_path_img = os.path.join(self.IMAGE_PATH, self.partition, cat)
            # cat_path_depth = os.path.join(self.DEPTH_PATH, self.partition, cat)
            cat_path_depth = os.path.join(self.DEPTH_PATH, cat, self.partition)
            for case in os.listdir(cat_path):
                if case.endswith('.off'):
                    self.paths.append(os.path.join(cat_path, case))
                    self.imgpaths.append(cat_path_img + '/' + case[:-4])
                    self.depthpaths.append(cat_path_depth + '/' + case[:-4])
                    self.labels.append(cats[cat])

    def _few(self):
        num_dict = {i: 0 for i in range(40)}
        few_paths = []
        few_labels = []
        random_list = [k for k in range(len(self.labels))]
        random.shuffle(random_list)
        for i in random_list:
            label = self.labels[i]
            if num_dict[label] == self.few_num:
                continue
            few_paths.append(self.paths[i])
            few_labels.append(self.labels[i])
            num_dict[label] += 1
        return few_paths, few_labels

    def __getitem__(self, index):
        # Load point cloud
        point1 = torch.from_numpy(offread_uniformed(self.paths[index], self.num_points))
        point1 = torch.index_select(point1, 1, torch.LongTensor([1, 2, 0]))

        # Load RGB image and depth image
        picked = random.randint(1, self.nviews)
        image_picked = '0' * (3 - len(str(picked))) + str(picked)
        img = pil_loader(self.imgpaths[index] + '_' + image_picked + '.png')
        depth_picked = '0' * (2 - len(str(picked))) + str(picked)
        depth = np.load(self.depthpaths[index] + '_view' + depth_picked + '.npy')
        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            point1 = point1[pt_idxs]
            
            # 数据增强 - 训练模式返回8个值
            img_weak, img_strong, mask, depth_weak, depth_strong, pcl_weak, pcl_strong, flag = self.transform(img, depth, point1)
            return (img_weak, img_strong, mask, pcl_weak, pcl_strong, depth_weak, depth_strong), label
            
        # 验证/测试阶段 - 测试模式返回3个值
        img_transformed, depth_transformed, pcl_transformed = self.transform(img, depth, point1)
        return (img_transformed, depth_transformed, pcl_transformed), label

    def __len__(self):
        return len(self.labels)

@DATASETS.register_module()
class ModelNet40_img_pcl(data.Dataset):
    '''
        points are randomly sampled from .off file, so the results of this dataset may be better or wrose than our claim results
    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 8192

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats, range(len(cats))))
        self.DATA_DIR = config.DATA_PATH
        # self.DATA_DIR2 = config.DATA_PATH2
        self.IMAGE_PATH = config.IMAGE_PATH

        try:
            self.nviews = config.NVIEWS
        except:
            self.nviews = 12

        print("Selected number of views ", self.nviews)

        self._load_data()

    def _load_data(self):
        self.paths = []
        self.imgpaths = []
        self.labels = []
        for cat in os.listdir(self.DATA_DIR):
            cat_path = os.path.join(self.DATA_DIR, cat, self.partition)
            cat_path_img = os.path.join(self.IMAGE_PATH, self.partition, cat)
            for case in os.listdir(cat_path):
                if case.endswith('.off'):
                    self.paths.append(os.path.join(cat_path, case))
                    self.imgpaths.append(cat_path_img + '/' + case[:-4])
                    self.labels.append(cats[cat])

    def _few(self):
        num_dict = {i: 0 for i in range(40)}
        few_paths = []
        few_labels = []
        random_list = [k for k in range(len(self.labels))]
        random.shuffle(random_list)
        for i in random_list:
            label = self.labels[i]
            if num_dict[label] == self.few_num:
                continue
            few_paths.append(self.paths[i])
            few_labels.append(self.labels[i])
            num_dict[label] += 1
        return few_paths, few_labels

    def __getitem__(self, index):
        # point = offread_uniformed(self.paths[index], self.num_points)
        # point = pc_normalize(point)

        point1 = torch.from_numpy(offread_uniformed(self.paths[index], self.num_points))
        point1 = torch.index_select(point1, 1, torch.LongTensor([1, 2, 0]))

        picked = random.randint(1, self.nviews)
        picked = '0' * (3 - len(str(picked))) + str(picked)
        img = pil_loader(self.imgpaths[index] + '_' + picked + '.png')
        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            return self.transform(img, point1), label
        (timg, tpoint) = self.transform(img, point1)
        # point = torch.from_numpy(point).to(torch.float32)
        return (timg, tpoint), label

    def __len__(self):
        return len(self.labels)

@DATASETS.register_module()
class ModelNet40_img_pcl_multiview(data.Dataset):
    '''
        points are randomly sampled from .off file, so the results of this dataset may be better or wrose than our claim results
    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 8192

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats, range(len(cats))))
        self.DATA_DIR = config.DATA_PATH
        # self.DATA_DIR2 = config.DATA_PATH2
        self.IMAGE_PATH = config.IMAGE_PATH

        self._load_data()

    def _load_data(self):
        self.paths = []
        self.imgpaths = []
        self.labels = []
        for cat in os.listdir(self.DATA_DIR):
            cat_path = os.path.join(self.DATA_DIR, cat, self.partition)
            # cat_path2 = os.path.join(self.DATA_DIR2, cat, self.partition)
            cat_path_img = os.path.join(self.IMAGE_PATH, self.partition, cat)
            for case in os.listdir(cat_path):
                if case.endswith('.off'):
                    self.paths.append(os.path.join(cat_path, case))
                    self.imgpaths.append(cat_path_img + '/' + case[:-4])
                    self.labels.append(cats[cat])

        # print(self.paths[:10])
        # print(self.labels[:10])

    def _few(self):
        num_dict = {i: 0 for i in range(40)}
        few_paths = []
        few_labels = []
        random_list = [k for k in range(len(self.labels))]
        random.shuffle(random_list)
        for i in random_list:
            label = self.labels[i]
            if num_dict[label] == self.few_num:
                continue
            few_paths.append(self.paths[i])
            few_labels.append(self.labels[i])
            num_dict[label] += 1
        return few_paths, few_labels

    def __getitem__(self, index):
        # point = offread_uniformed(self.paths[index], self.num_points)
        # point = pc_normalize(point)

        point1 = torch.from_numpy(offread_uniformed(self.paths[index], self.num_points))
        point1 = torch.index_select(point1, 1, torch.LongTensor([1, 2, 0]))

        # picked = random.randint(1, 12)
        imgs=[]
        for i in range(1,11):
            picked = '0' * (3 - len(str(i))) + str(i)
            imgs.append(pil_loader(self.imgpaths[index] + '_' + picked + '.png'))
        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            imgstw = []
            imgsts = []
            for i in range(10):
                (imgtw, imgts, mask, pc_weak, pc_strong, pc_mask)= self.transform(imgs[i], point1)
                imgstw.append(imgtw)
                imgsts.append(imgts)
            return (torch.stack(imgstw), torch.stack(imgsts), mask, pc_weak, pc_strong, pc_mask), label

        imgst = []
        for i in range(10):
            (imgt, tpoint1) = self.transform(imgs[i], point1)
            imgst.append(imgt)
        return (torch.stack(imgst), tpoint1), label

    def __len__(self):
        return len(self.labels)

@DATASETS.register_module()
class ModelNet40_img_pcl_h5(data.Dataset):
    '''
        points are randomly sampled from .off file, so the results of this dataset may be better or wrose than our claim results
    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 2048

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats, range(len(cats))))
        self.DATA_DIR = config.DATA_PATH
        if self.partition == 'test':
            data_path = os.path.join(self.DATA_DIR, 'test_files.txt')
        else:
            data_path = os.path.join(self.DATA_DIR, 'train_files.txt')
        self._load_data(data_path)

    def _load_data(self, data_path):
        self.points = []
        self.imgpaths = []
        self.labels = []
        with open(data_path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(h5_name.strip(), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                self.points.extend(data)
                self.labels.extend(label)
                image_path = h5_name.strip().split('/')[-1][:-3].replace('ply', 'image')
                image_full_path = os.path.join(os.path.join(*h5_name.strip().split('/')[:-1]), image_path)
                for i in range(len(data)):
                    self.imgpaths.append('/'+image_full_path+'/'+str(i))

    def _few(self):
        num_dict = {i: 0 for i in range(40)}
        few_paths = []
        few_labels = []
        random_list = [k for k in range(len(self.labels))]
        random.shuffle(random_list)
        for i in random_list:
            label = self.labels[i]
            if num_dict[label] == self.few_num:
                continue
            few_paths.append(self.paths[i])
            few_labels.append(self.labels[i])
            num_dict[label] += 1
        return few_paths, few_labels

    def __getitem__(self, index):
        point1 = torch.from_numpy(self.points[index])

        picked = random.randint(0, 9)
        picked = str(picked)
        img = pil_loader(self.imgpaths[index] + '/' + picked + '.png')
        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            return self.transform(img, point1), label
        (timg, tpoint) = self.transform(img, point1)
        return (timg, tpoint), label

    def __len__(self):
        return len(self.labels)

@DATASETS.register_module()
class ModelNet40_img_pcl_h5_multiview(data.Dataset):
    '''
        points are randomly sampled from .off file, so the results of this dataset may be better or wrose than our claim results
    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 2048

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats, range(len(cats))))
        self.DATA_DIR = config.DATA_PATH
        if self.partition == 'test':
            data_path = os.path.join(self.DATA_DIR, 'test_files.txt')
        else:
            data_path = os.path.join(self.DATA_DIR, 'train_files.txt')
        self._load_data(data_path)

    def _load_data(self, data_path):
        self.points = []
        self.imgpaths = []
        self.labels = []
        with open(data_path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(h5_name.strip(), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                self.points.extend(data)
                self.labels.extend(label)
                image_path = h5_name.strip().split('/')[-1][:-3].replace('ply', 'image')
                image_full_path = os.path.join(os.path.join(*h5_name.strip().split('/')[:-1]), image_path)
                for i in range(len(data)):
                    self.imgpaths.append('/'+image_full_path+'/'+str(i))

    def _few(self):
        num_dict = {i: 0 for i in range(40)}
        few_paths = []
        few_labels = []
        random_list = [k for k in range(len(self.labels))]
        random.shuffle(random_list)
        for i in random_list:
            label = self.labels[i]
            if num_dict[label] == self.few_num:
                continue
            few_paths.append(self.paths[i])
            few_labels.append(self.labels[i])
            num_dict[label] += 1
        return few_paths, few_labels

    def __getitem__(self, index):
        point1 = torch.from_numpy(self.points[index])

        imgs = []
        for i in range(0, 10):
            picked = str(i)
            imgs.append(pil_loader(self.imgpaths[index] + '/' + picked + '.png'))
        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            imgstw = []
            imgsts = []
            for i in range(10):
                (imgtw, imgts, mask, pc_weak, pc_strong, pc_mask) = self.transform(imgs[i], point1)
                imgstw.append(imgtw)
                imgsts.append(imgts)
            return (torch.stack(imgstw), torch.stack(imgsts), mask, pc_weak, pc_strong, pc_mask), label

        imgst = []
        for i in range(10):
            (imgt, tpoint1) = self.transform(imgs[i], point1)
            imgst.append(imgt)
        return (torch.stack(imgst), tpoint1), label

    def __len__(self):
        return len(self.labels)

@DATASETS.register_module()
class scanobjectnn_img_pcl_h5(data.Dataset):
    '''
        points are randomly sampled from .off file, so the results of this dataset may be better or wrose than our claim results
    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 2048

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats_scanobject, range(len(cats_scanobject))))
        self.DATA_DIR = config.DATA_PATH
        if self.partition == 'test':
            data_path = os.path.join(self.DATA_DIR, 'test_files.txt')
        else:
            data_path = os.path.join(self.DATA_DIR, 'train_files.txt')
        self._load_data(data_path)

    def _load_data(self, data_path):
        self.points = []
        self.imgpaths = []
        self.labels = []
        with open(data_path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(h5_name.strip(), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                self.points.extend(data)
                self.labels.extend(label)
                image_path = h5_name.strip().split('/')[-1][:-3]+'image'
                image_full_path = os.path.join(os.path.join(*h5_name.strip().split('/')[:-1]), image_path)
                for i in range(len(data)):
                    self.imgpaths.append('/'+image_full_path+'/'+str(i))

    def _few(self):
        num_dict = {i: 0 for i in range(40)}
        few_paths = []
        few_labels = []
        random_list = [k for k in range(len(self.labels))]
        random.shuffle(random_list)
        for i in random_list:
            label = self.labels[i]
            if num_dict[label] == self.few_num:
                continue
            few_paths.append(self.paths[i])
            few_labels.append(self.labels[i])
            num_dict[label] += 1
        return few_paths, few_labels

    def __getitem__(self, index):
        point1 = torch.from_numpy(self.points[index])
        # point1 = torch.index_select(point1, 1, torch.LongTensor([1, 0, 2]))

        picked = random.randint(0, 9)
        picked = str(picked)
        img = pil_loader(self.imgpaths[index] + '/' + picked + '.png')
        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            return self.transform(img, point1), label
        (timg, tpoint) = self.transform(img, point1)
        return (timg, tpoint), label

    def __len__(self):
        return len(self.labels)

@DATASETS.register_module()
class scanobjectnn_img_pcl_h5_multiview(data.Dataset):
    '''
        points are randomly sampled from .off file, so the results of this dataset may be better or wrose than our claim results
    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 2048

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats_scanobject, range(len(cats_scanobject))))
        self.DATA_DIR = config.DATA_PATH
        if self.partition == 'test':
            data_path = os.path.join(self.DATA_DIR, 'test_files.txt')
        else:
            data_path = os.path.join(self.DATA_DIR, 'train_files.txt')
        self._load_data(data_path)

    def _load_data(self, data_path):
        self.points = []
        self.imgpaths = []
        self.labels = []
        with open(data_path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(h5_name.strip(), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                self.points.extend(data)
                self.labels.extend(label)
                image_path = h5_name.strip().split('/')[-1][:-3]+'image'
                image_full_path = os.path.join(os.path.join(*h5_name.strip().split('/')[:-1]), image_path)
                for i in range(len(data)):
                    self.imgpaths.append('/'+image_full_path+'/'+str(i))

    def _few(self):
        num_dict = {i: 0 for i in range(40)}
        few_paths = []
        few_labels = []
        random_list = [k for k in range(len(self.labels))]
        random.shuffle(random_list)
        for i in random_list:
            label = self.labels[i]
            if num_dict[label] == self.few_num:
                continue
            few_paths.append(self.paths[i])
            few_labels.append(self.labels[i])
            num_dict[label] += 1
        return few_paths, few_labels

    def __getitem__(self, index):
        point1 = torch.from_numpy(self.points[index])
        # point1 = torch.index_select(point1, 1, torch.LongTensor([1, 0, 2]))

        imgs = []
        for i in range(0, 10):
            picked = str(i)
            imgs.append(pil_loader(self.imgpaths[index] + '/' + picked + '.png'))
        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            imgstw = []
            imgsts = []
            for i in range(10):
                (imgtw, imgts, mask, pc_weak, pc_strong, pc_mask) = self.transform(imgs[i], point1)
                imgstw.append(imgtw)
                imgsts.append(imgts)
            return (torch.stack(imgstw), torch.stack(imgsts), mask, pc_weak, pc_strong, pc_mask), label

        imgst = []
        for i in range(10):
            (imgt, tpoint1) = self.transform(imgs[i], point1)
            imgst.append(imgt)
        return (torch.stack(imgst), tpoint1), label

    def __len__(self):
        return len(self.labels)

@DATASETS.register_module()
class scanobjectnn_withbg_img_pcl_h5(data.Dataset):
    '''
        points are randomly sampled from .off file, so the results of this dataset may be better or wrose than our claim results
    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 2048

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats_scanobject, range(len(cats_scanobject))))
        self.DATA_DIR = config.DATA_PATH
        if self.partition == 'test':
            data_path = os.path.join(self.DATA_DIR, 'test_files_withbg.txt')
        else:
            data_path = os.path.join(self.DATA_DIR, 'train_files_withbg.txt')
        self._load_data(data_path)
        self.loaded = True

    def _load_data(self, data_path):
        self.points = []
        self.imgpaths = []
        self.labels = []
        with open(data_path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(h5_name.strip(), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                self.points.extend(data)
                self.labels.extend(label)
                image_path = h5_name.strip().split('/')[-1][:-3]+'image'
                image_full_path = os.path.join(os.path.join(*h5_name.strip().split('/')[:-1]), image_path)
                for i in range(len(data)):
                    self.imgpaths.append('/'+image_full_path+'/'+str(i))

    def _few(self):
        num_dict = {i: 0 for i in range(40)}
        few_paths = []
        few_labels = []
        random_list = [k for k in range(len(self.labels))]
        random.shuffle(random_list)
        for i in random_list:
            label = self.labels[i]
            if num_dict[label] == self.few_num:
                continue
            few_paths.append(self.paths[i])
            few_labels.append(self.labels[i])
            num_dict[label] += 1
        return few_paths, few_labels

    def __getitem__(self, index):
        point1 = torch.from_numpy(self.points[index])
        # point1 = torch.index_select(point1, 1, torch.LongTensor([1, 0, 2]))

        picked = random.randint(0, 9)
        picked = str(picked)
        img = pil_loader(self.imgpaths[index] + '/' + picked + '.png')
        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            return self.transform(img, point1), label
        (timg, tpoint) = self.transform(img, point1)
        return (timg, tpoint), label

    def __len__(self):
        return len(self.labels)

@DATASETS.register_module()
class scanobjectnn_withbg_img_pcl_h5_multiview(data.Dataset):
    '''
        points are randomly sampled from .off file, so the results of this dataset may be better or wrose than our claim results
    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 2048

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats_scanobject, range(len(cats_scanobject))))
        self.DATA_DIR = config.DATA_PATH
        if self.partition == 'test':
            data_path = os.path.join(self.DATA_DIR, 'test_files_withbg.txt')
        else:
            data_path = os.path.join(self.DATA_DIR, 'train_files_withbg.txt')
        self._load_data(data_path)

    def _load_data(self, data_path):
        self.points = []
        self.imgpaths = []
        self.labels = []
        with open(data_path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(h5_name.strip(), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                self.points.extend(data)
                self.labels.extend(label)
                image_path = h5_name.strip().split('/')[-1][:-3]+'image'
                image_full_path = os.path.join(os.path.join(*h5_name.strip().split('/')[:-1]), image_path)
                for i in range(len(data)):
                    self.imgpaths.append('/'+image_full_path+'/'+str(i))

    def _few(self):
        num_dict = {i: 0 for i in range(40)}
        few_paths = []
        few_labels = []
        random_list = [k for k in range(len(self.labels))]
        random.shuffle(random_list)
        for i in random_list:
            label = self.labels[i]
            if num_dict[label] == self.few_num:
                continue
            few_paths.append(self.paths[i])
            few_labels.append(self.labels[i])
            num_dict[label] += 1
        return few_paths, few_labels

    def __getitem__(self, index):
        point1 = torch.from_numpy(self.points[index])
        # point1 = torch.index_select(point1, 1, torch.LongTensor([1, 0, 2]))

        imgs = []
        for i in range(0, 10):
            picked = str(i)
            imgs.append(pil_loader(self.imgpaths[index] + '/' + picked + '.png'))
        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            imgstw = []
            imgsts = []
            for i in range(10):
                (imgtw, imgts, mask, pc_weak, pc_strong, pc_mask) = self.transform(imgs[i], point1)
                imgstw.append(imgtw)
                imgsts.append(imgts)
            return (torch.stack(imgstw), torch.stack(imgsts), mask, pc_weak, pc_strong, pc_mask), label

        imgst = []
        for i in range(10):
            (imgt, tpoint1) = self.transform(imgs[i], point1)
            imgst.append(imgt)
        return (torch.stack(imgst), tpoint1), label

    def __len__(self):
        return len(self.labels)

@DATASETS.register_module()
class scanobjectnn_hardest_img_pcl_h5(data.Dataset):
    '''
        points are randomly sampled from .off file, so the results of this dataset may be better or wrose than our claim results
    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 2048

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats_scanobject, range(len(cats_scanobject))))
        self.DATA_DIR = config.DATA_PATH
        if self.partition == 'test':
            data_path = os.path.join(self.DATA_DIR, 'test_files_hardest.txt')
        else:
            data_path = os.path.join(self.DATA_DIR, 'train_files_hardest.txt')
        self._load_data(data_path)
        self.loaded = True

    def _load_data(self, data_path):
        self.points = []
        self.imgpaths = []
        self.labels = []
        with open(data_path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(h5_name.strip(), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                self.points.extend(data)
                self.labels.extend(label)
                image_path = h5_name.strip().split('/')[-1][:-3]+'image'
                image_full_path = os.path.join(os.path.join(*h5_name.strip().split('/')[:-1]), image_path)
                for i in range(len(data)):
                    self.imgpaths.append('/'+image_full_path+'/'+str(i))

    def _few(self):
        num_dict = {i: 0 for i in range(40)}
        few_paths = []
        few_labels = []
        random_list = [k for k in range(len(self.labels))]
        random.shuffle(random_list)
        for i in random_list:
            label = self.labels[i]
            if num_dict[label] == self.few_num:
                continue
            few_paths.append(self.paths[i])
            few_labels.append(self.labels[i])
            num_dict[label] += 1
        return few_paths, few_labels

    def __getitem__(self, index):
        point1 = torch.from_numpy(self.points[index])
        # point1 = torch.index_select(point1, 1, torch.LongTensor([1, 0, 2]))

        picked = random.randint(0, 9)
        picked = str(picked)
        img = pil_loader(self.imgpaths[index] + '/' + picked + '.png')
        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            return self.transform(img, point1), label
        (timg, tpoint) = self.transform(img, point1)
        return (timg, tpoint), label

    def __len__(self):
        return len(self.labels)

@DATASETS.register_module()
class scanobjectnn_hardest_img_pcl_h5_multiview(data.Dataset):
    '''
        points are randomly sampled from .off file, so the results of this dataset may be better or wrose than our claim results
    '''
    def __init__(self, config):
        super().__init__()
        self.partition = config.subset
        assert self.partition in ('test', 'train', 'val')
        if self.partition == 'val':
            self.partition = 'test'
        self.few_num = 0
        self.num_points = 2048

        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()
        self.transform = build_transform(self.partition, config)
        self.classes = dict(zip(cats_scanobject, range(len(cats_scanobject))))
        self.DATA_DIR = config.DATA_PATH
        if self.partition == 'test':
            data_path = os.path.join(self.DATA_DIR, 'test_files_hardest.txt')
        else:
            data_path = os.path.join(self.DATA_DIR, 'train_files_hardest.txt')
        self._load_data(data_path)

    def _load_data(self, data_path):
        self.points = []
        self.imgpaths = []
        self.labels = []
        with open(data_path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(h5_name.strip(), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                self.points.extend(data)
                self.labels.extend(label)
                image_path = h5_name.strip().split('/')[-1][:-3]+'image'
                image_full_path = os.path.join(os.path.join(*h5_name.strip().split('/')[:-1]), image_path)
                for i in range(len(data)):
                    self.imgpaths.append('/'+image_full_path+'/'+str(i))

    def _few(self):
        num_dict = {i: 0 for i in range(40)}
        few_paths = []
        few_labels = []
        random_list = [k for k in range(len(self.labels))]
        random.shuffle(random_list)
        for i in random_list:
            label = self.labels[i]
            if num_dict[label] == self.few_num:
                continue
            few_paths.append(self.paths[i])
            few_labels.append(self.labels[i])
            num_dict[label] += 1
        return few_paths, few_labels

    def __getitem__(self, index):
        point1 = torch.from_numpy(self.points[index])
        # point1 = torch.index_select(point1, 1, torch.LongTensor([1, 0, 2]))

        imgs = []
        for i in range(0, 10):
            picked = str(i)
            imgs.append(pil_loader(self.imgpaths[index] + '/' + picked + '.png'))
        label = self.labels[index]

        if self.partition == 'train':
            pt_idxs = np.arange(point1.shape[0])
            np.random.shuffle(pt_idxs)
            imgstw = []
            imgsts = []
            for i in range(10):
                (imgtw, imgts, mask, pc_weak, pc_strong, pc_mask) = self.transform(imgs[i], point1)
                imgstw.append(imgtw)
                imgsts.append(imgts)
            return (torch.stack(imgstw), torch.stack(imgsts), mask, pc_weak, pc_strong, pc_mask), label

        imgst = []
        for i in range(10):
            (imgt, tpoint1) = self.transform(imgs[i], point1)
            imgst.append(imgt)
        return (torch.stack(imgst), tpoint1), label

    def __len__(self):
        return len(self.labels)


class MaskGenerator:
    def __init__(self, input_size, mask_patch_size, model_patch_size, mask_ratio):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size ** 2

        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


# class DataAugmentation:
#     def __init__(self, weak_transform, strong_transform, weak_pcltransform, strong_pcltransform, depth_transform_weak, depth_transform_strong, args, config = None):
#         if weak_pcltransform:
#             self.transforms = [weak_transform, strong_transform, weak_pcltransform, strong_pcltransform, depth_transform_weak, depth_transform_strong]
#             self.istrain = True

#             self.mask_generator = MaskGenerator(
#                 input_size=config.input_size,
#                 mask_patch_size=config.mask_patch_size,
#                 model_patch_size=config.model_patch_size,
#                 mask_ratio=config.mask_ratio,
#             )
            
#             # # 深度图增强 - 弱增强
#             # self.depth_transform_weak = transforms.Compose([
#             #     transforms.Resize(args.input_size, interpolation=T.InterpolationMode.NEAREST),
#             #     transforms.RandomCrop(args.input_size)
#             # ])
            
#             # # 深度图增强 - 强增强
#             # self.depth_transform_strong = transforms.Compose([
#             #     transforms.Resize(args.input_size, interpolation=T.InterpolationMode.NEAREST),
#             #     transforms.RandomCrop(args.input_size),
#             #     transforms.RandomHorizontalFlip(p=0.5),
#             #     transforms.RandomRotation(10),
#             #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
#             # ])

#         else:
#             image_transform = weak_transform
#             pcl_transform = strong_transform
#             self.transforms = [image_transform, pcl_transform]
#             self.istrain = False
            
#             # 测试时的深度图处理
#             self.depth_transform = transforms.Compose([
#                 transforms.Resize(args.input_size, interpolation=T.InterpolationMode.NEAREST),
#                 transforms.CenterCrop(args.input_size)
#             ])

#     def __call__(self, x, depth, y):
#         if self.istrain:
#             # 图像增强
#             images_weak = self.transforms[0](x)
#             images_strong = self.transforms[1](x)
            
#             # 点云增强
#             pcl_weak = self.transforms[2](y)
#             pcl_strong = self.transforms[3](y)
            
#             # 深度图增强
#             depth = torch.from_numpy(depth).float()  # 确保深度图是tensor
#             depth = depth.unsqueeze(0)  # 添加通道维度 [H, W] -> [1, H, W]
#             depth_weak = self.depth_transform_weak(depth)
#             depth_strong = self.depth_transform_strong(depth)
#             # 保持原有的返回格式
#             mask = self.mask_generator()
#             return images_weak, images_strong, mask, depth_weak, depth_strong, pcl_weak, pcl_strong, 0

#         else:
#             # 测试阶段
#             images = self.transforms[0](x)
#             pcl = self.transforms[1](y)
            
#             # 深度图处理
#             depth = torch.from_numpy(depth).float()
#             depth = depth.unsqueeze(0)
#             depth = self.depth_transform(depth)
            
#             return images, depth, pcl


def build_transform(is_train, config):
    args = config.args

    if is_train == 'train':
        # === 图像归一化器 ===
        normalize = transforms.Normalize(
            mean=torch.tensor(args.image_mean),
            std=torch.tensor(args.image_std)
        )

        # === 点云增强（强、弱）===
        # 要求这些类支持 angle / flip 参数（前面你已改了 PointcloudRotate）
        strong_pcltransform = pcl_transforms.Compose([
            pcl_transforms.PointcloudNormalize(),
            pcl_transforms.PointcloudScale(lo=0.5, hi=2, p=1),
            pcl_transforms.PointcloudRotate(),
            pcl_transforms.PointcloudTranslate(0.5, p=1),
            pcl_transforms.PointcloudJitter(p=1),
            pcl_transforms.PointcloudRandomInputDropout(p=1),
            pcl_transforms.PointcloudRandomCrop(x_min=9, x_max=1.1, ar_min=0.9, ar_max=1.1),
            pcl_transforms.PointcloudUpSampling(8192)
        ])

        weak_pcltransform = pcl_transforms.Compose([
            pcl_transforms.PointcloudNormalize(),
            pcl_transforms.PointcloudScale(lo=0.9, hi=1.1, p=1),
            pcl_transforms.PointcloudRotatePerturbation(),  # 也需支持 angle
        ])

        # === 创建 DataAugmentation ===
        transform = DataAugmentation(
            weak_transform=None,                     # 图像/深度图增强已整合在内部逻辑
            strong_transform=None,
            weak_pcltransform=weak_pcltransform,
            strong_pcltransform=strong_pcltransform,
            args=args,
            config=config
        )
        return transform

    else:
        # === 测试模式下：图像处理 ===
        transform = transforms.Compose([
            transforms.Resize(args.input_size, interpolation=T.InterpolationMode.BICUBIC),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(args.image_mean),
                std=torch.tensor(args.image_std)
            )
        ])

        # === 测试模式下：点云处理 ===
        pcltransform = transforms.Compose([
            pcl_transforms.PointcloudNormalize()
        ])

        # === DataAugmentation（测试模式，含深度图 CenterCrop）===
        test_transform = DataAugmentation(
            weak_transform=transform,
            strong_transform=pcltransform,
            weak_pcltransform=None,
            strong_pcltransform=None,
            args=args,
            config=config
        )
        return test_transform

    # args = config.args
    # if is_train == 'train':
    #     weak_transform = transforms.Compose([
    #         transforms.Resize(args.input_size, interpolation=T.InterpolationMode.BICUBIC),
    #         transforms.RandomCrop(args.input_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=torch.tensor(args.image_mean),
    #             std=torch.tensor(args.image_std))
    #     ])

    #     strong_transform = create_transform(
    #         input_size=args.input_size,
    #         scale=(args.train_crop_min, 1),
    #         is_training=True,
    #         color_jitter=args.color_jitter,
    #         auto_augment=args.aa,
    #         interpolation=args.train_interpolation,
    #         mean=args.image_mean,
    #         std=args.image_std
    #     )

    #     strong_pcltransform = transforms.Compose([
    #         pcl_transforms.PointcloudNormalize(),
    #         pcl_transforms.PointcloudScale(lo=0.5, hi=2, p=1),
    #         pcl_transforms.PointcloudRotate(),
    #         pcl_transforms.PointcloudTranslate(0.5, p=1),
    #         pcl_transforms.PointcloudJitter(p=1),
    #         pcl_transforms.PointcloudRandomInputDropout(p=1),
    #         pcl_transforms.PointcloudRandomCrop(x_min=9, x_max=1.1, ar_min=0.9, ar_max=1.1),
    #         pcl_transforms.PointcloudUpSampling(8192)
    #     ])

    #     weak_pcltransform = transforms.Compose([
    #         pcl_transforms.PointcloudNormalize(),
    #         pcl_transforms.PointcloudScale(lo=0.9, hi=1.1, p=1),
    #         pcl_transforms.PointcloudRotatePerturbation(),
    #         # pcl_transforms.PointcloudTranslate(0.2, p=1),
    #     ])

    #     # 深度图增强 - 弱增强
    #     depth_transform_weak = transforms.Compose([
    #         transforms.Resize(args.input_size, interpolation=T.InterpolationMode.NEAREST),
    #         transforms.RandomCrop(args.input_size)
    #     ])
        
    #     # 深度图增强 - 强增强
    #     depth_transform_strong = transforms.Compose([
    #         transforms.Resize(args.input_size, interpolation=T.InterpolationMode.NEAREST),
    #         transforms.RandomCrop(args.input_size),
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.RandomRotation(10),
    #         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
    #     ])
        
    #     transform = DataAugmentation(weak_transform, strong_transform, weak_pcltransform, strong_pcltransform, depth_transform_weak, depth_transform_strong, args, config)

    #     return transform

    # else:
    #     transform = transforms.Compose([
    #         transforms.Resize(args.input_size, interpolation=T.InterpolationMode.BICUBIC),
    #         transforms.CenterCrop(args.input_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=torch.tensor(args.image_mean),
    #             std=torch.tensor(args.image_std))
    #     ])

    #     pcltransform = transforms.Compose([
    #         pcl_transforms.PointcloudNormalize()
    #     ])

    #     transform = DataAugmentation(transform, pcltransform, None, None, args, config)

    #     return transform


import collections.abc as container_abcs

int_classes = int
# from torch._six import string_classes

import re

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")
np_str_obj_array_pattern = re.compile(r'[SaUO]')


def customized_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)

    if isinstance(batch, list):
        batch = [example for example in batch if example[4] is not None]

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return customized_collate_fn([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: customized_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(customized_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [customized_collate_fn(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        new_config = yaml.load(f, Loader=yaml.FullLoader)
    merge_new_config(config=config, new_config=new_config)
    return config


# Dataset_3D类用于构建和管理3D数据集
class Dataset_3D():
    def __init__(self, args, tokenizer, dataset_type, train_transform=None, train_config=None):
        # 根据数据集类型(train/val/test)设置数据集名称
        if dataset_type == 'train':
            self.dataset_name = args.pretrain_dataset_name  # 预训练数据集名称
        elif dataset_type == 'val':
            self.dataset_name = args.validate_dataset_name  # 验证数据集名称 
        elif dataset_type == 'test':
            self.dataset_name = args.validate_dataset_name  # 测试数据集名称
        else:
            raise ValueError("not supported dataset type.")

        # 从dataset_catalog.json加载数据集配置信息
        with open('./data/dataset_catalog.json', 'r') as f:
            self.dataset_catalog = json.load(f)
            # 设置数据集分割和配置目录
            self.dataset_split = dataset_type
            self.dataset_config_dir = self.dataset_catalog[self.dataset_name]['config']

        # 保存tokenizer和数据增强transform
        self.tokenizer = tokenizer
        self.train_transform = train_transform
        
        # 保存数据集提示信息
        self.pretrain_dataset_prompt = args.pretrain_dataset_prompt
        self.validate_dataset_prompt = args.validate_dataset_prompt
        
        # 构建3D数据集
        self.build_3d_dataset(args, self.dataset_config_dir, train_config)

    def build_3d_dataset(self, args, config, train_config = None):
        # 从yaml文件加载配置
        config = cfg_from_yaml_file(config)
        
        # 设置配置参数
        config.tokenizer = self.tokenizer
        config.train_transform = self.train_transform
        config.pretrain_dataset_prompt = self.pretrain_dataset_prompt  
        config.validate_dataset_prompt = self.validate_dataset_prompt
        config.args = args
        config.use_height = args.use_height  # 是否使用高度信息
        config.npoints = args.npoints  # 点云中点的数量
        config.update(train_config)  # 更新训练配置
        
        # 设置其他配置参数
        config_others = EasyDict({'subset': self.dataset_split, 'whole': True})
        
        # 根据数据集名称构建相应的数据集
        self.dataset = build_dataset_from_cfg(config, config_others)
