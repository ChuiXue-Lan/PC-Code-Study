import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import read_ply
from datasets.utils import pc_normalize, offread_uniformed

cats = {'airplane': 0, 'bathtub': 1, 'bed': 2, 'bench': 3, 'bookshelf': 4, 'bottle': 5, 'bowl': 6, 'car': 7, 'chair': 8, 'cone': 9, 'cup': 10, 'curtain': 11, 'desk': 12, 'door': 13, 'dresser': 14, 'flower_pot': 15, 'glass_box': 16, 'guitar': 17, 'keyboard': 18, 'lamp': 19, 'laptop': 20, 'mantel': 21, 'monitor': 22, 'night_stand': 23, 'person': 24, 'piano': 25, 'plant': 26, 'radio': 27, 'range_hood': 28, 'sink': 29, 'sofa': 30, 'stairs': 31, 'stool': 32, 'table': 33, 'tent': 34, 'toilet': 35, 'tv_stand': 36, 'vase': 37, 'wardrobe': 38, 'xbox': 39}


class ModelNet40Align(Dataset):
    '''
    ModelNet40Align数据集类
    从.off文件中随机采样点云数据,由于采样的随机性,结果可能比声明的结果更好或更差
    
    参数:
        partition (str): 数据集划分,可选'train'或'test'
        few_num (int): 小样本学习时每类选取的样本数,默认为0表示使用全部数据
        num_points (int): 采样的点数,默认1024
    '''
    def __init__(self, partition='test', few_num=0, num_points=1024):
        assert partition in ('test', 'train')
        super().__init__()
        self.partition = partition
        self.few_num = few_num  # 小样本学习时每类选取的样本数
        self.num_points = num_points  # 采样的点数
        self._load_data()  # 加载数据
        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()  # 小样本学习时筛选数据

    def _load_data(self):
        DATA_DIR = './data/modelnet40_manually_aligned'  # 数据集路径
        self.paths = []  # 存储文件路径
        self.labels = []  # 存储标签
        for cat in os.listdir(DATA_DIR):  # 遍历所有类别
            cat_path = os.path.join(DATA_DIR, cat, self.partition)
            for case in os.listdir(cat_path):  # 遍历该类别下所有样本
                if case.endswith('.off'):  # 只读取.off文件
                    self.paths.append(os.path.join(cat_path, case))
                    self.labels.append(cats[cat])  # 添加类别标签

    def _few(self):
        '''小样本学习时随机选取指定数量的样本'''
        num_dict = {i: 0 for i in range(40)}  # 记录每类已选样本数
        few_paths = []  # 存储选中的样本路径
        few_labels = []  # 存储选中的样本标签
        random_list = [k for k in range(len(self.labels))]
        random.shuffle(random_list)  # 随机打乱索引
        for i in random_list:
            label = self.labels[i]
            if num_dict[label] == self.few_num:  # 该类别已选够样本
                continue
            few_paths.append(self.paths[i])
            few_labels.append(self.labels[i]) 
            num_dict[label] += 1
        return few_paths, few_labels
    
    def __getitem__(self, index):
        '''获取指定索引的样本'''
        point = torch.from_numpy(offread_uniformed(self.paths[index], self.num_points)).to(torch.float32)
        label = self.labels[index]
        point = pc_normalize(point)  # 点云归一化
        if self.partition == 'train':  # 训练时随机打乱点的顺序
            pt_idxs = np.arange(point.shape[0])
            np.random.shuffle(pt_idxs)
            point = point[pt_idxs]
            return point, label
        return point, label
    
    def __len__(self):
        '''返回数据集大小'''
        return len(self.labels)


class ModelNet40Ply(Dataset):
    '''
        we save the random points in our few-shot learning, so the results are confirmed
    '''
    def __init__(self, partition='test', few_num=0, num_points=1024):
        assert partition in ('test', 'train')
        super().__init__()
        self.partition = partition
        self.few_num = few_num
        self.num_points = num_points
        self._load_data()
        if self.partition == 'train' and self.few_num > 0:
            self.paths, self.labels = self._few()

    def _load_data(self):
        DATA_DIR = './data/ModelNet40_Ply'
        self.paths = []
        self.labels = []
        for case in os.listdir(DATA_DIR):
            self.paths.append(os.path.join(DATA_DIR, case))
            self.labels.append(int(case.split('_')[0]))
    
    def __getitem__(self, index):
        label = self.labels[index]
        point = torch.from_numpy(read_ply(self.paths[index]))
        point = pc_normalize(point)
        return point, label
    
    def __len__(self):
        return len(self.labels)
