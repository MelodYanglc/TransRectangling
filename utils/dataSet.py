import os
import numpy as np
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import utils.constant as constant
import random
import torchvision.transforms.functional as F
from random import random

grid_w = constant.GRID_W
grid_h = constant.GRID_H

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

class SPRectanglingTrainDataSet(Dataset):
    def __init__(self,input_path,mask_path,gt_path,resize_h,resize_w):
        super(SPRectanglingTrainDataSet, self).__init__()
        self.index_all = list(sorted([x.split('.')[0] for x in  os.listdir(input_path)]))
        self.input_path = input_path
        self.mask_path = mask_path
        self.gt_path = gt_path
        self.resize_h = resize_h
        self.resize_w = resize_w
        self._origin_transform = transforms.Compose([
            transforms.Resize([self.resize_h, self.resize_w]),
            transforms.ToTensor(),
        ])
        self._origin_transform2 = transforms.Compose([
            transforms.Resize([384, 512]),
            transforms.ToTensor(),
        ])
        setup_seed(2023)


    def __len__(self):
        return len(self.index_all)

    def __getitem__(self, idx):
        idx = self.index_all[idx]

        input_img = cv2.imread(os.path.join(self.input_path, idx + '.jpg'))
        mask_img = cv2.imread(os.path.join(self.mask_path, idx + '.jpg'))
        gt_img = cv2.imread(os.path.join(self.gt_path, idx + '.jpg'))
        # print("ds_mesh:",ds_mesh.shape)

        input_img = Image.fromarray(input_img)
        mask_img = Image.fromarray(mask_img)
        gt_img = Image.fromarray(gt_img)
        # ds_mesh = Image.fromarray(ds_mesh)
        # # # augument image
        # if random() > 0.5:
        #     input_img = F.vflip(input_img)
        #     mask_img = F.vflip(mask_img)
        #     gt_img = F.vflip(gt_img)
        #     ds_mesh = F.vflip(ds_mesh)

        # if random() > 0.5:
        #     input_img = F.hflip(input_img)
        #     mask_img = F.hflip(mask_img)
        #     gt_img = F.hflip(gt_img)
        #     ds_mesh = F.hflip(ds_mesh)
        ###
        super_img = self._origin_transform2(gt_img.copy())
        gt_img = self._origin_transform(gt_img)
        input_img = self._origin_transform(input_img)
        mask_img = self._origin_transform(mask_img)

        return input_img,mask_img,gt_img,super_img



class SPRectanglingTestDataSet(Dataset):
    def __init__(self,input_path,mask_path,gt_path,resize_h,resize_w):
        super(SPRectanglingTestDataSet, self).__init__()
        self.index_all = list(sorted([x.split('.')[0] for x in  os.listdir(input_path)]))
        self.input_path = input_path
        self.mask_path = mask_path
        self.gt_path = gt_path
        self.resize_h = resize_h
        self.resize_w = resize_w
        self._origin_transform = transforms.Compose([
            transforms.Resize([self.resize_h, self.resize_w]),
            transforms.ToTensor(),
        ])
        self._origin_transform2 = transforms.Compose([
            transforms.Resize([384, 512]),
            transforms.ToTensor(),
        ])
        setup_seed(2023)


    def __len__(self):
        return len(self.index_all)

    def __getitem__(self, idx):
        idx = self.index_all[idx]
        input_img = cv2.imread(os.path.join(self.input_path, idx + '.jpg'))
        mask_img = cv2.imread(os.path.join(self.mask_path, idx + '.jpg'))
        gt_img = cv2.imread(os.path.join(self.gt_path, idx + '.jpg'))

        input_img = Image.fromarray(input_img)
        mask_img = Image.fromarray(mask_img)
        gt_img = Image.fromarray(gt_img)
        ###
        super_img = self._origin_transform2(gt_img.copy())
        gt_img = self._origin_transform(gt_img)
        input_img = self._origin_transform(input_img)
        mask_img = self._origin_transform(mask_img)
        return input_img,mask_img,gt_img,super_img