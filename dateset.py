import random
from tkinter.messagebox import NO

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import os


class CIFAR10_CIFAR100_Dataset(Dataset):
    # 确定数据路径
    def __init__(self, dataroot=None, dataset_name=None, mode='train', num_s=1):
        assert dataset_name in ['cifar10', 'cifar100']
        assert mode in ['train', 'test']

        rgb_hsv_label = []
        rgb_dataroot = os.path.join(dataroot, dataset_name, mode)
        hsv_dataroot = os.path.join(dataroot, dataset_name + '_hsv', mode)
        class_name_list = os.listdir(rgb_dataroot)
        for class_name in class_name_list:
            image_name_list = os.listdir(os.path.join(rgb_dataroot, class_name))
            for image_name in image_name_list:
                rgb_image_dir = os.path.join(rgb_dataroot, class_name, image_name)
                hsv_image_dir = os.path.join(hsv_dataroot, class_name, image_name)
                label_id = class_name_list.index(class_name)
                rgb_hsv_label.append([rgb_image_dir, hsv_image_dir, label_id])

        self.rgb_hsv_label = rgb_hsv_label
        self.mode = mode
        self.num_s = num_s

    def __getitem__(self, index):
        rgb_image_dir, hsv_image_dir, label_id = self.rgb_hsv_label[index]
        rgb = Image.open(rgb_image_dir)
        hsv = Image.open(hsv_image_dir)

        if self.mode == 'train':

            rgb, hsv = self.FLIP_LR(rgb, hsv)
            rgb, hsv = self.FLIP_UD(rgb, hsv)
            rgb, hsv = self.Random_Crop(rgb, hsv)

            rgb = rgb.resize((self.w, self.h), Image.ANTIALIAS)
            hsv = hsv.resize((self.w, self.h), Image.ANTIALIAS)

            hsv_s_channel = np.array(hsv)[:, :, 1]
            if self.num_s == 1:
                hsv_s_channel = np.ones_like(hsv_s_channel) * 255.0
            else:
                ratio = 255 / self.num_s
                hsv_s_channel = np.uint8(hsv_s_channel / ratio) * ratio + ratio / 2

            rgb, hsv_s_channel = (np.asarray(rgb) / 255.0), (np.asarray(hsv_s_channel) / 255.0)
            rgb, hsv_s_channel = torch.FloatTensor(rgb).permute(2, 0, 1), torch.FloatTensor(hsv_s_channel)
            return rgb, hsv_s_channel, label_id

        elif self.mode == 'test':
            hsv_s_channel = np.array(hsv)[:, :, 1]
            if self.num_s == 1:
                hsv_s_channel = np.ones_like(hsv_s_channel) * 255.0
            else:
                ratio = 255 / self.num_s
                hsv_s_channel = np.uint8(hsv_s_channel / ratio) * ratio + ratio / 2

            rgb, hsv_s_channel = (np.asarray(rgb) / 255.0), (np.asarray(hsv_s_channel) / 255.0)
            rgb, hsv_s_channel = torch.FloatTensor(rgb).permute(2, 0, 1), torch.FloatTensor(hsv_s_channel)
            return rgb, hsv_s_channel, label_id

    def __len__(self):
        return len(self.rgb_hsv_label)

    # Data Augmentation
    # TODO: more data augmentation methods
    def FLIP_LR(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_LEFT_RIGHT)
            high = high.transpose(Image.FLIP_LEFT_RIGHT)
        return low, high

    def FLIP_UD(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_TOP_BOTTOM)
            high = high.transpose(Image.FLIP_TOP_BOTTOM)
        return low, high

    def get_params(self, low):
        self.w, self.h = low.size

        self.crop_height = random.randint(self.h / 2, self.h)  # random.randint(self.MinCropHeight, self.MaxCropHeight)
        self.crop_width = random.randint(self.w / 2, self.w)  # random.randint(self.MinCropWidth,self.MaxCropWidth)
        # self.crop_height = 224 #random.randint(self.MinCropHeight, self.MaxCropHeight)
        # self.crop_width = 224 #random.randint(self.MinCropWidth,self.MaxCropWidth)

        i = random.randint(0, self.h - self.crop_height)
        j = random.randint(0, self.w - self.crop_width)
        return i, j

    def Random_Crop(self, low, high):
        self.i, self.j = self.get_params((low))
        if random.random() > 0.5:
            low = low.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
            high = high.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
        return low, high


class CIFAR10_CIFAR100_Dataset_for_visualise(Dataset):
    # 确定数据路径
    def __init__(self, dataroot=None, dataset_name=None, mode='train', num_s=1):
        assert dataset_name in ['cifar10', 'cifar100']
        assert mode in ['train', 'test']

        rgb_hsv_label = []
        rgb_dataroot = os.path.join(dataroot, dataset_name, mode)
        hsv_dataroot = os.path.join(dataroot, dataset_name + '_hsv', mode)
        class_name_list = os.listdir(rgb_dataroot)
        for class_name in class_name_list:
            image_name_list = os.listdir(os.path.join(rgb_dataroot, class_name))
            for image_name in image_name_list:
                rgb_image_dir = os.path.join(rgb_dataroot, class_name, image_name)
                hsv_image_dir = os.path.join(hsv_dataroot, class_name, image_name)
                label_id = class_name_list.index(class_name)
                rgb_hsv_label.append([rgb_image_dir, hsv_image_dir, label_id, class_name, image_name])

        self.rgb_hsv_label = rgb_hsv_label
        self.mode = mode
        self.num_s = num_s

    def __getitem__(self, index):
        rgb_image_dir, hsv_image_dir, label_id, class_name, image_name = self.rgb_hsv_label[index]
        rgb = Image.open(rgb_image_dir)
        hsv = Image.open(hsv_image_dir)

        if self.mode == 'train':

            rgb, hsv = self.FLIP_LR(rgb, hsv)
            rgb, hsv = self.FLIP_UD(rgb, hsv)
            rgb, hsv = self.Random_Crop(rgb, hsv)

            rgb = rgb.resize((self.w, self.h), Image.ANTIALIAS)
            hsv = hsv.resize((self.w, self.h), Image.ANTIALIAS)

            hsv_s_channel = np.array(hsv)[:, :, 1]
            if self.num_s == 1:
                hsv_s_channel = np.ones_like(hsv_s_channel) * 255.0
            else:
                ratio = 255 / self.num_s
                hsv_s_channel = np.uint8(hsv_s_channel / ratio) * ratio + ratio / 2

            rgb, hsv_s_channel = (np.asarray(rgb) / 255.0), (np.asarray(hsv_s_channel) / 255.0)
            rgb, hsv_s_channel = torch.FloatTensor(rgb).permute(2, 0, 1), torch.FloatTensor(hsv_s_channel)
            return rgb, hsv_s_channel, label_id, class_name, image_name

        elif self.mode == 'test':
            hsv_s_channel = np.array(hsv)[:, :, 1]
            if self.num_s == 1:
                hsv_s_channel = np.ones_like(hsv_s_channel) * 16.0
            else:
                ratio = 255 / self.num_s
                hsv_s_channel = np.uint8(hsv_s_channel / ratio) * ratio + ratio / 2

            rgb, hsv_s_channel = (np.asarray(rgb) / 255.0), (np.asarray(hsv_s_channel) / 255.0)
            rgb, hsv_s_channel = torch.FloatTensor(rgb).permute(2, 0, 1), torch.FloatTensor(hsv_s_channel)
            return rgb, hsv_s_channel, label_id, class_name, image_name

    def __len__(self):
        return len(self.rgb_hsv_label)

    # Data Augmentation
    # TODO: more data augmentation methods
    def FLIP_LR(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_LEFT_RIGHT)
            high = high.transpose(Image.FLIP_LEFT_RIGHT)
        return low, high

    def FLIP_UD(self, low, high):
        if random.random() > 0.5:
            low = low.transpose(Image.FLIP_TOP_BOTTOM)
            high = high.transpose(Image.FLIP_TOP_BOTTOM)
        return low, high

    def get_params(self, low):
        self.w, self.h = low.size

        self.crop_height = random.randint(self.h / 2, self.h)  # random.randint(self.MinCropHeight, self.MaxCropHeight)
        self.crop_width = random.randint(self.w / 2, self.w)  # random.randint(self.MinCropWidth,self.MaxCropWidth)
        # self.crop_height = 224 #random.randint(self.MinCropHeight, self.MaxCropHeight)
        # self.crop_width = 224 #random.randint(self.MinCropWidth,self.MaxCropWidth)

        i = random.randint(0, self.h - self.crop_height)
        j = random.randint(0, self.w - self.crop_width)
        return i, j

    def Random_Crop(self, low, high):
        self.i, self.j = self.get_params((low))
        if random.random() > 0.5:
            low = low.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
            high = high.crop((self.j, self.i, self.j + self.crop_width, self.i + self.crop_height))
        return low, high


if __name__ == "__main__":
    dataroot = '/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/Data/'

    dataset = CIFAR10_CIFAR100_Dataset(dataroot=dataroot, dataset_name='cifar10', mode='train', num_s=1)
    rgb, hsv, label_id = dataset.__getitem__(0)
    print(rgb.shape)
    print(hsv.shape)
