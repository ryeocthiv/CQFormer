import cv2

# from loss.label_smooth import LSR_loss
import utils.transforms as T
from torchvision import datasets
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
import datetime
from distutils.dir_util import copy_tree
import shutil
import argparse
from models.Colour_Quantisation import Colour_Quantisation
from models.resnet import ResNet18
import os
import sys

print(sys.path)
sys.path.append('/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/models/pretrain_model')
sys.path.append('/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/models')
sys.path.append('/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/')
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def main():
    # settings
    parser = argparse.ArgumentParser(description='Colour Quantisation')
    parser.add_argument('--num_colors', type=int, default=2)
    parser.add_argument('--log_dir', type=str,
                        default='/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/logs/CQ_new/cifar10/2_colours/2022-08-28_21-36-26/CQ_epoch60.pth')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'stl10', 'svhn', 'imagenet', 'tiny200'])
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: None)')
    args = parser.parse_args()

    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    print(time)
    print('Settings:')
    print(vars(args))

    # model
    model = Colour_Quantisation(num_colours=args.num_colors).cuda()
    model.load_state_dict(torch.load(args.log_dir, map_location='cuda'))
    WCS_r_mat = [
        [245, 246, 247, 248, 249, 249, 249, 247, 254, 253, 247, 241, 233, 227, 223, 217, 212, 210, 208, 208, 207, 207,
         207,
         209, 211, 214, 217, 220, 224, 226, 229, 232, 234, 235, 238, 240, 241, 243, 244, 244, ],
        [254, 255, 255, 255, 255, 251, 255, 255, 255, 241, 228, 217, 202, 188, 162, 153, 132, 142, 137, 134, 131, 157,
         157,
         157, 158, 163, 168, 175, 166, 177, 197, 203, 209, 213, 229, 236, 242, 247, 250, 252, ],
        [240, 242, 255, 254, 250, 255, 249, 237, 223, 211, 200, 189, 174, 156, 135, 111, 80, 93, 85, 77, 69, 60, 103,
         103,
         105, 111, 120, 106, 125, 141, 166, 176, 185, 192, 210, 219, 237, 244, 235, 238, ],
        [237, 239, 239, 246, 243, 222, 213, 203, 190, 180, 170, 161, 148, 132, 97, 60, 43, 0, 0, 0, 19, 0, 0, 0, 0, 25,
         57,
         31, 76, 101, 136, 150, 162, 171, 191, 200, 207, 214, 231, 235, ],
        [219, 220, 220, 222, 204, 190, 177, 169, 157, 149, 141, 133, 122, 109, 79, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
         0, 0,
         46, 105, 126, 141, 152, 163, 181, 190, 199, 214, 217, ],
        [189, 189, 188, 183, 160, 147, 141, 130, 124, 118, 111, 105, 96, 87, 62, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0,
         0,
         38, 74, 101, 116, 126, 136, 144, 152, 158, 162, 165, ],
        [140, 141, 150, 137, 123, 110, 106, 101, 91, 87, 82, 78, 72, 66, 49, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0,
         52, 79, 94, 103, 111, 118, 126, 131, 135, 139, ],
        [98, 99, 99, 88, 78, 75, 73, 62, 59, 57, 55, 52, 50, 48, 36, 30, 24, 0, 0, 0, 8, 4, 1, 0, 0, 3, 0, 0, 0, 0, 37,
         56,
         67, 73, 71, 75, 87, 91, 93, 95, ]]
    WCS_g_mat = [
        [221, 221, 221, 221, 221, 221, 222, 223, 223, 225, 227, 229, 232, 232, 230, 231, 232, 232, 232, 232, 232, 232,
         231,
         231, 230, 229, 228, 227, 226, 225, 225, 224, 223, 223, 222, 222, 222, 221, 221, 221, ],
        [181, 181, 181, 182, 184, 186, 185, 184, 189, 196, 201, 204, 209, 211, 216, 216, 218, 215, 215, 215, 215, 210,
         209,
         209, 208, 207, 205, 203, 203, 200, 197, 196, 194, 193, 187, 185, 183, 182, 181, 181, ],
        [148, 148, 141, 144, 147, 144, 151, 158, 164, 170, 174, 177, 182, 186, 189, 192, 195, 192, 192, 192, 192, 192,
         187,
         186, 185, 183, 181, 181, 178, 174, 170, 167, 165, 163, 155, 153, 144, 142, 148, 148, ],
        [104, 105, 107, 102, 107, 122, 128, 134, 139, 144, 148, 151, 155, 158, 165, 169, 168, 169, 169, 169, 165, 165,
         165,
         164, 162, 160, 157, 157, 152, 148, 143, 139, 136, 133, 124, 121, 119, 116, 106, 105, ],
        [62, 63, 66, 66, 86, 97, 105, 110, 115, 118, 122, 124, 127, 131, 136, 142, 144, 142, 142, 142, 142, 138, 138,
         137,
         135, 133, 131, 130, 127, 123, 116, 110, 106, 102, 98, 87, 82, 78, 64, 62, ],
        [18, 24, 32, 44, 69, 79, 83, 88, 91, 94, 96, 98, 101, 103, 108, 110, 115, 115, 115, 115, 112, 112, 112, 111,
         106,
         105, 105, 103, 100, 97, 88, 84, 79, 75, 71, 68, 64, 61, 59, 58, ],
        [21, 23, 0, 33, 49, 58, 61, 64, 68, 70, 72, 73, 75, 76, 80, 81, 85, 85, 88, 86, 83, 83, 82, 82, 81, 80, 78, 78,
         75,
         72, 61, 57, 51, 47, 43, 38, 32, 28, 25, 22, ],
        [14, 13, 16, 30, 38, 40, 42, 46, 47, 48, 49, 50, 50, 51, 54, 55, 55, 58, 58, 58, 56, 55, 55, 55, 54, 54, 55, 53,
         51,
         49, 39, 38, 34, 30, 35, 34, 22, 19, 17, 15, ],
    ]
    WCS_b_mat = [
        [233, 231, 229, 226, 223, 218, 213, 209, 179, 149, 147, 147, 149, 179, 211, 216, 222, 226, 229, 232, 234, 237,
         241,
         243, 245, 247, 247, 248, 248, 248, 248, 247, 247, 245, 243, 241, 239, 238, 236, 234, ],
        [192, 185, 177, 168, 159, 151, 120, 0, 0, 0, 0, 0, 0, 76, 101, 143, 162, 184, 192, 199, 205, 211, 216, 222, 226,
         230, 233, 235, 253, 254, 236, 234, 233, 230, 234, 228, 220, 212, 205, 198, ],
        [160, 151, 130, 114, 100, 17, 0, 0, 0, 0, 0, 0, 0, 0, 77, 102, 128, 154, 162, 170, 178, 189, 194, 201, 208, 214,
         218, 235, 238, 239, 223, 221, 218, 214, 216, 208, 203, 188, 178, 168, ],
        [124, 111, 93, 52, 0, 29, 0, 0, 0, 0, 0, 0, 0, 7, 4, 58, 105, 124, 134, 143, 153, 164, 173, 183, 192, 200, 205,
         221,
         225, 225, 209, 207, 203, 197, 198, 188, 176, 162, 151, 137, ],
        [97, 78, 58, 0, 0, 0, 10, 0, 21, 10, 0, 0, 11, 30, 20, 30, 74, 99, 109, 118, 129, 139, 147, 157, 165, 173, 177,
         194,
         211, 212, 196, 192, 186, 180, 171, 169, 156, 139, 127, 112, ],
        [75, 55, 36, 0, 20, 31, 19, 36, 30, 25, 23, 22, 28, 36, 25, 45, 60, 76, 85, 94, 103, 114, 123, 132, 130, 135,
         152,
         154, 169, 170, 182, 165, 158, 152, 145, 136, 126, 113, 102, 91, ],
        [58, 46, 25, 20, 18, 27, 20, 12, 34, 33, 32, 32, 35, 40, 30, 41, 48, 58, 63, 71, 78, 85, 92, 99, 105, 110, 114,
         129,
         144, 145, 156, 139, 133, 127, 121, 114, 103, 92, 81, 69, ],
        [49, 40, 31, 28, 31, 25, 20, 36, 34, 34, 35, 36, 38, 40, 28, 35, 40, 40, 45, 49, 54, 58, 62, 66, 69, 72, 88, 90,
         92,
         106, 116, 99, 94, 90, 77, 74, 74, 68, 62, 56, ],
    ]
    WCS_r_arrat, WCS_g_array, WCS_b_array = np.array(WCS_r_mat), np.array(WCS_g_mat), np.array(WCS_b_mat)
    WCS_rgb = np.concatenate((WCS_r_arrat[:, :, None], WCS_g_array[:, :, None], WCS_b_array[:, :, None]), axis=2)
    # WCS_hsv = cv2.cvtColor(WCS_rgb, cv2.COLOR_RGB2HSV_FULL)

    colour_1 = []
    colour_2 = []

    model.eval()
    value_size, hue_size, channel_size = WCS_rgb.shape
    for i in range(value_size):
        for j in range(hue_size):
            wcs_pixel = torch.tensor(WCS_rgb[i, j, :] / 255.0, dtype=torch.float32).cuda()
            wcs_pixel = wcs_pixel.unsqueeze(-1).unsqueeze(-1).repeat(1, 32, 32).unsqueeze(0)
            # torch.Size([1, 3, 32, 32])
            # print(wcs_pixel)
            # print(wcs_pixel.shape)
            with torch.no_grad():
                _, probability_map = model(wcs_pixel, training=False)  # torch.Size([1, 2, 32, 32])
                probability_map = probability_map.sum(dim=[2, 3], keepdim=False).squeeze()
                max_index = torch.argmax(probability_map).detach().cpu().numpy()
                # print(probability_map)
                # print(max_index)
                if max_index == 0:
                    colour_1.append(str(j) + '\t' + str(i) + '\n')
                else:
                    colour_2.append(str(j) + '\t' + str(i) + '\n')
    print(colour_1)
    save_txt_root = '/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/logs/WCS/{}_colours_new/'.format(args.num_colors)
    if not os.path.exists(save_txt_root):
        os.makedirs(save_txt_root)
    with open(os.path.join(save_txt_root, 'colour_1.txt'), 'w') as f:
        f.writelines(colour_1)
    with open(os.path.join(save_txt_root, 'colour_2.txt'), 'w') as f:
        f.writelines(colour_2)

if __name__ == '__main__':
    main()
