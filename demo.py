import cv2
from PIL import Image



from utils.trainer import CNNTrainer_1

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
from models.Colour_Quantisation_1 import Colour_Quantisation
from models.resnet import ResNet18
import os
import sys
from models.convert_RGB_HSV import RGB_HSV

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
                        default='/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/logs/CQ_5/cifar10/2_colours/2022-09-12_00-44-32/CQ_epoch60.pth')
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
    device = torch.device('cuda')
    # dataset
    data_path = os.path.expanduser('/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/Data/') + args.dataset
    # dataroot = '/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/Data/'
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        H, W, C = 32, 32, 3
        num_class = 10 if args.dataset == 'cifar10' else 100

        normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_trans = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), ])
        test_trans = T.Compose([T.ToTensor(), ])
        # visualise_set = CIFAR10_CIFAR100_Dataset_for_visualise(dataroot=dataroot)
        if args.dataset == 'cifar10':
            train_set = datasets.CIFAR10(data_path, train=True, download=True, transform=train_trans)
            test_set = datasets.CIFAR10(data_path, train=False, download=True, transform=test_trans)
        else:
            train_set = datasets.CIFAR100(data_path, train=True, download=True, transform=train_trans)
            test_set = datasets.CIFAR100(data_path, train=False, download=True, transform=test_trans)
    elif args.dataset == 'stl10':
        H, W, C = 96, 96, 3
        num_class = 10
        args.batch_size = 32
        # smaller batch size
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(96, padding=12), T.RandomHorizontalFlip(), T.ToTensor() ])
        test_trans = T.Compose([T.ToTensor() ])
        train_set = datasets.STL10(data_path, split='train', download=True, transform=train_trans)
        test_set = datasets.STL10(data_path, split='test', download=True, transform=test_trans)
    elif args.dataset == 'tiny200':
        H, W, C = 64, 64, 3
        num_class = 200
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor() ])
        test_trans = T.Compose([T.ToTensor() ])

        train_set = datasets.ImageFolder(
            '/home/ssh685/CV_project_AAAI2023/color_distillation-master/Data/tiny200/train', transform=train_trans)
        test_set = datasets.ImageFolder('/home/ssh685/CV_project_AAAI2023/color_distillation-master/Data/tiny200/val',
                                        transform=test_trans)
    else:
        raise Exception

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    print(time)
    print('Settings:')
    print(vars(args))

    # model
    model = Colour_Quantisation(num_colours=args.num_colors).cuda()
    log_dir = args.log_dir
    log_dict = torch.load(log_dir, map_location='cuda')
    model.load_state_dict(log_dict)

    
    WCS_matrix_tmp =  np.ones((args.num_colors,8,40))
    # learn
    for batch_idx, (rgb, target) in enumerate(test_loader):
        WCS_matrix = np.ones((args.num_colors,8,40))*(-1)
        model.eval()
        rgb, target = rgb.cuda(), target.cuda()
        hsv = RGB_HSV().rgb_to_hsv(rgb)

        with torch.no_grad():
            _, probability_map = model(rgb, training=False)
            hsv_color_contribution = (hsv.unsqueeze(2) * probability_map.unsqueeze(1))  # torch.Size([1, 3, 2, 32, 32])

        hsv_color_contribution =hsv_color_contribution.squeeze().flatten(2)
        for i in range(args.num_colors):
            hsv_color_contribution_per_cls = hsv_color_contribution[:,i,:]
            hsv_color_contribution_per_cls = hsv_color_contribution_per_cls.detach().cpu().numpy()
            hsv_color_contribution_per_cls = hsv_color_contribution_per_cls[:,[not np.all(hsv_color_contribution_per_cls[:, i] == 0) for i in
                                            range(hsv_color_contribution_per_cls.shape[1])]]
            h_color_contribution_per_cls = np.uint8(hsv_color_contribution_per_cls[0, :] * 39)
            v_color_contribution_per_cls = np.uint8(hsv_color_contribution_per_cls[2, :] * 7)
            assert len(h_color_contribution_per_cls) == len(v_color_contribution_per_cls)
            for j in range(len(h_color_contribution_per_cls)):
                WCS_matrix[i,v_color_contribution_per_cls[j],h_color_contribution_per_cls[j]] = WCS_matrix[i,v_color_contribution_per_cls[j],h_color_contribution_per_cls[j]]+1
        WCS_matrix_argmax = np.argmax(WCS_matrix,axis=0)
        print(np.sum((WCS_matrix_argmax-WCS_matrix_tmp)**2)/320)
        # print(WCS_matrix_argmax)
        WCS_matrix_tmp = WCS_matrix_argmax

    # for i in range(args.num_colors):
    #     print(WCS_matrix[i,:,:])
    # WCS_matrix_argmax = np.argmax(WCS_matrix,axis=0)
    # print(WCS_matrix_argmax)
            # hsv_color_contribution_per_cls = hsv_color_contribution[:, :, 0, :, :].squeeze().flatten(
            #     1)  # torch.Size([3, 1024])
            # hsv_color_contribution_per_cls = hsv_color_contribution_per_cls.detach().cpu().numpy()
            # hsv_color_contribution_per_cls = hsv_color_contribution_per_cls[:,
            #                                  [not np.all(hsv_color_contribution_per_cls[:, i] == 0) for i in
            #                                   range(hsv_color_contribution_per_cls.shape[1])]]  # (3, 483)

            # h_color_contribution_per_cls = np.uint8(hsv_color_contribution_per_cls[0, :] * 39)
            # v_color_contribution_per_cls = np.uint8(hsv_color_contribution_per_cls[2, :] * 7)
            # assert len(h_color_contribution_per_cls) == len(v_color_contribution_per_cls)
            # h_color_contribution_per_cls = h_color_contribution_per_cls.tolist()
            # v_color_contribution_per_cls = v_color_contribution_per_cls.tolist()
            # colour_1_h.extend(h_color_contribution_per_cls)
            # colour_1_v.extend(v_color_contribution_per_cls)

            # hsv_color_contribution_per_cls = hsv_color_contribution[:, :, 1, :, :].squeeze().flatten(1)
            # hsv_color_contribution_per_cls = hsv_color_contribution_per_cls.detach().cpu().numpy()
            # hsv_color_contribution_per_cls = hsv_color_contribution_per_cls[:,
            #                                  [not np.all(hsv_color_contribution_per_cls[:, i] == 0) for i in
            #                                   range(hsv_color_contribution_per_cls.shape[1])]]  # (3, 483)

            # h_color_contribution_per_cls = np.uint8(hsv_color_contribution_per_cls[0, :] * 39)
            # v_color_contribution_per_cls = np.uint8(hsv_color_contribution_per_cls[2, :] * 7)
            # assert len(h_color_contribution_per_cls) == len(v_color_contribution_per_cls)
            # h_color_contribution_per_cls = h_color_contribution_per_cls.tolist()
            # v_color_contribution_per_cls = v_color_contribution_per_cls.tolist()
            # colour_2_h.extend(h_color_contribution_per_cls)
            # colour_2_v.extend(v_color_contribution_per_cls)
        # print(len(colour_1_h))
        # print(len(colour_1_v))
        # print(len(colour_2_h))
        # print(len(colour_2_v))
        # print("*" * 20)
    # colour_1_h = [str(x) for x in colour_1_h]
    # colour_1_v = [str(x) for x in colour_1_v]
    # colour_2_h = [str(x) for x in colour_2_h]
    # colour_2_v = [str(x) for x in colour_2_v]
    # save_txt_root = '/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/logs/WCS/{}_colours_tmp/'.format(
    #     args.num_colors)
    # if not os.path.exists(save_txt_root):
    #     os.makedirs(save_txt_root)
    # assert len(colour_1_h) == len(colour_1_v)
    # with open(os.path.join(save_txt_root, 'colour_1.txt'), 'w') as f:
    #     for i in range(len(colour_1_h)):
    #         f.write(str(colour_1_h[i]) + '\t' + str(colour_1_v[i]) + '\n')

    # assert len(colour_2_h) == len(colour_2_v)
    # with open(os.path.join(save_txt_root, 'colour_2.txt'), 'w') as f:
    #     for i in range(len(colour_2_h)):
    #         f.write(str(colour_2_h[i]) + '\t' + str(colour_2_v[i]) + '\n')


if __name__ == '__main__':
    main()
    # print(0.98**200)