import cv2
from PIL import Image

from dateset import CIFAR10_CIFAR100_Dataset_for_visualise
from utils.image_utils import img_color_denormalize
from utils.logger import Logger
from utils.draw_curve import draw_curve
from utils.load_checkpoint import checkpoint_loader
from utils.trainer import CNNTrainer
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


def main():
    # settings
    parser = argparse.ArgumentParser(description='Colour Quantisation')
    parser.add_argument('--num_colors', type=int, default=2)
    parser.add_argument('--num_s', type=int, default=2)
    parser.add_argument('--log_dir', type=str,
                        default='/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/logs/CQ_HSV_RGB/cifar10/2_hv_2s/2022-08-22_16-02-37/CQ_epoch60.pth')
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
    dataroot = os.path.expanduser('/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/Data/')
    # dataroot = '/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/Data/'
    if args.dataset == 'cifar10':
        num_class = 10
        test_set = CIFAR10_CIFAR100_Dataset_for_visualise(dataroot=dataroot, dataset_name='cifar10', mode='test',
                                                          num_s=args.num_s)
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

    save_img_dir = '/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/logs/visualise_RGB/{}/{}_hv_{}_s/'.format(
        args.dataset, args.num_colors, args.num_s)
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    # learn
    for batch_idx, (rgb, hsv_s_channel, target, class_name, image_name) in enumerate(test_loader):
        model.eval()
        rgb, hsv_s_channel, target = rgb.cuda(), hsv_s_channel.cuda(), target.cuda()
        with torch.no_grad():
            transformed_img_rgb, probability_map = model(rgb, hsv_s_channel, training=False)
        transformed_img_rgb = transformed_img_rgb.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        # print(transformed_img_rgb)
        transformed_img_rgb = transformed_img_rgb + max(-np.min(transformed_img_rgb), 0)
        transformed_img_max = np.max(transformed_img_rgb)
        if transformed_img_max != 0:
            transformed_img_rgb /= transformed_img_max
        transformed_img_rgb *= 255
        transformed_img_rgb = transformed_img_rgb.astype('uint8')

        transformed_img_rgb = Image.fromarray(transformed_img_rgb)
        save_img_class_dir = os.path.join(save_img_dir, class_name[0])
        if not os.path.exists(save_img_class_dir):
            os.makedirs(save_img_class_dir)
        transformed_img_rgb.save(os.path.join(save_img_class_dir, image_name[0]))


if __name__ == '__main__':
    main()
