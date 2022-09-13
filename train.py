import imp
import cv2
from PIL import Image
from cifar import *
# from dateset import *
from utils.image_utils import img_color_denormalize
from utils.logger import Logger
from utils.draw_curve import draw_curve
from utils.load_checkpoint import checkpoint_loader
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

print(sys.path)
sys.path.append('/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/models/pretrain_model')
sys.path.append('/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/models')
sys.path.append('/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/')
os.environ['OMP_NUM_THREADS'] = '1'


def main():
    # settings
    parser = argparse.ArgumentParser(description='Colour Quantisation')
    parser.add_argument('--num_colors', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='multiplier of regularization terms')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='multiplier of regularization terms')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='multiplier of reconstruction loss')
    parser.add_argument('--temperature', type=float, default=0.01,
                        help='soften coefficient for softmax')
    parser.add_argument('-d', '--dataset', type=str, default='cifar100',
                        choices=['cifar10', 'cifar100', 'stl10', 'svhn', 'imagenet', 'tiny200'])
    parser.add_argument('-j', '--num_workers', type=int, default=8)
    parser.add_argument('-b', '--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=60,
                        metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.05,
                        metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.5,
                        metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
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

    # dataset
    data_path = os.path.expanduser('/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/Data/') + args.dataset
    dataroot = '/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/Data_1/'

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

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    # visualise_loader = torch.utils.data.DataLoader(visualise_set, batch_size=1, shuffle=False,
    #                                                num_workers=args.num_workers, pin_memory=True)
    time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    print(time)
    logdir = 'logs/CQ_6_1/{}/{}_colours/{}/'.format(args.dataset, args.num_colors, time)
    if args.resume is None:
        os.makedirs(logdir, exist_ok=True)
        copy_tree('./models', logdir + '/scripts/model')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(
                    logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))

    # model
    model = Colour_Quantisation(temperature=args.temperature, num_colours=args.num_colors).cuda()
    for param in model.Spectral_Reconstruction.parameters():
        param.requires_grad = False
    for param in model.LMS_Projection.parameters():
        param.requires_grad = False
    classifier = ResNet18(out_channel=num_class).cuda()
    optimizer = optim.SGD(list(model.parameters()) + list(classifier.parameters()),
                          lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = optim.Adam(list(model.parameters()) + list(classifier.parameters()),lr=args.lr)
    pretrain_optimizer = optim.Adam(model.parameters() ,lr=0.0001)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1)

    # loss

    criterion = nn.CrossEntropyLoss()

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    og_test_loss_s = []
    og_test_prec_s = []
    best_acc = 0
    trainer = CNNTrainer_1(model, criterion, args.num_colors,
                           classifier, args.alpha, args.beta, args.gamma)

    # for epoch in range(1, 50):
    #     print('PreTraining...')
    #     trainer.pretrain(epoch, train_loader, pretrain_optimizer, args.log_interval, scheduler)

    #     print('PreTraining Testing...')
    #     trainer.pretrain_test(test_loader)

    for epoch in range(1, args.epochs + 1):
        print('Training...')
        train_loss, train_prec = trainer.train(
            epoch, train_loader, optimizer, args.log_interval, scheduler)

        print('Testing...')
        og_test_loss, og_test_prec = trainer.test(test_loader)

        x_epoch.append(epoch)
        train_loss_s.append(train_loss)
        train_prec_s.append(train_prec)
        og_test_loss_s.append(og_test_loss)
        og_test_prec_s.append(og_test_prec)
        draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, train_prec_s,
                   og_test_loss_s, og_test_prec_s)
        if og_test_prec > best_acc:
            best_acc = og_test_prec
            torch.save(model.state_dict(), os.path.join(logdir, 'CQ_best.pth'))
            torch.save(classifier.state_dict(), os.path.join(logdir, 'classifier_best'))
        if epoch % 20 == 0:
            # save
            torch.save(model.state_dict(), os.path.join(
                logdir, 'CQ_epoch{}.pth'.format(epoch)))
            torch.save(classifier.state_dict(), os.path.join(
                logdir, 'classifier_epoch{}.pth'.format(epoch)))
        print('best acc: {}'.format(best_acc * 100))

    # if args.dataset == 'cifar10' or args.dataset == 'cifar100':
    #     save_img_dir = 'logs/visualise_RGB/{}/{}_colours/{}/'.format(
    #         args.dataset, args.num_colors, time)
    #     if not os.path.exists(save_img_dir):
    #         os.makedirs(save_img_dir)
    #
    #     for batch_idx, (rgb, hsv, target, class_name, image_name) in enumerate(visualise_loader):
    #         model.eval()
    #         rgb, hsv, target = rgb.cuda(), hsv.cuda(), target.cuda()
    #         with torch.no_grad():
    #             transformed_img_rgb, _ = model(rgb, training=False)
    #         transformed_img_rgb = transformed_img_rgb.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    #         transformed_img_rgb = transformed_img_rgb + max(-np.min(transformed_img_rgb), 0)
    #         transformed_img_max = np.max(transformed_img_rgb)
    #         if transformed_img_max != 0:
    #             transformed_img_rgb /= transformed_img_max
    #         transformed_img_rgb *= 255
    #         transformed_img_rgb = transformed_img_rgb.astype('uint8')
    #
    #         transformed_img_rgb = Image.fromarray(transformed_img_rgb)
    #         save_img_class_dir = os.path.join(save_img_dir, class_name[0])
    #         if not os.path.exists(save_img_class_dir):
    #             os.makedirs(save_img_class_dir)
    #         transformed_img_rgb.save(os.path.join(save_img_class_dir, image_name[0]))


if __name__ == '__main__':
    main()
    # CUDA_VISIBLE_DEVICES=0 python train.py -d stl10 --num_colors 2
    # CUDA_VISIBLE_DEVICES=1 python train.py -d stl10 --num_colors 4
    # CUDA_VISIBLE_DEVICES=2 python train.py -d stl10 --num_colors 8
    # CUDA_VISIBLE_DEVICES=3 python train.py -d stl10 --num_colors 16    
    # CUDA_VISIBLE_DEVICES=4 python train.py -d stl10 --num_colors 32
    # CUDA_VISIBLE_DEVICES=5 python train.py -d stl10 --num_colors 64
    # CUDA_VISIBLE_DEVICES=6 python train.py -d cifar10 --num_colors 2
