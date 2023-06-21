from PIL import Image
from utils.image_utils import img_color_denormalize
from utils.logger import Logger
from utils.draw_curve import draw_curve
from utils.load_checkpoint import checkpoint_loader
from utils.trainer import Trainer2
import utils.transforms as T
from dataset_cifar import CIFAR10, CIFAR100
from dataset_stl10 import STL10
from dataset_Imagefolder import ImageFolder
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
import torchvision.models as models 
import os
import sys


def main():
    # settings
    parser = argparse.ArgumentParser(description='Colour Quantisation')
    parser.add_argument('--num_colors', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0.01,
                        help='soften coefficient for softmax')
    parser.add_argument('-d', '--dataset', type=str, default='EuroSAT',
                        choices=['cifar10', 'cifar100', 'stl10', 'svhn', 'imagenet', 'tiny200','EuroSAT','AID'])
    parser.add_argument('-j', '--num_workers', type=int, default=16)
    parser.add_argument('-b', '--batch_size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=60,
                        metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5,
                        metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42,
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
    data_path = os.path.expanduser('/home/ssh685/ICCV2023/Colour-Quantisation-main/Data/') + args.dataset
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        H, W, C = 32, 32, 3
        num_class = 10 if args.dataset == 'cifar10' else 100

        normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_trans = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),  ])
        test_trans = None
        if args.dataset == 'cifar10':
            train_set =  CIFAR10(data_path, train=True, download=True, transform=train_trans, num_colors=args.num_colors)
            test_set = CIFAR10(data_path, train=False, download=True, transform=test_trans, num_colors=args.num_colors)
        else:
            train_set = CIFAR100(data_path, train=True, download=True, transform=train_trans, num_colors=args.num_colors)
            test_set = CIFAR100(data_path, train=False, download=True, transform=test_trans, num_colors=args.num_colors)
    elif args.dataset == 'stl10':
        H, W, C = 96, 96, 3
        num_class = 10
        args.batch_size = 32   
        # smaller batch size
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(96, padding=12), T.RandomHorizontalFlip(),  ])
        test_trans = None
        train_set = STL10(data_path, split='train', download=True, transform=train_trans, num_colors=args.num_colors)
        test_set = STL10(data_path, split='test', download=True, transform=test_trans, num_colors=args.num_colors)
    elif args.dataset == 'tiny200':
        H, W, C = 64, 64, 3
        num_class = 200
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(),   ])
        test_trans = None

        train_set = ImageFolder(os.path.join(data_path,'train'), transform=train_trans, num_colors=args.num_colors)
        test_set = ImageFolder(os.path.join(data_path,'val'),transform=test_trans, num_colors=args.num_colors)
    elif args.dataset == 'EuroSAT':
        H, W, C = 64, 64, 3
        num_class = 10
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(),   ])
        test_trans = None

        train_set = ImageFolder(os.path.join(data_path,'train'), transform=train_trans, num_colors=args.num_colors)
        test_set = ImageFolder(os.path.join(data_path,'val'),transform=test_trans, num_colors=args.num_colors)
    elif args.dataset == 'AID':
        H, W, C = 600, 600, 3
        num_class = 30
        # args.batch_size = 8 
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomResizedCrop(64), T.RandomHorizontalFlip(),   ])
        test_trans = T.Compose([T.Resize((64, 64))])
        train_set = ImageFolder(os.path.join(data_path,'train'), transform=train_trans, num_colors=args.num_colors)
        test_set = ImageFolder(os.path.join(data_path,'val'),transform=test_trans, num_colors=args.num_colors)
    else:
        raise Exception

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    print(time)
    logdir = 'logs/CQ_mcut/{}/{}_colours/{}/'.format(args.dataset, args.num_colors, time)
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
  
    classifier = ResNet18(out_channel=num_class).cuda()
    

    optimizer = optim.SGD(list(model.parameters()),
                          lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1)
    criterion = nn.CrossEntropyLoss()

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    og_test_loss_s = []
    og_test_prec_s = []
    best_acc = 0
    trainer = Trainer2(model, criterion, args.num_colors,classifier)

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
            torch.save(model, os.path.join(logdir, 'CQ_best.pth'))
            torch.save(classifier, os.path.join(logdir, 'classifier_best.pth'))
        print('best acc: {}'.format(best_acc * 100))


if __name__ == '__main__':
    main()
    # CUDA_VISIBLE_DEVICES=1 python train_with_mcut.py -d EuroSAT --num_colors 2


