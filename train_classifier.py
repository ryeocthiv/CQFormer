
from PIL import Image

from utils.image_utils import img_color_denormalize
from utils.logger import Logger
from utils.draw_curve import draw_curve
from utils.load_checkpoint import checkpoint_loader
from utils.trainer import Classifier_Trainer
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
import torchvision.models as models 
import os
import sys

def main():
    # settings
    parser = argparse.ArgumentParser(description='Colour Quantisation')
    parser.add_argument('-d', '--dataset', type=str, default='EuroSAT',
                        choices=['cifar10', 'cifar100', 'stl10', 'svhn', 'imagenet', 'tiny200','EuroSAT','AID'])
    parser.add_argument('-j', '--num_workers', type=int, default=16)
    parser.add_argument('-b', '--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=60,
                        metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1,
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
        train_trans = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor() ])
        test_trans = T.Compose([T.ToTensor() ])
        if args.dataset == 'cifar10':
            train_set =  datasets.CIFAR10(data_path, train=True, download=True, transform=train_trans)
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
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor()  ])
        test_trans = T.Compose([T.ToTensor()  ])

        train_set = datasets.ImageFolder(os.path.join(data_path,'train'), transform=train_trans)
        test_set = datasets.ImageFolder(os.path.join(data_path,'val'),transform=test_trans)
    elif args.dataset == 'EuroSAT':
        H, W, C = 64, 64, 3
        num_class = 10
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor()  ])
        test_trans = T.Compose([T.ToTensor()  ])

        train_set = datasets.ImageFolder(os.path.join(data_path,'train'), transform=train_trans)
        test_set = datasets.ImageFolder(os.path.join(data_path,'val'),transform=test_trans)
    elif args.dataset == 'AID':
        H, W, C = 600, 600, 3
        num_class = 30  
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomResizedCrop(96), T.RandomHorizontalFlip(), T.ToTensor()  ])
        test_trans = T.Compose([T.Resize((96, 96)),T.ToTensor()  ])

        train_set = datasets.ImageFolder(os.path.join(data_path,'train'), transform=train_trans)
        test_set = datasets.ImageFolder(os.path.join(data_path,'val'),transform=test_trans)
    elif args.dataset == 'CLRS':
        H, W, C = 256, 256 ,3
        num_class = 25

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.Resize((64, 64)), T.ToTensor(), normalize, ])
        denormalizer = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_set = datasets.ImageFolder(data_path + '/train', transform=train_trans)
        test_set = datasets.ImageFolder(data_path + '/val', transform=test_trans)
    elif args.dataset == 'MASATI-v2':
        H, W, C = 512, 512 ,3
        num_class = 7

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.Resize((64, 64)), T.ToTensor(), normalize, ])
        denormalizer = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_set = datasets.ImageFolder(data_path + '/train', transform=train_trans)
        test_set = datasets.ImageFolder(data_path + '/val', transform=test_trans)
    elif args.dataset == 'OPTIMAL-31':
        H, W, C = 256, 256 ,3
        num_class = 31

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.Resize((64, 64)), T.ToTensor(), normalize, ])
        denormalizer = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_set = datasets.ImageFolder(data_path + '/train', transform=train_trans)
        test_set = datasets.ImageFolder(data_path + '/val', transform=test_trans)
    elif args.dataset == 'PatternNet':
        H, W, C = 256, 256 ,3
        num_class = 38

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.Resize((64, 64)), T.ToTensor(), normalize, ])
        denormalizer = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_set = datasets.ImageFolder(data_path + '/train', transform=train_trans)
        test_set = datasets.ImageFolder(data_path + '/val', transform=test_trans)
    elif args.dataset == 'RSD46-WHU':
        H, W, C = 256, 256 ,3
        num_class = 46

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.Resize((64, 64)), T.ToTensor(), normalize, ])
        denormalizer = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_set = datasets.ImageFolder(data_path + '/train', transform=train_trans)
        test_set = datasets.ImageFolder(data_path + '/val', transform=test_trans)
    elif args.dataset == 'RSI-CB128':
        H, W, C = 128, 128 ,3
        num_class = 45

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.Resize((64, 64)), T.ToTensor(), normalize, ])
        denormalizer = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_set = datasets.ImageFolder(data_path + '/train', transform=train_trans)
        test_set = datasets.ImageFolder(data_path + '/val', transform=test_trans)
    elif args.dataset == 'RSI-CB256':
        H, W, C = 256, 256 ,3
        num_class = 35

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.Resize((64, 64)), T.ToTensor(), normalize, ])
        denormalizer = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_set = datasets.ImageFolder(data_path + '/train', transform=train_trans)
        test_set = datasets.ImageFolder(data_path + '/val', transform=test_trans)
    elif args.dataset == 'sen12ms_10':
        H, W, C = 256, 256 ,4
        num_class = 10

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.Resize((64, 64)), T.ToTensor(), normalize, ])
        denormalizer = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_set = datasets.ImageFolder(data_path + '/train', transform=train_trans)
        test_set = datasets.ImageFolder(data_path + '/val', transform=test_trans)
    elif args.dataset == 'sen12ms_17':
        H, W, C = 256, 256 ,4
        num_class = 17

        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.Resize((64, 64)), T.ToTensor(), normalize, ])
        denormalizer = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_set = datasets.ImageFolder(data_path + '/train', transform=train_trans)
        test_set = datasets.ImageFolder(data_path + '/val', transform=test_trans)
    elif args.dataset == 'SIRI-WHU':
        H, W, C = 200, 200 ,3
        num_class = 12
        
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor(), normalize, ])
        test_trans = T.Compose([T.Resize((64, 64)), T.ToTensor(), normalize, ])
        denormalizer = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        train_set = datasets.ImageFolder(data_path + '/train', transform=train_trans)
        test_set = datasets.ImageFolder(data_path + '/val', transform=test_trans)
    else:
        raise Exception

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    print(time)
    logdir = 'logs/upper_bound/{}/{}/'.format(args.dataset, time)
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
    classifier = ResNet18(out_channel=num_class).cuda()
    optimizer = optim.SGD(classifier.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    steps_per_epoch=len(train_loader), epochs=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    og_test_loss_s = []
    og_test_prec_s = []
    best_acc = 0
    trainer = Classifier_Trainer(criterion, classifier)

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
            torch.save(classifier, os.path.join(logdir, 'classifier_best.pth'))
        print('best acc: {}'.format(best_acc * 100))


if __name__ == '__main__':
    main()

# CUDA_VISIBLE_DEVICES=0 python train_classifier.py -d EuroSAT
# 'EuroSAT','AID']