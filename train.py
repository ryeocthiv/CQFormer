from dateset import CIFAR10_CIFAR100_Dataset
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
    parser.add_argument('--alpha', type=float, default=1,
                        help='multiplier of regularization terms')
    parser.add_argument('--beta', type=float, default=0,
                        help='multiplier of regularization terms')
    parser.add_argument('--gamma', type=float, default=0,
                        help='multiplier of reconstruction loss')
    parser.add_argument('--color_jitter', type=float, default=1)
    parser.add_argument('--color_norm', type=float, default=4,
                        help='normalizer for color palette')
    parser.add_argument('--label_smooth', type=float, default=0.0)
    parser.add_argument('--soften', type=float, default=1,
                        help='soften coefficient for softmax')
    parser.add_argument('-d', '--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'stl10', 'svhn', 'imagenet', 'tiny200'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=256, metavar='N',
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
    parser.add_argument('--backbone', type=str,
                        default='unet', choices=['unet', 'dncnn'])
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
    device = torch.device('cuda')
    # dataset
    dataroot = os.path.expanduser('/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/Data/')
    # dataroot = '/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/Data/'
    if args.dataset == 'cifar10':
        num_class = 10
        train_set = CIFAR10_CIFAR100_Dataset(dataroot=dataroot, dataset_name='cifar10', mode='train', num_s=args.num_s)
        test_set = CIFAR10_CIFAR100_Dataset(dataroot=dataroot, dataset_name='cifar10', mode='test', num_s=args.num_s)
    else:
        raise Exception

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)
    time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    print(time)
    logdir = 'logs/CQ_HSV/{}/{}_hv_{}s/{}'.format(args.dataset,
                                                  'full_' if args.num_colors is None else args.num_colors, args.num_s,
                                                  time)
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
    model = Colour_Quantisation(num_colours=args.num_colors).cuda()
    for param in model.Spectral_Reconstruction.parameters():
        param.requires_grad = False
    for param in model.LMS_Projection.parameters():
        param.requires_grad = False
    classifier = ResNet18(out_channel=num_class).cuda()
    optimizer = optim.SGD(list(model.parameters()) + list(classifier.parameters()),
                          lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 60, 1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
    #                                                 steps_per_epoch=len(train_loader), epochs=args.epochs)

    # loss

    criterion = nn.CrossEntropyLoss()

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    og_test_loss_s = []
    og_test_prec_s = []
    best_acc = 0
    trainer = CNNTrainer(model, criterion, args.num_colors,
                         classifier, args.alpha, args.beta, args.gamma)

    # learn

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
            torch.save(model.state_dict(), os.path.join(
                logdir, 'CQ_best_epoch_{}_acc_{}.pth'.format(epoch, best_acc * 100)))
            torch.save(classifier.state_dict(), os.path.join(
                logdir, 'classifier_best_epoch_{}_acc_{}.pth'.format(epoch, best_acc * 100)))
        if epoch % 20 == 0:
            # save
            torch.save(model.state_dict(), os.path.join(
                logdir, 'CQ_epoch{}.pth'.format(epoch)))
            torch.save(classifier.state_dict(), os.path.join(
                logdir, 'classifier_epoch{}.pth'.format(epoch)))
        print('best acc: {}'.format(best_acc * 100))


if __name__ == '__main__':
    main()
