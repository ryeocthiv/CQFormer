
from PIL import Image

from utils.image_utils import img_color_denormalize
from utils.logger import Logger
from utils.draw_curve import draw_curve
from utils.load_checkpoint import checkpoint_loader
from utils.trainer import Trainer
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
import captum
from captum.attr import *
from tqdm import *

def main():
    # settings
    parser = argparse.ArgumentParser(description='Colour Quantisation')
    parser.add_argument('-d', '--dataset', type=str, default='EuroSAT',
                        choices=['cifar10', 'cifar100', 'stl10', 'svhn', 'imagenet', 'tiny200','EuroSAT','AID'])
    parser.add_argument('-j', '--num_workers', type=int, default=16)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 128)')
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
        model_dir = '/home/ssh685/ICCV2023/Colour-Quantisation-main/logs/upper_bound/EuroSAT/2023-06-03_17-17-52/classifier_best.pth'
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor()  ])
        test_trans = T.Compose([T.ToTensor()  ])

        train_set = datasets.ImageFolder(os.path.join(data_path,'train'), transform=train_trans)
        test_set = datasets.ImageFolder(os.path.join(data_path,'val'),transform=test_trans)
    elif args.dataset == 'AID':
        model_dir = '/home/ssh685/ICCV2023/Colour-Quantisation-main/logs/upper_bound/AID/2023-06-03_17-17-55/classifier_best.pth'
        H, W, C = 600, 600, 3
        num_class = 30
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomResizedCrop(96), T.RandomHorizontalFlip(), T.ToTensor()  ])
        test_trans = T.Compose([T.Resize((96, 96)),T.ToTensor()])

        train_set = datasets.ImageFolder(os.path.join(data_path,'train'), transform=train_trans)
        test_set = datasets.ImageFolder(os.path.join(data_path,'val'),transform=test_trans)
    else:
        raise Exception
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    print(time)
    logdir = 'logs/XAI_upper_bound/{}/{}/'.format(args.dataset, time)
    if args.resume is None:
        os.makedirs(logdir, exist_ok=True)
        copy_tree('./models', logdir + '/scripts/model')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(
                    logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
    sys.stdout = Logger(os.path.join(logdir, 'log.txt'))
    print('Settings:')
    print(vars(args))

    # model
    classifier = torch.load(model_dir, map_location='cpu')
    classifier = classifier.cuda()
    classifier.eval()

    criterion = nn.CrossEntropyLoss()
    losses = 0
    correct = 0
    miss = 0
    sensitivity_dict = dict()

    gradient_shap = GradientShap(classifier)
    dl = DeepLift(classifier)
    dl_shap = DeepLiftShap(classifier)
    ig = IntegratedGradients(classifier)
    input_x_gradient  = InputXGradient(classifier)
    gbp  = GuidedBackprop(classifier)
    deconv  = Deconvolution(classifier)
    saliency  = Saliency(classifier)
    occlusion = Occlusion(classifier)
    # guided_gc  = GuidedGradCam(classifier, classifier.layer4.)
    # layer_gc  = LayerGradCam(classifier)

    XAI_list = ['GradientShap','DeepLift', 'DeepLiftShap', 'IntegratedGradients', 'InputXGradient', 'GuidedBackprop', 'Deconvolution', 'Saliency', 'Occlusion']
    for XAI_method in XAI_list:
        sensitivity_dict[XAI_method]=0
    # loop = tqdm(enumerate(test_loader),total=len(test_loader))
    for batch_idx, (rgb, target) in enumerate(test_loader):
        rgb, target = rgb.cuda(), target.cuda()
        B,C,H,W = rgb.shape
        with torch.no_grad():
            output = classifier(rgb)
        pred = torch.argmax(output, 1)
        correct += pred.eq(target).sum().item()
        miss += target.shape[0] - pred.eq(target).sum().item()
        loss = criterion(output, target)
        losses += loss.item()
        
      
        # GradientShap 
        sensitivity = captum.metrics.sensitivity_max(gradient_shap.attribute, rgb, baselines=torch.randn(20, 3, H, W).cuda() ,target=target)
        sensitivity_dict['GradientShap']+=sensitivity.mean().item()
    
        # DeepLift 
        sensitivity = captum.metrics.sensitivity_max(dl.attribute, rgb, target=target)
        sensitivity_dict['DeepLift']+=sensitivity.mean().item()

        # DeepLiftShap 
        sensitivity = captum.metrics.sensitivity_max(dl_shap.attribute, rgb,  baselines=torch.randn(20, 3, H, W).cuda(), target=target)
        sensitivity_dict['DeepLiftShap']+=sensitivity.mean().item()

        # IntegratedGradients 
        sensitivity = captum.metrics.sensitivity_max(ig.attribute, rgb, target=target)
        sensitivity_dict['IntegratedGradients']+=sensitivity.mean().item()

        # InputXGradient 
        sensitivity = captum.metrics.sensitivity_max(input_x_gradient.attribute, rgb, target=target)
        sensitivity_dict['InputXGradient']+=sensitivity.mean().item()

        # GuidedBackprop 
        sensitivity = captum.metrics.sensitivity_max(gbp.attribute, rgb, target=target)
        sensitivity_dict['GuidedBackprop']+=sensitivity.mean().item()

        # Deconvolution 
        sensitivity = captum.metrics.sensitivity_max(deconv.attribute, rgb, target=target)
        sensitivity_dict['Deconvolution']+=sensitivity.mean().item()

        # GuidedBackprop 
        sensitivity = captum.metrics.sensitivity_max(gbp.attribute, rgb, target=target)
        sensitivity_dict['GuidedBackprop']+=sensitivity.mean().item()

        # Saliency 
        sensitivity = captum.metrics.sensitivity_max(saliency.attribute, rgb, target=target)
        sensitivity_dict['Saliency']+=sensitivity.mean().item()

        # Occlusion 
        sensitivity = captum.metrics.sensitivity_max(occlusion.attribute, rgb, target=target,strides=(1, 2, 2),sliding_window_shapes=(3, 3, 3),baselines=0)
        sensitivity_dict['Occlusion']+=sensitivity.mean().item()

        # # GuidedGradCam 
        # sensitivity = captum.metrics.sensitivity_max(guided_gc.attribute, rgb, target=target)
        # sensitivity_dict['GuidedGradCam']+=sensitivity.mean().item()

        # # LayerGradCam 
        # sensitivity = captum.metrics.sensitivity_max(layer_gc.attribute, rgb, target=target)
        # sensitivity_dict['LayerGradCam']+=sensitivity.mean().item()

        log_string = 'Batch:{} '.format(batch_idx+1)
        for k,v in sensitivity_dict.items():
            log_string = log_string + k + ':{:.2f}'.format(v/(batch_idx+1))+' '
        print(log_string)
        # loop.set_description
    loss_all = losses/len(test_loader)
    acc = correct / (correct + miss)
    
    log_string = 'Total Loss: {:.2f} Acc:{:.2f} '.format(loss_all,acc)
    for k,v in sensitivity_dict.items():
        log_string = log_string + k + ':{:.2f}'.format(v/len(test_loader))+' '
    print(log_string)








if __name__ == '__main__':
    main()
    # CUDA_VISIBLE_DEVICES=0 python XAI_classifier.py -d EuroSAT 
    # CUDA_VISIBLE_DEVICES=1 python XAI_classifier.py -d AID 
