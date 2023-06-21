import math
from re import S
from PIL import Image


# from loss.label_smooth import LSR_loss
import utils.transforms as T
from torchvision import datasets
import torch.utils.data
import torch
import numpy as np
import datetime
from distutils.dir_util import copy_tree
from models.resnet import ResNet18
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

def visualise(num_colors,log_dir,dataset):
    print(num_colors,log_dir,dataset)

    data_path = os.path.expanduser('/home/ssh685/ICCV2023/Colour-Quantisation-main/Data/') + dataset
    if dataset == 'cifar10' or dataset == 'cifar100':
        H, W, C = 32, 32, 3
        num_class = 10 if dataset == 'cifar10' else 100

        normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_trans = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor() ])
        test_trans = T.Compose([T.ToTensor() ])
        if dataset == 'cifar10':
            train_set =  datasets.CIFAR10(data_path, train=True, download=True, transform=train_trans)
            test_set = datasets.CIFAR10(data_path, train=False, download=True, transform=test_trans)
        else:
            train_set = datasets.CIFAR100(data_path, train=True, download=True, transform=train_trans)
            test_set = datasets.CIFAR100(data_path, train=False, download=True, transform=test_trans)
    elif dataset == 'stl10':
        H, W, C = 96, 96, 3
        num_class = 10
        batch_size = 32   
        # smaller batch size
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(96, padding=12), T.RandomHorizontalFlip(), T.ToTensor() ])
        test_trans = T.Compose([T.ToTensor() ])
        train_set = datasets.STL10(data_path, split='train', download=True, transform=train_trans)
        test_set = datasets.STL10(data_path, split='test', download=True, transform=test_trans)
    elif dataset == 'tiny200':
        H, W, C = 64, 64, 3
        num_class = 200
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor()  ])
        test_trans = T.Compose([T.ToTensor()  ])

        train_set = datasets.ImageFolder(os.path.join(data_path,'train'), transform=train_trans)
        test_set = datasets.ImageFolder(os.path.join(data_path,'val'),transform=test_trans)
    elif dataset == 'EuroSAT':
        H, W, C = 64, 64, 3
        num_class = 10
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomCrop(64, padding=8), T.RandomHorizontalFlip(), T.ToTensor()  ])
        test_trans = T.Compose([T.ToTensor()  ])

        train_set = datasets.ImageFolder(os.path.join(data_path,'train'), transform=train_trans)
        test_set = datasets.ImageFolder(os.path.join(data_path,'val'),transform=test_trans)
    elif dataset == 'AID':
        H, W, C = 600, 600, 3
        num_class = 30
        normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        train_trans = T.Compose([T.RandomResizedCrop(96), T.RandomHorizontalFlip(), T.ToTensor()  ])
        test_trans = T.Compose([T.Resize((96, 96)),T.ToTensor()  ])

        train_set = datasets.ImageFolder(os.path.join(data_path,'train'), transform=train_trans)
        test_set = datasets.ImageFolder(os.path.join(data_path,'val'),transform=test_trans)
    else:
        raise Exception

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    print(time)

    # model
    model = torch.load(os.path.join(log_dir), map_location='cpu')
    model = model.cuda()
    model.eval()

    save_img_dir = '/home/ssh685/ICCV2023/Colour-Quantisation-main/visualise_CQ7/{}/{}_colours/'.format(dataset, num_colors)
    if not os.path.exists(save_img_dir):
        os.makedirs(save_img_dir)
    for batch_idx, (rgb, target) in enumerate(test_loader):
        model.eval()
        rgb, target = rgb.cuda(),  target.cuda()
        with torch.no_grad():
            transformed_img_rgb, _ = model(rgb, training=False)
        class_idx = target.squeeze().detach().cpu().numpy()
        transformed_img_rgb = transformed_img_rgb.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        transformed_img_rgb = transformed_img_rgb + \
            max(-np.min(transformed_img_rgb), 0)
        transformed_img_max = np.max(transformed_img_rgb)
        if transformed_img_max != 0:
            transformed_img_rgb /= transformed_img_max
        transformed_img_rgb *= 255
        transformed_img_rgb = transformed_img_rgb.astype('uint8')
        transformed_img_rgb = Image.fromarray(transformed_img_rgb)
        save_img_class_dir = os.path.join(save_img_dir, str(class_idx))
        if not os.path.exists(save_img_class_dir):
            os.makedirs(save_img_class_dir)
        transformed_img_rgb.save(os.path.join(
            save_img_class_dir, str(batch_idx)+'.png'))
if __name__ == '__main__':

    # main()
    num_colors_list = [i for i in range(2,9)] + [16,32,64]
    log_root = '/home/ssh685/ICCV2023/Colour-Quantisation-main/logs/CQ7'
    dataset_list = os.listdir(log_root)
    for num_colors in num_colors_list:
        for dataset in dataset_list:
            dataset_root = os.path.join(log_root,dataset,'{}_colours'.format(num_colors))
            log_time =  os.listdir(dataset_root)[0]
            log_dir =  os.path.join(dataset_root,log_time,'CQ_best.pth') #CQ_epoch60.pth
            visualise(num_colors,log_dir,dataset)