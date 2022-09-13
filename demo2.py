import torch
import torch.nn.functional as F
from kmeans_pytorch import kmeans
import numpy as np
import torch.nn as nn
from models.convert_RGB_HSV import RGB_HSV
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
num_colors=6
rgb = torch.randn((3,3,32,32)).cuda()
B, _, H, W = rgb.shape
cluster_ids_list = []
for i in range(B):
    x = np.uint8(rgb[i,:,:,:].permute(1,2,0).detach().cpu().numpy()*255)
    x = Image.fromarray(x)
    x = x.quantize(colors=num_colors,method=0).convert('P')
    x = np.array(x)
    print(x)