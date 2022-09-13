import sys
import os

from PIL import Image
from matplotlib import pyplot as plt

print(sys.path)
sys.path.append('/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/models/pretrain_model')
sys.path.append('/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/models')
sys.path.append('/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/')
os.environ['OMP_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from models.MST_Plus_Plus import MST_Plus_Plus
from models.UNeXt import UNext_S, UNext
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import torch.nn.functional as F
import cv2
from convert_RGB_HSV import RGB_HSV


class LMS_Projection(nn.Module):
    def __init__(self):
        super(LMS_Projection, self).__init__()
        weight = [[2.4100e-03, 8.7200e-03, 1.8400e-02, 2.8200e-02, 4.0300e-02, 4.9900e-02,
                   6.4700e-02, 9.9500e-02, 1.4000e-01, 1.9200e-01, 2.8900e-01, 4.4400e-01,
                   6.2900e-01, 7.7100e-01, 8.8100e-01, 9.4000e-01, 9.8100e-01, 1.0000e+00,
                   9.6900e-01, 9.2800e-01, 8.3400e-01, 7.0600e-01, 5.5400e-01, 4.0100e-01,
                   2.6600e-01, 1.6500e-01, 9.3000e-02, 4.9900e-02, 2.5400e-02, 1.2200e-02,
                   5.9000e-03],
                  [2.2700e-03, 8.7900e-03, 2.1700e-02, 3.9500e-02, 6.4800e-02, 8.7100e-02,
                   1.1600e-01, 1.7600e-01, 2.3600e-01, 3.0400e-01, 4.2800e-01, 6.1600e-01,
                   8.1700e-01, 9.3600e-01, 9.9500e-01, 9.7700e-01, 9.1800e-01, 8.1400e-01,
                   6.5300e-01, 4.9300e-01, 3.3400e-01, 2.0500e-01, 1.1700e-01, 6.2100e-02,
                   3.1400e-02, 1.5400e-02, 7.3000e-03, 3.4400e-03, 1.6400e-03, 7.6100e-04,
                   3.6500e-04],
                  [5.6600e-02, 2.3300e-01, 5.4400e-01, 8.0300e-01, 9.9100e-01, 9.5500e-01,
                   7.8700e-01, 6.4600e-01, 3.9000e-01, 2.1200e-01, 1.2300e-01, 6.0800e-02,
                   2.9200e-02, 1.2600e-02, 5.0900e-03, 1.9600e-03, 7.4000e-04, 2.8200e-04,
                   1.0900e-04, 4.3900e-05, 1.8300e-05, 8.0300e-06, 0.0000e+00, 0.0000e+00,
                   0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
                   0.0000e+00]]

        weight = torch.FloatTensor(weight)
        weight1 = weight[0, :] / torch.sum(weight[0, :])
        weight2 = weight[1, :] / torch.sum(weight[1, :])
        weight3 = weight[2, :] / torch.sum(weight[2, :])
        weight = torch.cat((weight1.unsqueeze(0), weight2.unsqueeze(0), weight3.unsqueeze(0)), dim=0)
        self.weight = nn.Parameter(data=weight.unsqueeze(2).unsqueeze(3), requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.weight)
        return x


class ConeResponseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.Spectral_Reconstruction = MST_Plus_Plus()
        self.Spectral_Reconstruction.load_state_dict(
            torch.load(
                '/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/models/pretrain_model/mst_plus_plus.pth',
                map_location='cuda')['state_dict'])

        self.LMS_Projection = LMS_Projection()

    def forward(self, x_RGB):
        HSI = self.Spectral_Reconstruction(x_RGB)
        LMS = self.LMS_Projection(HSI)
        return LMS


def show(x):
    transformed_img_rgb = x + max(-np.min(x), 0)
    transformed_img_max = np.max(transformed_img_rgb)
    if transformed_img_max != 0:
        transformed_img_rgb /= transformed_img_max
    transformed_img_rgb *= 255
    transformed_img_rgb = transformed_img_rgb.astype('uint8')

    transformed_img_rgb = Image.fromarray(transformed_img_rgb)
    return transformed_img_rgb


if __name__ == "__main__":
    x_RGB = Image.open('/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/pictures/n01443537_1970.JPEG')
    # x_RGB = x_RGB.resize((482,512), Image.BILINEAR)
    x_RGB = np.array(x_RGB)
    plt.imshow(x_RGB)
    plt.show()
    # x_RGB.save(r"C:\CV_project_AAAI2023\color_quantization\pretrain_model\n03763968_8.png")
    x_RGB = x_RGB / 255.0
    x_RGB = torch.from_numpy(x_RGB).type(torch.FloatTensor).permute(2, 0, 1).unsqueeze(0).cuda()

    model = ConeResponseModule().cuda()
    with torch.no_grad():
        LMS = model(x_RGB)

    LMS = LMS.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    L, M, S = LMS[:, :, 0], LMS[:, :, 1], LMS[:, :, 2]
    L, M, S = L[:, :, None], M[:, :, None], S[:, :, None]
    L = show(L.repeat(3, axis=2))
    M = show(M.repeat(3, axis=2))
    S = show(S.repeat(3, axis=2))
    L.save("/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/pictures/L.png")
    M.save("/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/pictures/M.png")
    S.save("/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/pictures/S.png")
    plt.imshow(L)
    plt.show()
    plt.imshow(M)
    plt.show()
    plt.imshow(S)
    plt.show()

    # print(cq_image)
    # cq_image = np.uint8(cq_image)
    # cq_image = cq_image[:, :, (2, 1, 0)]
    # cq_image = Image.fromarray(cq_image)
    # cq_image = cq_image.resize((32, 32), Image.BILINEAR)
    # cq_image = cq_image.resize((64, 64), Image.BILINEAR)
    # cq_image.show()
    # cq_image.save(r"C:\CV_project_AAAI2023\color_quantization\pretrain_model\n01443537_1970_CQ.png")
# tensor([[-0.0233,  0.2282],
#         [ 0.1908,  0.1296]])
#     self.upsampler = nn.Upsample(size=(416,416), mode='bilinear')
#     total = sum([param.nelement() for param in model.parameters()])

# print("Number of parameter: %.2fM" % (total / 1e6))
