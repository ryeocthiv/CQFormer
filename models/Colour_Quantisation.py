import sys
import os

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


class Mlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


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


class Query_Attention(nn.Module):
    def __init__(self, dim, num_colours, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Parameter(torch.randn((1, num_colours, dim)), requires_grad=True)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.num_colours = num_colours

    def forward(self, x):
        B, N, C = x.shape
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q.expand(B, -1, -1)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, self.num_colours, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Subsequent_Query_Attention(nn.Module):
    def __init__(self, dim, num_colours, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        # self.q = nn.Parameter(torch.randn((1, num_colours, dim)), requires_grad=True)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.num_colours = num_colours

    def forward(self, x, q):
        B, N, C = x.shape
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q = self.q.expand(B, -1, -1)
        q = q.view(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, self.num_colours, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Colour_Cluster_Transformer(nn.Module):
    def __init__(self, dim, num_colours, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Query_Attention(
            dim,
            num_colours,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Subsequent_Colour_Cluster_Transformer(nn.Module):
    def __init__(self, dim, num_colours, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Query_Attention(
            dim,
            num_colours,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Colour_Quantisation(nn.Module):
    def __init__(self, temperature=0.05, num_colours=2, num_heads=4):
        super().__init__()
        self.num_colors = num_colours
        self.Spectral_Reconstruction = MST_Plus_Plus()
        self.Spectral_Reconstruction.load_state_dict(
            torch.load(
                '/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/models/pretrain_model/mst_plus_plus.pth',
                map_location='cuda')['state_dict'])

        self.LMS_Projection = LMS_Projection()
        self.Image_Encoder = UNext(num_classes=64)
        self.color_mask = nn.Sequential(nn.Conv2d(64, 256, 1), nn.ReLU(),
                                        nn.Conv2d(256, num_colours, 1, bias=False))
        self.temperature = temperature
        self.Colour_Cluster_Transformer = Colour_Cluster_Transformer(dim=160, num_colours=num_colours,
                                                                     num_heads=num_heads)
        self.Colour_Coordinate_Projection = Mlp(in_features=160, hidden_features=160, out_features=3)
        self.sigmoid = nn.Sigmoid()
        self.convertor = RGB_HSV()

    def forward(self, x_RGB, training=True):
        # HSI = self.Spectral_Reconstruction(x_RGB)
        # LMS = self.LMS_Projection(HSI)
        feature, low_resolution_feature = self.Image_Encoder(x_RGB)
        probability_map = self.color_mask(feature)
        updated_colour_query = self.Colour_Cluster_Transformer(low_resolution_feature)  # torch.Size([3, 4, dim])

        colour_palette = self.Colour_Coordinate_Projection(updated_colour_query)  # torch.Size([3, 4, 2])
        colour_palette = self.sigmoid(colour_palette)
        colour_palette = colour_palette.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        # print(colour_palette)
        if training:
            probability_map = F.softmax(probability_map / self.temperature, dim=1)  # torch.Size([3, 4, 256, 256])
            transformed_img = (probability_map.unsqueeze(1) * colour_palette).sum(dim=2)
            transformed_img = self.convertor.hsv_to_rgb(transformed_img)
            return transformed_img, probability_map
        else:
            probability_map = F.softmax(probability_map, dim=1)  # torch.Size([3, 4, 256, 256])
            index_map = torch.argmax(probability_map, dim=1, keepdim=True)
            index_map_one_hot = torch.zeros_like(probability_map).scatter(1, index_map, 1)
            transformed_img = (index_map_one_hot.unsqueeze(1) * colour_palette).sum(dim=2)
            transformed_img = self.convertor.hsv_to_rgb(transformed_img)
            return transformed_img, index_map_one_hot
            # probability_map = F.softmax(probability_map / self.temperature, dim=1)  # torch.Size([3, 4, 256, 256])
            # transformed_img = (probability_map.unsqueeze(1) * colour_palette).sum(dim=2)
            # transformed_img = self.convertor.hsv_to_rgb(transformed_img)
            # return transformed_img, probability_map


if __name__ == "__main__":
    img = torch.randn((2, 3, 32, 32))
    model = Colour_Quantisation(num_colours=2)
    transformed_img, probability_map = model(img, training=True)
    prob_mean = torch.mean(probability_map, dim=[2, 3])

    Shannon_entropy = -prob_mean * torch.log2(prob_mean)
    Shannon_entropy = torch.mean(Shannon_entropy)
    print(transformed_img.shape)
    print(Shannon_entropy)
