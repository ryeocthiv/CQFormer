from statistics import mode
import sys
import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from models.UNeXt import UNext_S, UNext
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import torch.nn.functional as F
from thop import profile


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


class conv_embedding(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_embedding, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            # nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            # nn.BatchNorm2d(out_channels // 2),
            # nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        return x
    
class Query_Attention(nn.Module):
    def __init__(self, dim, num_colours, num_heads=1, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Parameter(torch.randn((1, num_colours, dim)), requires_grad=True)
        # print(self.q)
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




class Colour_Quantisation(nn.Module):
    def __init__(self, temperature=0.01, num_colours=2, num_heads=4,out_channels = 256):
        super().__init__()
        self.num_colors = num_colours
        self.Image_Encoder = UNext(num_classes=num_colours)
        self.conv_large = conv_embedding(3, out_channels)
        self.temperature = temperature
        self.Colour_Cluster_Transformer = Colour_Cluster_Transformer(dim=out_channels, num_colours=num_colours,
                                                                     num_heads=num_heads)
        self.Colour_Coordinate_Projection = Mlp(in_features=out_channels, hidden_features=out_channels, out_features=3)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax2d()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x_RGB, training=True):
        origin_probability_map, _ = self.Image_Encoder(x_RGB)
        origin_probability_map = self.softmax(origin_probability_map)
        updated_colour_query = self.Colour_Cluster_Transformer(self.conv_large(x_RGB))  # torch.Size([3, 4, dim])
        colour_palette = self.Colour_Coordinate_Projection(updated_colour_query)  # torch.Size([3, 4, 2])
        colour_palette = self.sigmoid(colour_palette)
        colour_palette = colour_palette.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)# torch.Size([b, 3, n,1,1])
        if training:
            probability_map = self.softmax(origin_probability_map / self.temperature)  # torch.Size([3, 4, 256, 256])
            transformed_img = (probability_map.unsqueeze(1) * colour_palette).sum(dim=2)
            return transformed_img, probability_map
        else:
            probability_map = self.softmax(origin_probability_map)   # torch.Size([3, 4, 256, 256])
            index_map = torch.argmax(probability_map, dim=1, keepdim=True)
            index_map_one_hot = torch.zeros_like(probability_map).scatter(1, index_map, 1)
            transformed_img = (index_map_one_hot.unsqueeze(1) * colour_palette).sum(dim=2)# !!!!!!!!!!!!!!!!!!!!!
            return transformed_img, index_map_one_hot

if __name__ == "__main__":
    device = torch.device("cuda:7")
    img = torch.randn((2, 3, 256, 256)).to(device)
    model = Colour_Quantisation(num_colours=2).to(device)
    output = model(img)
