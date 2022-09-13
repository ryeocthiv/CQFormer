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
        self.D65_list = [0.7024318405595356, 0.7765422877126269, 0.7930584320782263, 0.7357680032594303,
                         0.8901045733881099, 0.9931755678538688, 1.0, 0.9749516178317998, 0.9839659796964656,
                         0.9235986147421316, 0.9282076528706753, 0.9150341221607308, 0.8894679659117918,
                         0.9140749668964112, 0.8862000475333582, 0.8831528197467152, 0.8488099684242691,
                         0.8176942926017723, 0.813058092554239, 0.7527722133568736, 0.7639815977998846,
                         0.7605260924184294, 0.7443953077784946, 0.7069619393610159, 0.710447153091366,
                         0.679275455810953, 0.6808695209316539, 0.6983821682001834, 0.6644840933011917,
                         0.59180134451499, 0.6078251790989033]

        self.D50_list = [0.4787045037523179, 0.5486539227012805, 0.5828373930856383, 0.5613234565983515,
                         0.7264351523742028, 0.8470335815461686, 0.8797025329359338, 0.8870421249866508,
                         0.9233614554915875, 0.8928186557673077, 0.9293321553741154, 0.9379629719522732,
                         0.9429725347805404, 0.991223556595439, 0.9781753929497199, 0.9933399998058309,
                         0.9708455093540965, 0.9488558585672262, 0.9603409609428852, 0.9077308427909866,
                         0.9483995611778298, 0.9637486286807181, 0.9615448093744843, 0.9293127384639281,
                         0.9597487451821791, 0.9287787734337835, 0.9532732056347873, 1.0, 0.9624282787879964,
                         0.848334514528703, 0.8893333203887265]
        self.F2_list = [0.09834190966266439, 0.11006289308176102, 0.11978273299028018, 0.14465408805031446,
                        0.33762149799885655, 0.1895368782161235, 0.2055460263007433, 0.21555174385363066,
                        0.218696397941681, 0.21783876500857635, 0.2081189251000572, 0.20154373927958835,
                        0.20468839336763867, 0.22984562607204118, 0.2861635220125786, 0.4757004002287022,
                        0.4619782732990281, 0.5323041738136078, 0.6515151515151516, 0.5334476843910807,
                        0.47284162378502004, 0.3945111492281304, 0.3130360205831904, 0.24013722126929676,
                        0.18038879359634077, 0.13379073756432247, 0.0986277873070326, 0.07289879931389366,
                        0.05403087478559177, 0.0437392795883362, 0.03144654088050315]
        self.A_list = [0.056222811752203734, 0.0675656149417818, 0.08025550263377193, 0.09430700071100374,
                       0.10971896239325389, 0.12647418597717144, 0.14454056161650142, 0.16387221810230812,
                       0.18441105190327292, 0.20608863846606681, 0.22882508543512667, 0.2525343842936981,
                       0.27712288132353735, 0.302492335685507, 0.3285410661997997, 0.35516548038623563,
                       0.3822600744642625, 0.40972163821377516, 0.4374431388139235, 0.46532901124609144,
                       0.4932798678909183, 0.5211963211290435, 0.5489942737440846, 0.5765858059189151,
                       0.6038906430378974, 0.6308361556868831, 0.6573458918509798, 0.6833625125190175,
                       0.7088210334783374, 0.7336755835200038, 0.7578726462335915]

        weight = torch.FloatTensor(weight)
        weight1 = weight[0, :] / torch.sum(weight[0, :])
        weight2 = weight[1, :] / torch.sum(weight[1, :])
        weight3 = weight[2, :] / torch.sum(weight[2, :])
        weight = torch.cat((weight1.unsqueeze(0), weight2.unsqueeze(0), weight3.unsqueeze(0)), dim=0)
        self.weight = nn.Parameter(data=weight.unsqueeze(2).unsqueeze(3), requires_grad=False)

    def forward(self, HSI):
        D65 = self.D65_list
        D50 = self.D50_list
        for i in range(31):
            HSI[:, i, :, :] = HSI[:, i, :, :] / D65[i] * D50[i]
        LMS = F.conv2d(HSI, self.weight)
        return LMS

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
        print(self.q)
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
        self.Spectral_Reconstruction = MST_Plus_Plus()
        self.Spectral_Reconstruction.load_state_dict(
            torch.load(
                '/home/ssh685/CV_project_ICLR2023/Colour-Quantisation-main/models/pretrain_model/mst_plus_plus.pth',
                map_location='cuda')['state_dict'])

        self.LMS_Projection = LMS_Projection()
        self.Image_Encoder = UNext(num_classes=num_colours)
        self.conv_large = conv_embedding(3, out_channels)
        # self.color_mask = nn.Sequential(nn.Conv2d(64, 256, 1), nn.ReLU(),
        #                                 nn.Conv2d(256, num_colours, 1, bias=False))
        self.temperature = temperature
        self.Colour_Cluster_Transformer = Colour_Cluster_Transformer(dim=out_channels, num_colours=num_colours,
                                                                     num_heads=num_heads)
        self.Colour_Coordinate_Projection = Mlp(in_features=out_channels, hidden_features=out_channels, out_features=3)
        # self.color_mask = nn.Sequential(nn.Conv2d(64, 256, 1), nn.ReLU(),
        #                                 nn.Conv2d(256, num_colours, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
        self.convertor = RGB_HSV()
        self.softmax = nn.Softmax2d()
        
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x_RGB, training=True):
        # HSI = self.Spectral_Reconstruction(x_RGB)
        # LMS = self.LMS_Projection(HSI)
        origin_probability_map, _ = self.Image_Encoder(x_RGB)
        
        # origin_probability_map = self.color_mask(origin_probability_map)
        # origin_probability_map = self.softmax(origin_probability_map)
        # probability_map = self.color_mask(feature)
        updated_colour_query = self.Colour_Cluster_Transformer(self.conv_large(x_RGB))  # torch.Size([3, 4, dim])

        colour_palette = self.Colour_Coordinate_Projection(updated_colour_query)  # torch.Size([3, 4, 2])
        colour_palette = self.sigmoid(colour_palette)
        colour_palette = colour_palette.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        # print(colour_palette)
        if training:
            # print(torch.mean(origin_probability_map))
            probability_map = self.softmax(origin_probability_map / self.temperature)  # torch.Size([3, 4, 256, 256])
            # print(torch.mean(probability_map))
            transformed_img = (probability_map.unsqueeze(1) * colour_palette).sum(dim=2)
            # print(torch.mean(probability_map))
            # transformed_img = self.convertor.hsv_to_rgb(transformed_img)
            return transformed_img, probability_map#,self.softmax(origin_probability_map)
        else:
            probability_map = self.softmax(origin_probability_map)   # torch.Size([3, 4, 256, 256])
            index_map = torch.argmax(probability_map, dim=1, keepdim=True)
            index_map_one_hot = torch.zeros_like(probability_map).scatter(1, index_map, 1)
            transformed_img = (index_map_one_hot.unsqueeze(1) * colour_palette).sum(dim=2)
            # transformed_img = self.convertor.hsv_to_rgb(transformed_img)
            return transformed_img, index_map_one_hot#,self.softmax(origin_probability_map)
            # probability_map = F.softmax(probability_map / self.temperature, dim=1)  # torch.Size([3, 4, 256, 256])
            # transformed_img = (probability_map.unsqueeze(1) * colour_palette).sum(dim=2)
            # transformed_img = self.convertor.hsv_to_rgb(transformed_img)
            # return transformed_img, probability_map


if __name__ == "__main__":
    img = torch.randn((2, 3, 32, 32))
    model = Colour_Quantisation(num_colours=2)
    transformed_img, probability_map = model(img, training=True)
    print(transformed_img.shape)
    print(probability_map.shape)

