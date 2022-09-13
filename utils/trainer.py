import math
import time
import numpy as np
import torch
import torch.nn as nn
from models.convert_RGB_HSV import RGB_HSV
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from io import BytesIO
from IQA_pytorch import SSIM
from kmeans_pytorch import kmeans
WCS_r_mat = [
    [245, 246, 247, 248, 249, 249, 249, 247, 254, 253, 247, 241, 233, 227, 223, 217, 212, 210, 208, 208, 207, 207,
     207,
     209, 211, 214, 217, 220, 224, 226, 229, 232, 234, 235, 238, 240, 241, 243, 244, 244, ],
    [254, 255, 255, 255, 255, 251, 255, 255, 255, 241, 228, 217, 202, 188, 162, 153, 132, 142, 137, 134, 131, 157,
     157,
     157, 158, 163, 168, 175, 166, 177, 197, 203, 209, 213, 229, 236, 242, 247, 250, 252, ],
    [240, 242, 255, 254, 250, 255, 249, 237, 223, 211, 200, 189, 174, 156, 135, 111, 80, 93, 85, 77, 69, 60, 103,
     103,
     105, 111, 120, 106, 125, 141, 166, 176, 185, 192, 210, 219, 237, 244, 235, 238, ],
    [237, 239, 239, 246, 243, 222, 213, 203, 190, 180, 170, 161, 148, 132, 97, 60, 43, 0, 0, 0, 19, 0, 0, 0, 0, 25,
     57,
     31, 76, 101, 136, 150, 162, 171, 191, 200, 207, 214, 231, 235, ],
    [219, 220, 220, 222, 204, 190, 177, 169, 157, 149, 141, 133, 122, 109, 79, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
     0, 0,
     46, 105, 126, 141, 152, 163, 181, 190, 199, 214, 217, ],
    [189, 189, 188, 183, 160, 147, 141, 130, 124, 118, 111, 105, 96, 87, 62, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0,
     0,
     38, 74, 101, 116, 126, 136, 144, 152, 158, 162, 165, ],
    [140, 141, 150, 137, 123, 110, 106, 101, 91, 87, 82, 78, 72, 66, 49, 37, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,
     52, 79, 94, 103, 111, 118, 126, 131, 135, 139, ],
    [98, 99, 99, 88, 78, 75, 73, 62, 59, 57, 55, 52, 50, 48, 36, 30, 24, 0, 0, 0, 8, 4, 1, 0, 0, 3, 0, 0, 0, 0, 37,
     56,
     67, 73, 71, 75, 87, 91, 93, 95, ]]
WCS_g_mat = [
    [221, 221, 221, 221, 221, 221, 222, 223, 223, 225, 227, 229, 232, 232, 230, 231, 232, 232, 232, 232, 232, 232,
     231,
     231, 230, 229, 228, 227, 226, 225, 225, 224, 223, 223, 222, 222, 222, 221, 221, 221, ],
    [181, 181, 181, 182, 184, 186, 185, 184, 189, 196, 201, 204, 209, 211, 216, 216, 218, 215, 215, 215, 215, 210,
     209,
     209, 208, 207, 205, 203, 203, 200, 197, 196, 194, 193, 187, 185, 183, 182, 181, 181, ],
    [148, 148, 141, 144, 147, 144, 151, 158, 164, 170, 174, 177, 182, 186, 189, 192, 195, 192, 192, 192, 192, 192,
     187,
     186, 185, 183, 181, 181, 178, 174, 170, 167, 165, 163, 155, 153, 144, 142, 148, 148, ],
    [104, 105, 107, 102, 107, 122, 128, 134, 139, 144, 148, 151, 155, 158, 165, 169, 168, 169, 169, 169, 165, 165,
     165,
     164, 162, 160, 157, 157, 152, 148, 143, 139, 136, 133, 124, 121, 119, 116, 106, 105, ],
    [62, 63, 66, 66, 86, 97, 105, 110, 115, 118, 122, 124, 127, 131, 136, 142, 144, 142, 142, 142, 142, 138, 138,
     137,
     135, 133, 131, 130, 127, 123, 116, 110, 106, 102, 98, 87, 82, 78, 64, 62, ],
    [18, 24, 32, 44, 69, 79, 83, 88, 91, 94, 96, 98, 101, 103, 108, 110, 115, 115, 115, 115, 112, 112, 112, 111,
     106,
     105, 105, 103, 100, 97, 88, 84, 79, 75, 71, 68, 64, 61, 59, 58, ],
    [21, 23, 0, 33, 49, 58, 61, 64, 68, 70, 72, 73, 75, 76, 80, 81, 85, 85, 88, 86, 83, 83, 82, 82, 81, 80, 78, 78,
     75,
     72, 61, 57, 51, 47, 43, 38, 32, 28, 25, 22, ],
    [14, 13, 16, 30, 38, 40, 42, 46, 47, 48, 49, 50, 50, 51, 54, 55, 55, 58, 58, 58, 56, 55, 55, 55, 54, 54, 55, 53,
     51,
     49, 39, 38, 34, 30, 35, 34, 22, 19, 17, 15, ],
]
WCS_b_mat = [
    [233, 231, 229, 226, 223, 218, 213, 209, 179, 149, 147, 147, 149, 179, 211, 216, 222, 226, 229, 232, 234, 237,
     241,
     243, 245, 247, 247, 248, 248, 248, 248, 247, 247, 245, 243, 241, 239, 238, 236, 234, ],
    [192, 185, 177, 168, 159, 151, 120, 0, 0, 0, 0, 0, 0, 76, 101, 143, 162, 184, 192, 199, 205, 211, 216, 222, 226,
     230, 233, 235, 253, 254, 236, 234, 233, 230, 234, 228, 220, 212, 205, 198, ],
    [160, 151, 130, 114, 100, 17, 0, 0, 0, 0, 0, 0, 0, 0, 77, 102, 128, 154, 162, 170, 178, 189, 194, 201, 208, 214,
     218, 235, 238, 239, 223, 221, 218, 214, 216, 208, 203, 188, 178, 168, ],
    [124, 111, 93, 52, 0, 29, 0, 0, 0, 0, 0, 0, 0, 7, 4, 58, 105, 124, 134, 143, 153, 164, 173, 183, 192, 200, 205,
     221,
     225, 225, 209, 207, 203, 197, 198, 188, 176, 162, 151, 137, ],
    [97, 78, 58, 0, 0, 0, 10, 0, 21, 10, 0, 0, 11, 30, 20, 30, 74, 99, 109, 118, 129, 139, 147, 157, 165, 173, 177,
     194,
     211, 212, 196, 192, 186, 180, 171, 169, 156, 139, 127, 112, ],
    [75, 55, 36, 0, 20, 31, 19, 36, 30, 25, 23, 22, 28, 36, 25, 45, 60, 76, 85, 94, 103, 114, 123, 132, 130, 135,
     152,
     154, 169, 170, 182, 165, 158, 152, 145, 136, 126, 113, 102, 91, ],
    [58, 46, 25, 20, 18, 27, 20, 12, 34, 33, 32, 32, 35, 40, 30, 41, 48, 58, 63, 71, 78, 85, 92, 99, 105, 110, 114,
     129,
     144, 145, 156, 139, 133, 127, 121, 114, 103, 92, 81, 69, ],
    [49, 40, 31, 28, 31, 25, 20, 36, 34, 34, 35, 36, 38, 40, 28, 35, 40, 40, 45, 49, 54, 58, 62, 66, 69, 72, 88, 90,
     92,
     106, 116, 99, 94, 90, 77, 74, 74, 68, 62, 56, ],
]
WCS_r_arrat, WCS_g_array, WCS_b_array = np.array(
    WCS_r_mat), np.array(WCS_g_mat), np.array(WCS_b_mat)
WCS_rgb = np.concatenate(
    (WCS_r_arrat[:, :, None], WCS_g_array[:, :, None], WCS_b_array[:, :, None]), axis=2)

class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


# class CNNTrainer(BaseTrainer):
#     def __init__(self, model, criterion, num_colors, classifier=None,
#                  alpha=None, beta=None, gamma=None, sample_method=None):
#         super(BaseTrainer, self).__init__()
#         self.model = model
#         self.criterion = criterion
#         self.classifier = classifier
#         self.alpha = alpha
#         self.beta = beta
#         self.gamma = gamma
#         self.reconsturction_loss = nn.MSELoss()
#         self.sample_method = sample_method
#         self.num_colors = num_colors
#         self.convertor = RGB_HSV()
#
#     def HSVDistance(self, hsv_1, hsv_2):
#         H_1, S_1, V_1 = hsv_1.split(1, dim=1)
#         H_2, S_2, V_2 = hsv_2.split(1, dim=1)
#         H_1 = H_1 * 360
#         H_2 = H_2 * 360
#         R = 1
#         angle = 30
#         h = R * math.cos(angle / 180 * math.pi)
#         r = R * math.sin(angle / 180 * math.pi)
#         x1 = r * V_1 * S_1 * torch.cos(H_1 / 180 * torch.pi)
#         y1 = r * V_1 * S_1 * torch.sin(H_1 / 180 * torch.pi)
#         z1 = h * (1 - V_1)
#         x2 = r * V_2 * S_2 * torch.cos(H_2 / 180 * torch.pi)
#         y2 = r * V_2 * S_2 * torch.sin(H_2 / 180 * torch.pi)
#         z2 = h * (1 - V_2)
#         dx = x1 - x2
#         dy = y1 - y2
#         dz = z1 - z2
#         return dx * dx + dy * dy + dz * dz
#
#     def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):
#
#         self.model.train()
#         self.classifier.train()
#         losses = 0
#         correct = 0
#         miss = 0
#         t0 = time.time()
#         for batch_idx, (rgb, hsv, target) in enumerate(data_loader):
#             rgb, hsv, target = rgb.cuda(), hsv.cuda(), target.cuda()
#             optimizer.zero_grad()
#             B, _, H, W = rgb.shape
#             transformed_img, probability_map = self.model(rgb, training=True)
#             rgb_color_palette = (rgb.unsqueeze(2) * probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
#                     probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
#             colorcnn_img = (probability_map.unsqueeze(1) * rgb_color_palette).sum(dim=2)
#             # rgb_color_var = ((color_contribution - color_palette).pow(2) *
#             #              probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
#             #                     probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
#
#             hsv_color_palette = self.convertor.rgb_to_hsv(rgb_color_palette.view(B, 3, self.num_colors, 1)) \
#                 .unsqueeze(-1)
#             hsv_color_contribution = (hsv.unsqueeze(2) * probability_map.unsqueeze(1))
#             hsv_color_var = (self.HSVDistance(hsv_color_contribution, hsv_color_palette) *
#                              probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
#                                     probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
#
#             probability_map_mean = torch.mean(probability_map, dim=[2, 3])
#             Shannon_entropy = -probability_map_mean * torch.log2(1e-8 + probability_map_mean)
#             Shannon_entropy = torch.mean(Shannon_entropy)
#
#             output = self.classifier(transformed_img)
#             pred = torch.argmax(output, 1)
#             correct += pred.eq(target).sum().item()
#             miss += target.shape[0] - pred.eq(target).sum().item()
#
#             loss = self.criterion(output, target) \
#                    + self.alpha * (hsv_color_var.mean()) \
#                    - self.beta * Shannon_entropy \
#                    + self.gamma * self.reconsturction_loss(rgb, colorcnn_img)
#             loss.backward()
#             optimizer.step()
#             losses += loss.item()
#
#             cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))
#
#             if (batch_idx + 1) % log_interval == 0:
#                 t1 = time.time()
#                 t_epoch = t1 - t0
#                 print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
#                     epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))
#
#         t1 = time.time()
#         t_epoch = t1 - t0
#         print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
#             epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))
#
#         return losses / len(data_loader), correct / (correct + miss)
#
#     def test(self, test_loader):
#
#         self.model.eval()
#         self.classifier.eval()
#         losses = 0
#         correct = 0
#         miss = 0
#         t0 = time.time()
#
#         for batch_idx, (rgb, hsv, target) in enumerate(test_loader):
#             rgb, hsv, target = rgb.cuda(), hsv.cuda(), target.cuda()
#             with torch.no_grad():
#                 transformed_img, probability_map = self.model(rgb, training=False)
#                 output = self.classifier(transformed_img)
#             pred = torch.argmax(output, 1)
#             correct += pred.eq(target).sum().item()
#             miss += target.shape[0] - pred.eq(target).sum().item()
#             loss = self.criterion(output, target)
#             losses += loss.item()
#
#         print('Test, Loss: {:.6f}, Prec: {:.1f}%, time: {:.1f}'.format(losses / (len(test_loader) + 1),
#                                                                        100. * correct / (correct + miss),
#                                                                        time.time() - t0))
#
#         return losses / len(test_loader), correct / (correct + miss)


class CNNTrainer_1(BaseTrainer):
    def __init__(self, model, criterion, num_colors, classifier=None,
                 alpha=None, beta=None, gamma=None, sample_method=None):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.classifier = classifier
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reconsturction_loss = nn.L1Loss()
        # self.reconsturction_loss = nn.MSELoss()
        self.sample_method = sample_method
        self.num_colors = num_colors
        self.convertor = RGB_HSV()

    def HSVDistance(self, hsv_1, hsv_2):
        H_1, S_1, V_1 = hsv_1.split(1, dim=1)
        H_2, S_2, V_2 = hsv_2.split(1, dim=1)
        H_1 = H_1 * 360
        H_2 = H_2 * 360
        R = 1
        angle = 30
        h = R * math.cos(angle / 180 * math.pi)
        r = R * math.sin(angle / 180 * math.pi)
        x1 = r * V_1 * S_1 * torch.cos(H_1 / 180 * torch.pi)
        y1 = r * V_1 * S_1 * torch.sin(H_1 / 180 * torch.pi)
        z1 = h * (1 - V_1)
        x2 = r * V_2 * S_2 * torch.cos(H_2 / 180 * torch.pi)
        y2 = r * V_2 * S_2 * torch.sin(H_2 / 180 * torch.pi)
        z2 = h * (1 - V_2)
        dx = x1 - x2
        dy = y1 - y2
        dz = z1 - z2
        return dx * dx + dy * dy + dz * dz

    def train(self, epoch, data_loader, optimizer, log_interval=100, scheduler=None):

        self.model.train()
        self.classifier.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (rgb, target) in enumerate(data_loader):
            rgb, target = rgb.cuda(), target.cuda()

            optimizer.zero_grad()
            B, _, H, W = rgb.shape
            # cluster_ids_list = []
            # for i in range(B):
            #     x = np.uint8(rgb[i,:,:,:].permute(1,2,0).detach().cpu().numpy()*255)
            #     x = Image.fromarray(x)
            #     x = x.quantize(colors=self.num_colors,method=0).convert('P')
            #     cluster_ids,cluster_cneters = kmeans(X=x,num_clusters=self.num_colors,distance='euclidean',device='cuda')
            #     cluster_ids = cluster_ids.reshape(H,W).unsqueeze(0).unsqueeze(0)
            #     cluster_ids_list.append(cluster_ids)
            # cluster_ids_tensor = torch.cat(cluster_ids_list,dim=0)
            # cluster_ids_one_hot = torch.zeros((B,self.num_colors,H,W)).scatter(1, cluster_ids_tensor, 1)
            # A =(probability_map.flatten(2).permute(0,2,1))@ (probability_map.flatten(2))
            # A_star = (cluster_ids_one_hot.flatten(2).permute(0,2,1))@(cluster_ids_one_hot.flatten(2))
            # A = A/torch.norm(A,dim=2,keepdim=True)
            # A_star = A_star/torch.norm(A_star,dim=2,keepdim=True)
            # loss_RP = torch.sum((A-A_star)**2,dim=[1,2])/(H*W)

            transformed_img, probability_map = self.model(rgb, training=True)

            prob_max, _ = torch.max(probability_map.view([B, self.num_colors, -1]), dim=2)
            # prob_max, _ = torch.max(probability_map, dim=1)
            avg_max = torch.mean(prob_max)


            # reconsturction_loss
            rgb_color_palette = (rgb.unsqueeze(2) * probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            # colorcnn_img = (probability_map.unsqueeze(1) * rgb_color_palette).sum(dim=2)
            # hsv_color_var
            hsv = self.convertor.rgb_to_hsv(rgb)
            hsv_color_palette = self.convertor.rgb_to_hsv(rgb_color_palette.view(B, 3, self.num_colors, 1)) \
                .unsqueeze(-1)
            hsv_color_contribution = (hsv.unsqueeze(2) * probability_map.unsqueeze(1))
            hsv_color_var = (self.HSVDistance(hsv_color_contribution, hsv_color_palette) *
                             probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                                    probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)

            # rgb_color_contribution = (rgb.unsqueeze(2) * probability_map.unsqueeze(1))
            # rgb_color_var = ((rgb_color_contribution - rgb_color_palette).pow(2) *
            #              probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
            #                     probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            
            
            # Shannon_entropy
            probability_map_mean = torch.mean(probability_map.view([B, self.num_colors, -1]), dim=2)
            # print('111',probability_map_mean)
            
            Shannon_entropy = -probability_map_mean * torch.log2(torch.tensor([1e-8], device='cuda') + probability_map_mean)
            # torch.tensor([1e-8], device='cuda') + 
            # print('222',Shannon_entropy)
            Shannon_entropy = torch.sum(Shannon_entropy, dim=1)
            
            Shannon_entropy = torch.mean(Shannon_entropy)
            # print(Shannon_entropy)
            output = self.classifier(transformed_img)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()

            loss = self.criterion(output, target) + self.alpha * (
                hsv_color_var.mean()) + self.beta * np.log2(self.num_colors) * (1 - avg_max)+\
                 self.gamma* self.reconsturction_loss(rgb,transformed_img)
                #=  +(np.log2(self.num_colors)-Shannon_entropy)

            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()

            loss.backward()
            optimizer.step()
            losses += loss.item()
            scheduler.step(epoch - 1 + batch_idx / len(data_loader))

            if (batch_idx + 1) % log_interval == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}, lr: {:.5f}, avgmax: {:.5f}, Shannon_entropy: {:.5f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch,
                    optimizer.param_groups[0]['lr'],avg_max.item(),Shannon_entropy.item()))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}, lr: {:.5f}, avgmax: {:.5f}, Shannon_entropy: {:.5f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch,
            optimizer.param_groups[0]['lr'],avg_max.item(),Shannon_entropy.item()))
        # if epoch <= 40:
        #     scheduler.step()
        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader):

        self.model.eval()
        self.classifier.eval()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()

        for batch_idx, (rgb, target) in enumerate(test_loader):
            rgb, target = rgb.cuda(), target.cuda()
            B, _, H, W = rgb.shape
            with torch.no_grad():
                transformed_img, probability_map = self.model(rgb, training=False)
                output = self.classifier(transformed_img)
            # Shannon_entropy
            probability_map_mean = torch.mean(probability_map.view([B, self.num_colors, -1]), dim=2)
            # print('111',probability_map_mean)
            
            Shannon_entropy = -probability_map_mean * torch.log2(torch.tensor([1e-8], device='cuda') + probability_map_mean)
            # torch.tensor([1e-8], device='cuda') + 
            # print('222',Shannon_entropy)
            Shannon_entropy = torch.sum(Shannon_entropy, dim=1)
            
            Shannon_entropy = torch.mean(Shannon_entropy)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()

        
        WCS_chip_size=32
        value_size, hue_size, channel_size = WCS_rgb.shape
        max_index_array = np.zeros((value_size, hue_size))
        for i in range(value_size):
            for j in range(hue_size):
                wcs_pixel = torch.tensor(WCS_rgb[i, j, :] / 255.0, dtype=torch.float32).cuda()
                # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                wcs_pixel_R = (wcs_pixel[0]-0.485)/0.229
                wcs_pixel_G = (wcs_pixel[1]-0.456)/0.224
                wcs_pixel_B = (wcs_pixel[2]-0.406)/0.225
                wcs_pixel = torch.tensor([wcs_pixel_R,wcs_pixel_G,wcs_pixel_B]).cuda()
                wcs_pixel = wcs_pixel.unsqueeze(-1).unsqueeze(-1).repeat(1, WCS_chip_size, WCS_chip_size).unsqueeze(0)
                
                with torch.no_grad():
                    _, probability_map = self.model(wcs_pixel, training=False)
                    probability_map = probability_map.sum(dim=[2, 3], keepdim=False).squeeze()
                    max_index = torch.argmax(probability_map).detach().cpu().item()
                    max_index_array[i, j] = max_index

        print(max_index_array)
        print('Test, Loss: {:.6f}, Prec: {:.1f}%, time: {:.1f}, Shannon_entropy: {:.5f}'.format(losses / (len(test_loader) + 1),
                                                                       100. * correct / (correct + miss),
                                                                       time.time() - t0,Shannon_entropy.item()))
        return losses / len(test_loader), correct / (correct + miss)

    def pretrain(self, epoch, data_loader, optimizer, log_interval=100, scheduler=None):

        self.model.train()
        losses = 0
        t0 = time.time()
        for batch_idx, (rgb, target) in enumerate(data_loader):
            rgb, target = rgb.cuda(), target.cuda()
            optimizer.zero_grad()
            B, _, H, W = rgb.shape
            transformed_img, probability_map = self.model(rgb, training=True)
            # print(torch.mean(probability_map))
            # prob_max, _ = torch.max(probability_map.view([B, self.num_colors, -1]), dim=2)
            prob_max, _ = torch.max(probability_map, dim=1)
            avg_max = torch.mean(prob_max)


            # reconsturction_loss
            rgb_color_palette = (rgb.unsqueeze(2) * probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            colorcnn_img = (probability_map.unsqueeze(1) * rgb_color_palette).sum(dim=2)
            # hsv_color_var
            # hsv = self.convertor.rgb_to_hsv(rgb)
            # hsv_color_palette = self.convertor.rgb_to_hsv(rgb_color_palette.view(B, 3, self.num_colors, 1)) \
            #     .unsqueeze(-1)
            # hsv_color_contribution = (hsv.unsqueeze(2) * probability_map.unsqueeze(1))
            # hsv_color_var = (self.HSVDistance(hsv_color_contribution, hsv_color_palette) *
            #                  probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
            #                         probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            # Shannon_entropy
            probability_map_mean = torch.mean(probability_map.view([B, self.num_colors, -1]), dim=2)
      
            
            Shannon_entropy = -probability_map_mean * torch.log2(torch.tensor([1e-8], device='cuda') + probability_map_mean)
            # torch.tensor([1e-8], device='cuda') + 
            # print('222',Shannon_entropy)
            Shannon_entropy = torch.sum(Shannon_entropy, dim=1)
            
            Shannon_entropy = torch.mean(Shannon_entropy)
            # print(Shannon_entropy)

            loss = self.reconsturction_loss(colorcnn_img,rgb)+np.log2(self.num_colors) * (1 - avg_max)
            ssim = SSIM()
            ssim_score = ssim(colorcnn_img,rgb, as_loss=False)
            
            loss.backward()
            optimizer.step()
            losses += loss.item()
            # cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))

            if (batch_idx + 1) % log_interval == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, SSIM: {:.3f}, Time: {:.3f}, lr: {:.5f}, avgmax: {:.5f}, Shannon_entropy: {:.5f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), ssim_score.mean().item(), t_epoch,
                    optimizer.param_groups[0]['lr'],avg_max.item(),Shannon_entropy.item()))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, Loss: {:.6f}, SSIM: {:.3f}, Time: {:.3f}, lr: {:.5f}, avgmax: {:.5f}, Shannon_entropy: {:.5f}'.format(
            epoch, len(data_loader), losses / len(data_loader), ssim_score.mean().item(), t_epoch,
            optimizer.param_groups[0]['lr'],avg_max.item(),Shannon_entropy.item()))

    def pretrain_test(self, test_loader):

        self.model.eval()
        losses = 0
        ssim_scores = 0
        Shannon_entropys = 0
        t0 = time.time()

        for batch_idx, (rgb, target) in enumerate(test_loader):
            rgb, target = rgb.cuda(), target.cuda()
            B, _, H, W = rgb.shape
            with torch.no_grad():
                transformed_img, probability_map = self.model(rgb, training=False)
            prob_max, _ = torch.max(probability_map, dim=1)
            avg_max = torch.mean(prob_max)
            # Shannon_entropy

            rgb_color_palette = (rgb.unsqueeze(2) * probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            colorcnn_img = (probability_map.unsqueeze(1) * rgb_color_palette).sum(dim=2)

            probability_map_mean = torch.mean(probability_map.view([B, self.num_colors, -1]), dim=2)
            Shannon_entropy = -probability_map_mean * torch.log2(torch.tensor([1e-8], device='cuda') + probability_map_mean)
            Shannon_entropy = torch.sum(Shannon_entropy, dim=1)
            Shannon_entropy = torch.mean(Shannon_entropy)
            ssim = SSIM()
            ssim_score = ssim(colorcnn_img,rgb, as_loss=False) 
            loss = self.reconsturction_loss(colorcnn_img,rgb)+np.log2(self.num_colors) * (1 - avg_max)
            losses += loss.item()
            ssim_scores+= ssim_score.mean().item()
            Shannon_entropys+= Shannon_entropy.item()

        
        WCS_chip_size=32
        value_size, hue_size, channel_size = WCS_rgb.shape
        max_index_array = np.zeros((value_size, hue_size))
        for i in range(value_size):
            for j in range(hue_size):
                wcs_pixel = torch.tensor(WCS_rgb[i, j, :] / 255.0, dtype=torch.float32).cuda()
                # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                wcs_pixel_R = (wcs_pixel[0]-0.485)/0.229
                wcs_pixel_G = (wcs_pixel[1]-0.456)/0.224
                wcs_pixel_B = (wcs_pixel[2]-0.406)/0.225
                wcs_pixel = torch.tensor([wcs_pixel_R,wcs_pixel_G,wcs_pixel_B]).cuda()
                wcs_pixel = wcs_pixel.unsqueeze(-1).unsqueeze(-1).repeat(1, WCS_chip_size, WCS_chip_size).unsqueeze(0)
                
                with torch.no_grad():
                    _, probability_map = self.model(wcs_pixel, training=False)
                    probability_map = probability_map.sum(dim=[2, 3], keepdim=False).squeeze()
                    max_index = torch.argmax(probability_map).detach().cpu().item()
                    max_index_array[i, j] = max_index

        print(max_index_array)
        print('Test, Loss: {:.6f}, SSIM: {:.1f}, time: {:.1f}, Shannon_entropy: {:.5f}'.format(losses / (len(test_loader) + 1),
                                                                       ssim_scores / (len(test_loader) + 1),
                                                                       time.time() - t0,Shannon_entropys/ (len(test_loader) + 1)))