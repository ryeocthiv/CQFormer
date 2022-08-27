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


class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class CNNTrainer(BaseTrainer):
    def __init__(self, model, criterion, num_colors, classifier=None,
                 alpha=None, beta=None, gamma=None, sample_method=None):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.classifier = classifier
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reconsturction_loss = nn.MSELoss()
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

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):

        self.model.train()
        self.classifier.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (rgb, hsv, target) in enumerate(data_loader):
            rgb, hsv, target = rgb.cuda(), hsv.cuda(), target.cuda()
            optimizer.zero_grad()
            B, _, H, W = rgb.shape
            transformed_img, probability_map = self.model(rgb, training=True)
            rgb_color_palette = (rgb.unsqueeze(2) * probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            colorcnn_img = (probability_map.unsqueeze(1) * rgb_color_palette).sum(dim=2)
            # rgb_color_var = ((color_contribution - color_palette).pow(2) *
            #              probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
            #                     probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)

            hsv_color_palette = self.convertor.rgb_to_hsv(rgb_color_palette.view(B, 3, self.num_colors, 1)) \
                .unsqueeze(-1)
            hsv_color_contribution = (hsv.unsqueeze(2) * probability_map.unsqueeze(1))
            hsv_color_var = (self.HSVDistance(hsv_color_contribution, hsv_color_palette) *
                             probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                                    probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)

            probability_map_mean = torch.mean(probability_map, dim=[2, 3])
            Shannon_entropy = -probability_map_mean * torch.log2(1e-8 + probability_map_mean)
            Shannon_entropy = torch.mean(Shannon_entropy)

            output = self.classifier(transformed_img)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()

            loss = self.criterion(output, target) \
                   + self.alpha * (hsv_color_var.mean()) \
                   - self.beta * Shannon_entropy \
                   + self.gamma * self.reconsturction_loss(rgb, colorcnn_img)
            loss.backward()
            optimizer.step()
            losses += loss.item()

            cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))

            if (batch_idx + 1) % log_interval == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))

        return losses / len(data_loader), correct / (correct + miss)

    def test(self, test_loader):

        self.model.eval()
        self.classifier.eval()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()

        for batch_idx, (rgb, hsv, target) in enumerate(test_loader):
            rgb, hsv, target = rgb.cuda(), hsv.cuda(), target.cuda()
            with torch.no_grad():
                transformed_img, probability_map = self.model(rgb, training=False)
                output = self.classifier(transformed_img)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()

        print('Test, Loss: {:.6f}, Prec: {:.1f}%, time: {:.1f}'.format(losses / (len(test_loader) + 1),
                                                                       100. * correct / (correct + miss),
                                                                       time.time() - t0))

        return losses / len(test_loader), correct / (correct + miss)


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
        self.reconsturction_loss = nn.MSELoss()
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

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):

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
            transformed_img, probability_map = self.model(rgb, training=True)
            # reconsturction_loss
            rgb_color_palette = (rgb.unsqueeze(2) * probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                    probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            colorcnn_img = (probability_map.unsqueeze(1) * rgb_color_palette).sum(dim=2)
            # hsv_color_var
            hsv = self.convertor.rgb_to_hsv(rgb)
            hsv_color_palette = self.convertor.rgb_to_hsv(rgb_color_palette.view(B, 3, self.num_colors, 1)) \
                .unsqueeze(-1)
            hsv_color_contribution = (hsv.unsqueeze(2) * probability_map.unsqueeze(1))
            hsv_color_var = (self.HSVDistance(hsv_color_contribution, hsv_color_palette) *
                             probability_map.unsqueeze(1)).sum(dim=[3, 4], keepdim=True) / (
                                    probability_map.unsqueeze(1).sum(dim=[3, 4], keepdim=True) + 1e-8)
            # Shannon_entropy
            probability_map_mean = torch.mean(probability_map, dim=[2, 3])
            Shannon_entropy = -probability_map_mean * torch.log2(1e-8 + probability_map_mean)
            Shannon_entropy = torch.mean(Shannon_entropy)

            output = self.classifier(transformed_img)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()

            loss = self.criterion(output, target) \
                   + self.alpha * (hsv_color_var.mean()) \
                   - self.beta * Shannon_entropy \
                   + self.gamma * self.reconsturction_loss(rgb, colorcnn_img)



            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()

            loss.backward()
            optimizer.step()
            losses += loss.item()

            cyclic_scheduler.step(epoch - 1 + batch_idx / len(data_loader))

            if (batch_idx + 1) % log_interval == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))

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
            with torch.no_grad():
                transformed_img, probability_map = self.model(rgb, training=False)
                output = self.classifier(transformed_img)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            loss = self.criterion(output, target)
            losses += loss.item()

        print('Test, Loss: {:.6f}, Prec: {:.1f}%, time: {:.1f}'.format(losses / (len(test_loader) + 1),
                                                                       100. * correct / (correct + miss),
                                                                       time.time() - t0))

        return losses / len(test_loader), correct / (correct + miss)