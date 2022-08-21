import time
import numpy as np
import torch
import torch.nn as nn
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

    def train(self, epoch, data_loader, optimizer, log_interval=100, cyclic_scheduler=None):

        self.model.train()
        self.classifier.train()
        losses = 0
        correct = 0
        miss = 0
        t0 = time.time()
        for batch_idx, (rgb, hsv_s_channel, target) in enumerate(data_loader):
            rgb, hsv_s_channel, target = rgb.cuda(), hsv_s_channel.cuda(), target.cuda()
            optimizer.zero_grad()

            transformed_img,probability_map = self.model(rgb, hsv_s_channel, training=True)
            output = self.classifier(transformed_img)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()

            loss = self.criterion(output, target)
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

        for batch_idx, (rgb, hsv_s_channel, target) in enumerate(test_loader):
            rgb, hsv_s_channel, target = rgb.cuda(), hsv_s_channel.cuda(), target.cuda()
            with torch.no_grad():
                transformed_img,probability_map = self.model(rgb, hsv_s_channel, training=False)
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
