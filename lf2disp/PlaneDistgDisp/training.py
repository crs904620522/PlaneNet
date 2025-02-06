import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from lf2disp.training import BaseTrainer
import torch.nn as nn
import cv2
import math
from lf2disp.utils.utils import depth_metric
import numpy as np
import random
from scipy.stats import truncnorm


class Trainer(BaseTrainer):

    def __init__(self, model, optimizer, criterion=nn.MSELoss, device=None, cfg=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion()
        self.vis_dir = cfg['vis']['vis_dir']
        self.test_dir = cfg['test']['test_dir']

        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        print("use model:", self.model)
        print("use loss:", self.criterion)

    def train_step(self, data, iter=0):
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, iter)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def eval_step(self, data, imgid=0):
        device = self.device
        self.model.eval()
        image = data.get('image').to(device)
        label = data.get('label').to(device)
        B1, B2, H, W, C, M, M = image.shape

        image = image.reshape(B1*B2,H,W,C,M,M)
        label = label.reshape(B1 * B2, H, W)

        with torch.no_grad():
            depthmap = self.model(image)['init_disp']
        depthmap = depthmap.cpu().numpy().reshape(B1 * B2, H, W, 1)[0][15:-15, 15:-15]
        label = label.cpu().numpy().reshape(B1 * B2, H, W, 1)[0][15:-15, 15:-15]
        metric = depth_metric(label, depthmap)
        metric['id'] = imgid
        return metric

    def visualize(self, data, id=0, vis_dir=None):
        self.model.eval()
        device = self.device
        if vis_dir is None:
            vis_dir = self.vis_dir

        image = data.get('image').to(device)
        label = data.get('label').to(device)
        B1, B2, H, W, C, M, M = image.shape

        image = image.reshape(B1*B2,H,W,C,M,M)
        label = label.reshape(B1 * B2, H, W)
        with torch.no_grad():
            depthmap = self.model(image)['init_disp']

        depthmap = depthmap.cpu().numpy().reshape(B1 * B2, H, W, 1)[0][15:-15, 15:-15]
        label = label.cpu().numpy().reshape(B1 * B2, H, W, 1)[0][15:-15, 15:-15]

        depthmap = (depthmap - label.min()) / (label.max() - label.min())
        label = (label - label.min()) / (label.max() - label.min())

        path = os.path.join(vis_dir, str(id) + '_.png')
        labelpath = os.path.join(vis_dir, '%03d_label.png' % id)

        cv2.imwrite(path, depthmap.copy() * 255.0)
        print('save depth map in', path)
        cv2.imwrite(labelpath, label.copy() * 255.0)
        print('save label in', labelpath)

    def compute_loss(self, data, iter=0):
        device = self.device
        image = data.get('image').to(device)
        B1, B2, H, W, C, M, M = image.shape

        label = data.get('label').reshape(B1 * B2, H, W).to(device)
        image = image.reshape(B1*B2,H,W,C,M,M)

        if iter <50000:
            out = self.model(image,label)
            depthmap = out['init_disp'].reshape(B1 * B2, H, W)
            label = label.reshape(B1 * B2, H, W)
        else:
            out = self.model(image,label)
            depthmap = out['init_disp'].reshape(B1 * B2, H, W)
            label = out['gt'].reshape(B1 * B2, H, W)

        loss_d = self.criterion(depthmap.reshape(-1), label.reshape(-1)).mean()

        loss = loss_d

        return loss
