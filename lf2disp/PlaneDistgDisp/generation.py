import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
import time
import os
import cv2
from lf2disp.utils.utils import depth_metric, write_pfm


class GeneratorDepth(object):

    def __init__(self, model, cfg=None, device=None):
        self.model = model.to(device)
        self.device = device
        self.generate_dir = cfg['generation']['generation_dir']
        self.name = cfg['generation']['name']

        if not os.path.exists(self.generate_dir):
            os.makedirs(self.generate_dir)

    def generate_depth(self, data, id=0):
        ''' Generates the output depthmap
        '''
        self.model.eval()
        device = self.device
        image = data.get('image').to(device)
        label = data.get('label').to(device)
        B1, B2, H, W, C, M, M = image.shape

        image = image.reshape(B1*B2,H,W,C,M,M)
        label = label.reshape(B1 * B2, H, W)
        with torch.no_grad():
            depthmap = self.model(image)['init_disp']
        depthmap = depthmap.cpu().numpy().reshape(B1 * B2, H, W, 1)[0][15:-15, 15:-15]
        label = label.cpu().numpy().reshape(B1 * B2, H, W, 1)[0][15:-15, 15:-15]
        depth_fix = np.zeros((512, 512, 1), dtype=float)
        depth_fix[15:-15,15:-15] = depthmap
        pfm_path = os.path.join(self.generate_dir, self.name[id] + '.pfm')
        write_pfm(depth_fix, pfm_path, scale=1.0)

        metric = depth_metric(depthmap,label)
        print(metric)

        return metric
