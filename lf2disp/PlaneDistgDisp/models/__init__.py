import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
from einops import rearrange
class Net(nn.Module):
    def __init__(self, cfg, device=None):
        super(Net, self).__init__()
        feaC = 16
        channel = 160
        mindisp, maxdisp = -4, 4
        angRes = 9
        self.angRes = angRes
        self.init_feature = nn.Sequential(
            nn.Conv2d(1, feaC, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feaC, feaC, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            SpaResB(feaC, angRes),
            nn.Conv2d(feaC,  feaC, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(feaC),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(feaC, feaC, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            ).to(device)

        self.build_costvolume = BuildCostVolume(feaC, channel, angRes, mindisp, maxdisp).to(device)

        self.aggregation = nn.Sequential(
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channel),
            nn.LeakyReLU(0.1, inplace=True),
            ResB3D(channel),
            ResB3D(channel),
            nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channel, 1, kernel_size=3, stride=1, padding=1, bias=False),
        ).to(device)
        self.regression = Regression(mindisp, maxdisp).to(device)

        self.shift_flag=True
        self.shift_random= 7

    def forward(self, x, gt=None):
        self.shift_flag=True
        x = rearrange(x,'b h w c u v -> b c (h u) (w v)',u=self.angRes, v=self.angRes)
        x, gt =self.PlaneRegularSampling(x,gt)
        init_feat = self.init_feature(x)
        init_feat, gt =self.PlaneRegularSampling(init_feat,gt)
        cost = self.build_costvolume(init_feat)
        cost = self.aggregation(cost)
        init_disp = self.regression(cost.squeeze(1))
        self.shift_flag=False
        out = {'init_disp': init_disp,
               'gt':gt,
               }
        return out

    def PlaneRegularSampling(self,feats,gt):
        if (gt==None) | (not self.shift_flag): 
            return feats, gt
        pixel_random = np.random.randint(self.shift_random)
        if pixel_random!=0: #
            return feats, gt
        # self.shift_flag = False #
        B,C,HU,WV = feats.shape

        feats = rearrange(feats, 'b c (h u) (w v) -> (b u v) c h w',u=self.angRes,v=self.angRes)
        H, W = int(HU/self.angRes), int(WV/self.angRes)
        offsetx = (np.random.rand() - 0.5) *2 
        offsety = (np.random.rand() - 0.5) *2
        rate = [offsetx / H, offsety / W]
        theta = torch.tensor([[1, 0, rate[0]], [0, 1, rate[1]]], dtype=float).to(feats.device)
        grid = F.affine_grid(theta.unsqueeze(0).repeat(feats.size(0), 1, 1), feats.size()).type_as(feats)
        feats = F.grid_sample(feats, grid)#,mode='bicubic')

        gt = gt.reshape(B,1,H,W)
        grid = F.affine_grid(theta.unsqueeze(0).repeat(gt.size(0), 1, 1), gt.size()).type_as(gt)
        gt = F.grid_sample(gt, grid)#,mode='bicubic')

        feats = rearrange(feats,'(b u v) c h w -> b c (h u) (w v)',u=self.angRes,v=self.angRes)

        return feats, gt


class BuildCostVolume(nn.Module):
    def __init__(self, channel_in, channel_out, angRes, mindisp, maxdisp):
        super(BuildCostVolume, self).__init__()
        self.DSAFE = nn.Conv2d(channel_in, channel_out, angRes, stride=angRes, padding=0, bias=False)
        self.angRes = angRes
        self.mindisp = mindisp
        self.maxdisp = maxdisp

    def forward(self, x):
        cost_list = []
        for d in range(self.mindisp, self.maxdisp + 1):
            if d < 0:
                dilat = int(abs(d) * self.angRes + 1)
                pad = int(0.5 * self.angRes * (self.angRes - 1) * abs(d))
            if d == 0:
                dilat = 1
                pad = 0
            if d > 0:
                dilat = int(abs(d) * self.angRes - 1)
                pad = int(0.5 * self.angRes * (self.angRes - 1) * abs(d) - self.angRes + 1)
            cost = F.conv2d(x, weight=self.DSAFE.weight, stride=self.angRes, dilation=dilat, padding=pad)
            cost_list.append(cost)
        cost_volume = torch.stack(cost_list, dim=2)

        return cost_volume


class Regression(nn.Module):
    def __init__(self, mindisp, maxdisp):
        super(Regression, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.maxdisp = maxdisp
        self.mindisp = mindisp

    def forward(self, cost):
        score = self.softmax(cost)              # B, D, H, W
        temp = torch.zeros(score.shape).to(score.device)            # B, D, H, W
        for d in range(self.maxdisp - self.mindisp + 1):
            temp[:, d, :, :] = score[:, d, :, :] * (self.mindisp + d)
        disp = torch.sum(temp, dim=1, keepdim=True)     # B, 1, H, W

        return disp


class SpaResB(nn.Module):
    def __init__(self, channels, angRes):
        super(SpaResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=angRes, padding=angRes, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        buffer = self.body(x)
        return buffer + x


class ResB3D(nn.Module):
    def __init__(self, channels):
        super(ResB3D, self).__init__()
        self.body = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels),
        )

    def forward(self, x):
        buffer = self.body(x)
        return buffer + x



if __name__ == "__main__":
    net = Net(angRes=9).cuda()
    angRes = 9
    from thop import profile
    input = torch.randn(1, 32,32,1,9,9).cuda()
    gt = torch.randn(1, 1, 32, 32).cuda()
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (params / 1e6))
    print('   Number of FLOPs: %.5fT' % (flops/ 1e12))

    for i in range(70):
        with torch.no_grad():
            net(input,gt)



