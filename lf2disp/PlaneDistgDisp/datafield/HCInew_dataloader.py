# coding:utf-8
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
import os
import cv2
import csv
from PIL import Image
import random
import imageio
from lf2disp.utils import utils

np.random.seed(160)


class HCInew(Dataset):
    '''
	输入：
	输出： B*s_h*s_w*c
	'''

    def __init__(self, cfg, mode='train'):
        super(HCInew, self).__init__()
        self.datadir = cfg['data']['path']
        self.mode = mode
        self.views = cfg['data']['views']
        if mode == 'train':
            self.imglist = []
            self.batch_size = cfg['training']['image_batch_size']
            with open(os.path.join(self.datadir, 'onlytrain_allviews.txt'), "r") as f:
                for line in f.readlines():
                    imgdir = line.strip("\n")
                    self.imglist.append(os.path.join(self.datadir, imgdir))
            self.number = len(self.imglist)
            self.inputsize = cfg['training']['input_size']
            self.augmentation = cfg['training']['augmentation']
            self.transform = cfg['training']['transform']
            self.traindata_label = np.zeros((len(self.imglist), 512, 512, 9, 9), np.float32)
        elif mode == 'vis':  # val or test
            self.imglist = []
            self.batch_size = cfg['vis']['image_batch_size']
            datafile = os.path.join(self.datadir, 'vis.txt')
            with open(datafile, "r") as f:
                for line in f.readlines():
                    imgdir = line.strip("\n")
                    self.imglist.append(os.path.join(self.datadir, imgdir))
            self.number = len(self.imglist)
            self.inputsize = cfg['vis']['input_size']
            self.transform = cfg['vis']['transform']
            self.traindata_label = np.zeros((len(self.imglist), 512, 512, 9, 9), np.float32)
        elif mode == 'test':  # test / generation
            self.imglist = []
            self.batch_size = cfg['test']['image_batch_size']
            datafile = os.path.join(self.datadir, 'test.txt')
            # datafile = os.path.join(self.datadir, 'test_train.txt')
            with open(datafile, "r") as f:
                for line in f.readlines():
                    imgdir = line.strip("\n")
                    self.imglist.append(os.path.join(self.datadir, imgdir))
            self.number = len(self.imglist)
            self.inputsize = cfg['test']['input_size']
            self.transform = cfg['test']['transform']
            self.traindata_label = np.zeros((len(self.imglist), 512, 512, 9, 9), np.float32)
        elif mode == 'generate':  # test / generation
            self.imglist = []
            self.batch_size = cfg['generation']['image_batch_size']
            datafile = os.path.join(self.datadir, 'generate.txt')
            with open(datafile, "r") as f:
                for line in f.readlines():
                    imgdir = line.strip("\n")
                    self.imglist.append(os.path.join(self.datadir, imgdir))
            self.number = len(self.imglist)
            self.inputsize = cfg['generation']['input_size']
            self.transform = cfg['generation']['transform']
            self.traindata_label = np.zeros((len(self.imglist), 512, 512, 9, 9), np.float32)

        self.invalidpath = []
        with open(os.path.join(self.datadir, 'invalid.txt'), "r") as f:
            for line in f.readlines():
                imgpath = line.strip("\n")
                self.invalidpath.append(os.path.join(self.datadir, imgpath))

        self.traindata_all = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
        self.boolmask_data = np.zeros((len(self.invalidpath), 512, 512), np.float32)
        # 图片预先加载好
        self.imgPreloading()

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        '''
        input：self.img_all: N*MM*H*W*C   self.label_all:N*H*W*1
        output: B, H, W, C,M,M   B, H, W,C
        '''

        if self.mode == 'train':
            image, label = self.train_data()
        else:
            image, label = self.val_data(idx)
        image = np.expand_dims(image, axis=3)
        label = np.expand_dims(label, axis=3)
        out = {'image': np.float32(np.clip(image, 0.0, 1.0)),
               'label': np.float32(label[:, :, :, :, 4, 4]),
               }
        return out

    def imgPreloading(self):  # 图片预加载
        '''
        input: 数据路径
        output： 16x512x512x9x9x3 16x512x512x9x9
        '''

        # 获取光场图像
        for idx in range(0, len(self.imglist)):
            imgdir = self.imglist[idx]
            # Load image
            for i in range(0, self.views ** 2):
                imgname = 'input_Cam' + str(i).zfill(3) + '.png'
                imgpath = os.path.join(imgdir, imgname)
                img = np.uint8(imageio.imread(imgpath))
                self.traindata_all[idx, :, :, i // 9, i - 9 * (i // 9), :] = img

                labelname = 'gt_disp_lowres_Cam' + str(i).zfill(3) + '.pfm'
                labelpath = os.path.join(imgdir, labelname)
                if not os.path.exists(labelpath):
                    labelname = 'gt_disp_lowres.pfm'
                    labelpath = os.path.join(imgdir, labelname)
                    if not os.path.exists(labelpath):
                        imgLabel = np.zeros((512, 512))
                    else:
                        imgLabel = utils.read_pfm(labelpath)
                else:
                    imgLabel = utils.read_pfm(labelpath)
                self.traindata_label[idx, :, :, i // 9, i - 9 * (i // 9)] = imgLabel

        # 获取掩码图像
        for idx in range(0, len(self.invalidpath)):
            boolmask_img = np.float32(imageio.imread(self.invalidpath[idx]))
            boolmask_img = 1.0 * boolmask_img[:, :, 3] > 0
            self.boolmask_data[idx] = boolmask_img
            # 数据预处理 偏移

        # 数据预处理 偏移
        if self.mode == 'train':
            self.traindata_all_add1 = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
            self.traindata_all_sub1 = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
            self.traindata_all_add2 = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
            self.traindata_all_sub2 = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
            self.traindata_all_add3 = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
            self.traindata_all_sub3 = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)

            center = int(self.views / 2)
            for batch_i in range(0, len(self.imglist)):
                for v in range(0, self.views):  # v
                    for u in range(0, self.views):  # u
                        offsety = (center - v)
                        offsetx = (center - u)
                        mat_translation = np.float32([[1, 0, 1 * offsetx], [0, 1, 1 * offsety]])
                        self.traindata_all_add1[batch_i, :, :, v, u, :] = cv2.warpAffine(
                            self.traindata_all[batch_i, :, :, v, u, :],
                            mat_translation,
                            (512, 512))
                        mat_translation = np.float32([[1, 0, -1 * offsetx], [0, 1, -1 * offsety]])
                        self.traindata_all_sub1[batch_i, :, :, v, u, :] = cv2.warpAffine(
                            self.traindata_all[batch_i, :, :, v, u, :],
                            mat_translation,
                            (512, 512))

                        mat_translation = np.float32([[1, 0, 2 * offsetx], [0, 1, 2 * offsety]])
                        self.traindata_all_add2[batch_i, :, :, v, u, :] = cv2.warpAffine(
                            self.traindata_all[batch_i, :, :, v, u, :],
                            mat_translation,
                            (512, 512))
                        mat_translation = np.float32([[1, 0, -2 * offsetx], [0, 1, -2 * offsety]])
                        self.traindata_all_sub2[batch_i, :, :, v, u, :] = cv2.warpAffine(
                            self.traindata_all[batch_i, :, :, v, u, :],
                            mat_translation,
                            (512, 512))

                        mat_translation = np.float32([[1, 0, 3 * offsetx], [0, 1, 3 * offsety]])
                        self.traindata_all_add3[batch_i, :, :, v, u, :] = cv2.warpAffine(
                            self.traindata_all[batch_i, :, :, v, u, :],
                            mat_translation,
                            (512, 512))
                        mat_translation = np.float32([[1, 0, -3 * offsetx], [0, 1, -3 * offsety]])
                        self.traindata_all_sub3[batch_i, :, :, v, u, :] = cv2.warpAffine(
                            self.traindata_all[batch_i, :, :, v, u, :],
                            mat_translation,
                            (512, 512))


        print('imgPreloading', self.boolmask_data.shape, self.traindata_all.shape, self.traindata_label.shape)
        return

    def train_data(self):
        """ initialize image_stack & label """
        self.train_views = 9
        batch_size = self.batch_size
        label_size, input_size = self.inputsize, self.inputsize
        traindata_batch = np.zeros((batch_size, input_size, input_size, self.train_views, self.train_views),
                                   dtype=np.uint8)
        traindata_batch_label = np.zeros((batch_size, label_size, label_size, self.train_views, self.train_views))

        """ inital variable """
        crop_half1 = int(0.5 * (input_size - label_size))

        """ Generate image stacks """
        for ii in range(0, batch_size):
            sum_diff = 0
            valid = 0
            while sum_diff < 0.01 * input_size * input_size or valid < 1:

                """//Variable for gray conversion//"""
                rand_3color = 0.05 + np.random.rand(3)
                rand_3color = rand_3color / np.sum(rand_3color)
                R = rand_3color[0]
                G = rand_3color[1]
                B = rand_3color[2]
                # 图像增强
                aa_arr = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                                   0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                                   0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ])

                image_id = np.random.choice(aa_arr)
                traindata_all, traindata_label = self.choose_delta(image_id)
                # 视角shift增强
                (ix_rd, iy_rd) = self.view_aug()
                # scale增强
                scale = self.scale_aug()

                idx_start = np.random.randint(0, 512 - scale * input_size + 1)  # random_size
                idy_start = np.random.randint(0, 512 - scale * input_size + 1)
                valid = 1
                """
                    boolmask: reflection masks for images(4,6,15)
                """

                # 这是去除高光
                if image_id in [4, 6, 15]:
                    if image_id == 4:
                        a_tmp = self.boolmask_data[0]
                    if image_id == 6:
                        a_tmp = self.boolmask_data[1]
                    if image_id == 15:
                        a_tmp = self.boolmask_data[2]

                    if (np.sum(a_tmp[
                               idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                               idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]) > 0
                            or np.sum(a_tmp[idx_start: idx_start + scale * input_size:scale,
                                      idy_start: idy_start + scale * input_size:scale]) > 0):
                        valid = 0
                if valid > 0:
                    image_center = (1 / 255) * np.squeeze(
                        R * traindata_all[idx_start: idx_start + scale * input_size:scale,
                            idy_start: idy_start + scale * input_size:scale, ix_rd + int(self.train_views / 2),
                            iy_rd + int(self.train_views / 2), 0].astype(
                            'float32') +
                        G * traindata_all[idx_start: idx_start + scale * input_size:scale,
                            idy_start: idy_start + scale * input_size:scale, ix_rd + int(self.train_views / 2),
                            iy_rd + int(self.train_views / 2), 1].astype(
                            'float32') +
                        B * traindata_all[idx_start: idx_start + scale * input_size:scale,
                            idy_start: idy_start + scale * input_size:scale, ix_rd + int(self.train_views / 2),
                            iy_rd + int(self.train_views / 2), 2].astype('float32'))
                    sum_diff = np.sum(
                        np.abs(image_center - np.squeeze(image_center[int(0.5 * input_size), int(0.5 * input_size)])))

                    # 这是转为灰度图了

                    traindata_batch[ii, :, :, :, :] = np.squeeze(
                        R * traindata_all[idx_start: idx_start + scale * input_size:scale,
                            idy_start: idy_start + scale * input_size:scale, ix_rd:ix_rd + self.train_views,
                            iy_rd:iy_rd + self.train_views, 0].astype(
                            'float32') +
                        G * traindata_all[idx_start: idx_start + scale * input_size:scale,
                            idy_start: idy_start + scale * input_size:scale, ix_rd:ix_rd + self.train_views,
                            iy_rd:iy_rd + self.train_views, 1].astype(
                            'float32') +
                        B * traindata_all[idx_start: idx_start + scale * input_size:scale,
                            idy_start: idy_start + scale * input_size:scale, ix_rd:ix_rd + self.train_views,
                            iy_rd:iy_rd + self.train_views, 2].astype(
                            'float32'))

                    if len(traindata_label.shape) == 5:  # 全选
                        traindata_batch_label[ii, :, :] = (1.0 / scale) * traindata_label[
                                                                          idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                                                                          idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale,
                                                                          :, :]
                    else:
                        traindata_batch_label[ii, :, :] = (1.0 / scale) * traindata_label[
                                                                          idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                                                                          idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale,
                                                                          :, :]

        traindata_batch = np.float32((1 / 255) * traindata_batch)
        traindata_batch = np.minimum(np.maximum(traindata_batch, 0), 1)

        # if scale == 1:  # 没有缩放
        #     traindata_batch, traindata_batch_label = self.resize_aug(traindata_batch, traindata_batch_label)

        traindata_batch, traindata_batch_label = self.rotation_aug(traindata_batch, traindata_batch_label)
        traindata_batch = self.noise_aug(traindata_batch)

        return traindata_batch, traindata_batch_label

    def choose_delta(self, image_id):
        # start = time.time()
        trans = np.random.randint(27)
        if trans < 15:  # 0
            traindata_all = self.traindata_all[image_id]
            traindata_label = self.traindata_label[image_id]
        elif trans < 18:
            # 偏移一位
            traindata_all = self.traindata_all_add1[image_id]
            traindata_label = self.traindata_label[image_id] + 1
        elif trans < 21:
            traindata_all = self.traindata_all_sub1[image_id]
            traindata_label = self.traindata_label[image_id] - 1
        elif trans < 23:
            # 偏移一位
            traindata_all = self.traindata_all_add2[image_id]
            traindata_label = self.traindata_label[image_id] + 2
        elif trans < 25:
            traindata_all = self.traindata_all_sub2[image_id]
            traindata_label = self.traindata_label[image_id] - 2
        elif trans < 26:
            traindata_all = self.traindata_all_add3[image_id]
            traindata_label = self.traindata_label[image_id] + 3
        elif trans < 27:
            traindata_all = self.traindata_all_sub3[image_id]
            traindata_label = self.traindata_label[image_id] - 3



        return traindata_all, traindata_label

    def view_aug(self):
        ix_rd, iy_rd = 0, 0
        if self.train_views == 5:
            ix_rd = np.random.randint(0, 5)
            iy_rd = np.random.randint(0, 5)
        if self.train_views == 7:
            ix_rd = np.random.randint(0, 3)
            iy_rd = np.random.randint(0, 3)
        if self.train_views == 9:
            ix_rd = 0
            iy_rd = 0
        return (ix_rd, iy_rd)

    def scale_aug(self):
        # scale aug
        kk = np.random.randint(17)
        if (kk < 8):
            scale = 1
        elif (kk < 14):
            scale = 2
        elif (kk < 17):
            scale = 3

        return scale

    def resize_aug(self, traindata_all, traindata_label):  # 放到后面？
        # opencv resize  放大就行了
        for batch_i in range(self.batch_size):
            data = traindata_all[batch_i]
            label = traindata_label[batch_i]
            resize_rand = np.random.randint(4)
            if resize_rand < 1:  # 增强
                H, W, M, M = data.shape
                size = random.uniform(1.0, 2.0)
                x_dsize, y_dsize = int(H * size), int(W * size)
                idx = np.random.randint(0, x_dsize - H + 1)
                idy = np.random.randint(0, y_dsize - W + 1)
                data = np.transpose(data, (0, 1, 2, 3))
                data = data.reshape(H, W, M * M)
                data = cv2.resize(data, dsize=(x_dsize, y_dsize))[idx:H + idx, idy:W + idy]
                label = np.transpose(label, (0, 1, 2, 3))
                label = label.reshape(H, W, M * M)
                label = (cv2.resize(label, dsize=(x_dsize, y_dsize))[idx:H + idx, idy:W + idy]) * (x_dsize / H)
                traindata_all[batch_i] = data.reshape(H, W, M, M)
                traindata_label[batch_i] = label.reshape(H, W, M, M)
            else:
                pass
        return traindata_all, traindata_label

    def rotation_aug(self, traindata_batch, traindata_label_batchNxN):
        traindata_batch = traindata_batch
        traindata_label_batchNxN = traindata_label_batchNxN

        for batch_i in range(self.batch_size):
            gray_rand = 0.4 * np.random.rand() + 0.8

            traindata_batch[batch_i, :, :, :] = pow(traindata_batch[batch_i, :, :, :], gray_rand)

            """ transpose """
            transp_rand = np.random.randint(0, 2)
            if transp_rand == 1:
                traindata_batch_tmp6 = np.copy(
                    np.rot90(np.transpose(np.squeeze(traindata_batch[batch_i, :, :, :, :]), (1, 0, 2, 3))))
                traindata_batch[batch_i, :, :, :, :] = traindata_batch_tmp6[:, :, ::-1]

                traindata_label_batchNxN_tmp6 = np.copy(
                    np.rot90(np.transpose(np.squeeze(traindata_label_batchNxN[batch_i, :, :, :, :]), (1, 0, 2, 3))))
                traindata_label_batchNxN[batch_i, :, :, :, :] = traindata_label_batchNxN_tmp6[:, :, ::-1]

            """ rotation """
            rotation_rand = np.random.randint(0, 4)
            """ 90 """
            if rotation_rand == 1:
                traindata_batch_tmp6 = np.copy(np.rot90(np.squeeze(traindata_batch[batch_i, :, :, :, :])))
                traindata_batch[batch_i, :, :, :, :] = np.copy(np.rot90(traindata_batch_tmp6, 1, (2, 3)))

                traindata_label_batchNxN_tmp6 = np.copy(
                    np.rot90(np.squeeze(traindata_label_batchNxN[batch_i, :, :, :, :])))
                traindata_label_batchNxN[batch_i, :, :, :, :] = np.copy(
                    np.rot90(traindata_label_batchNxN_tmp6, 1, (2, 3)))

            """ 180 """
            if rotation_rand == 2:
                traindata_batch_tmp6 = np.copy(np.rot90(np.squeeze(traindata_batch[batch_i, :, :, :, :]), 2))
                traindata_batch[batch_i, :, :, :, :] = np.copy(np.rot90(traindata_batch_tmp6, 2, (2, 3)))

                traindata_label_batchNxN_tmp6 = np.copy(
                    np.rot90(np.squeeze(traindata_label_batchNxN[batch_i, :, :, :, :]), 2))
                traindata_label_batchNxN[batch_i, :, :, :, :] = np.copy(
                    np.rot90(traindata_label_batchNxN_tmp6, 2, (2, 3)))

            """ 270 """
            if rotation_rand == 3:
                traindata_batch_tmp6 = np.copy(np.rot90(np.squeeze(traindata_batch[batch_i, :, :, :, :]), 3))
                traindata_batch[batch_i, :, :, :, :] = np.copy(np.rot90(traindata_batch_tmp6, 3, (2, 3)))

                traindata_label_batchNxN_tmp6 = np.copy(
                    np.rot90(np.squeeze(traindata_label_batchNxN[batch_i, :, :, :, :]), 3))
                traindata_label_batchNxN[batch_i, :, :, :, :] = np.copy(
                    np.rot90(traindata_label_batchNxN_tmp6, 3, (2, 3)))

        return traindata_batch, traindata_label_batchNxN

    def noise_aug(self, traindata_batch):
        """ gaussian noise """
        for batch_i in range(self.batch_size):
            noise_rand = np.random.randint(0, 12)
            if noise_rand == 0:
                gauss = np.random.normal(0.0, np.random.uniform() * np.sqrt(0.2), (
                    traindata_batch.shape[1], traindata_batch.shape[2], traindata_batch.shape[3],
                    traindata_batch.shape[4]))
                traindata_batch[batch_i, :, :, :, :] = np.clip(traindata_batch[batch_i, :, :, :, :] + gauss, 0.0, 1.0)
        return traindata_batch

    def val_data(self, idx):
        batch_size = 1
        label_size, input_size = self.inputsize, self.inputsize
        self.test_views = 9

        test_data = np.zeros((batch_size, input_size, input_size, self.test_views, self.test_views),
                             dtype=np.uint8)

        test_data_label = np.zeros((batch_size, label_size, label_size, self.test_views, self.test_views))
        crop_half1 = int(0.5 * (input_size - label_size))

        R = 0.299
        G = 0.587
        B = 0.114
        ix_rd = int((self.views - self.test_views) / 2)
        iy_rd = int((self.views - self.test_views) / 2)
        idx_start = 0
        idy_start = 0

        test_image = self.traindata_all[idx]
        test_label = self.traindata_label[idx]

        test_data[0] = np.squeeze(
            R * test_image[idx_start: idx_start + input_size,
                idy_start: idy_start + input_size, ix_rd:ix_rd + self.test_views, iy_rd:iy_rd + self.test_views,
                0].astype('float32') +
            G * test_image[idx_start: idx_start + input_size,
                idy_start: idy_start + input_size, ix_rd:ix_rd + self.test_views, iy_rd:iy_rd + self.test_views,
                1].astype('float32') +
            B * test_image[idx_start: idx_start + input_size,
                idy_start: idy_start + input_size, ix_rd:ix_rd + self.test_views, iy_rd:iy_rd + self.test_views,
                2].astype('float32'))

        test_data_label[0] = test_label[crop_half1: crop_half1 + label_size,
                             crop_half1:crop_half1 + label_size, ]

        test_data = np.float32((1 / 255) * test_data)
        test_data = np.minimum(np.maximum(test_data, 0), 1)

        return test_data, test_data_label


import math


def testData():
    cfg = {'data': {'path': 'D:/code/LFdepth/LFData/HCInew',
                    'views': 9},
           'training': {'input_size': 512, 'transform': False,
                        'image_batch_size': 1,
                        'augmentation': True,
                        "views": 9},
           'test': {'input_size': 512, 'transform': False,
                    'image_batch_size': 1,
                    'augmentation': True,
                    "views": 9},
           }
    mydataset = HCInew_fast(cfg, mode='test')
    train_loader = DataLoader(mydataset, batch_size=1, shuffle=False)
    for epoch in range(1):
        for i, data in enumerate(train_loader):
            out = data
            print(out['image'].shape, out['label'].shape, type(out['image']))
            B1, B2, H, W, C, M, M = out["image"].shape
            temp = out['image'].reshape(B1 * B2, H, W, 1, M, M).cpu().numpy()[0]
            label = out['label'].reshape(B1 * B2, H, W, 1, M, M).cpu().numpy()[0]

            label = (label - label.min()) / (label.max() - label.min())
            compare = 0.2 * temp + 0.8 * label
            for u in range(0, M):
                for v in range(0, M):
                    cv2.imwrite('./image/' + str(u) + "_" + str(v) + '.png', temp[:, :, :, u, v] * 255)
                    cv2.imwrite('./depth/' + str(u) + "_" + str(v) + '.png', label[:, :, :, u, v] * 255)
                    cv2.imwrite('./compare/' + str(u) + "_" + str(v) + '.png', compare[:, :, :, u, v] * 255)


if __name__ == '__main__':
    testData()
