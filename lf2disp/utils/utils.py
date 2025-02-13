# -*- coding: utf-8 -*-
"""
@Time: 2021/9/11 13:14
@Auth: Rongshan Chen
@File: utils.py
@IDE:PyCharm
@Motto: Happy coding, Thick hair
@Email: 904620522@qq.com
"""
import numpy as np


def read_pfm(fpath, expected_identifier="Pf",print_limit=30):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    def _get_next_line(f):
        next_line = f.readline().decode('utf-8').rstrip()
        # ignore comments
        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line

    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data

import sys
def write_pfm(data, fpath, scale=1, file_identifier=b'Pf', dtype="float32"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as file:
        file.write((file_identifier))
        file.write(('\n%d %d\n' % (width, height)).encode())
        file.write(('%d\n' % scale).encode())
        file.write(values)
    file.close()

def cal_mse(label, pre):  # 计算mse 输入为numpy数组

    mae = np.abs(label - pre)
    mean_mse = 100 * np.average(np.square(mae))
    return mean_mse


def cal_mae(label, pre):  # 计算mae 输入为numpy数组
    mae = np.abs(label - pre)
    mean_mae = 100 * np.average(mae)
    return mean_mae


def cal_bp(label, pre):
    mae = np.abs(label - pre)
    bp1 = 100 * np.average((mae >= 0.01))
    bp3 = 100 * np.average((mae >= 0.03))
    bp7 = 100 * np.average((mae >= 0.07))
    return bp1, bp3, bp7


def depth_metric(label, pre):
    metric = {}
    metric['mae'] = cal_mae(label, pre)
    metric['mse'] = cal_mse(label, pre)
    metric['bp1'], metric['bp3'], metric['bp7'] = cal_bp(label, pre)
    return metric
