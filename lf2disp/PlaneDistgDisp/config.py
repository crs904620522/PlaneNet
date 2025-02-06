import torch
import torch.distributions as dist
from torch import nn
import os
from lf2disp.PlaneDistgDisp import models, training, generation
from lf2disp.PlaneDistgDisp.datafield.HCInew_dataloader import HCInew

Datadict = {
    'HCInew': HCInew,
}


def get_model(cfg, dataset=None, device=None):
    model = models.Net(cfg, device=device)
    return model


def get_dataset(mode, cfg):

    type = cfg['data']['dataset']
    dataset = Datadict[type](cfg, mode=mode)
    return dataset


def get_trainer(model, optimizer, cfg, criterion, device, **kwargs):

    trainer = training.Trainer(
        model, optimizer,
        device=device,
        criterion=criterion,
        cfg=cfg,
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''

    generator = generation.GeneratorDepth(
        model,
        device=device,
        cfg=cfg,
    )
    return generator
