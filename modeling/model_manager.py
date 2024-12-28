
from .resnet import resnet18, resnet50, resnet101
from .nets import *
import torch
import timm

def get_net(cfg):
    if cfg.ALGORITHM == 'ERM' or cfg.ALGORITHM == 'GDRNet':
        net = get_backbone(cfg)
    elif cfg.ALGORITHM == 'GREEN':
        net = SoftLabelGCN(cfg)
    elif cfg.ALGORITHM == 'CABNet':
        net = CABNet(cfg)
    elif cfg.ALGORITHM == 'MixupNet':
        net = MixupNet(cfg)
    elif cfg.ALGORITHM == 'MixStyleNet':
        net = MixStyleNet(cfg)
    elif cfg.ALGORITHM == 'Fishr' or cfg.ALGORITHM == 'DRGen':
        net = FishrNet(cfg)
    else:
        raise ValueError('Wrong type')
    return net

def get_backbone(cfg):
    if cfg.BACKBONE == 'resnet18':
        model = resnet18(pretrained=True)
    elif cfg.BACKBONE == 'resnet50':
        model = resnet50(pretrained=True)
    elif cfg.BACKBONE == 'resnet101':
        model = resnet101(pretrained=True)
    elif cfg.BACKBONE == 'swint':
        model = timm.create_model('swin_base_patch4_window7_224_in22k', pretrained=True)
    elif cfg.BACKBONE == 'swint_iw':
        model = timm.create_model('swin_base_patch4_window7_224_in22k_iw', pretrained=True)
    else:
        raise ValueError('Wrong type')
    return model

def get_classifier(out_feature_size, cfg):
    return torch.nn.Linear(out_feature_size, cfg.DATASET.NUM_CLASSES)