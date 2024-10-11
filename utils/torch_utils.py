import functools
import os

import numpy as np
import torch
import random
from einops import rearrange
from omegaconf import OmegaConf

def reshape_x(x, reshape_len=2):
    if reshape_len == 2:
        x = rearrange(x, 'b -> b 1')
    elif reshape_len == 4:
        x = rearrange(x, 'b -> b 1 1 1')
    return x

def seed_everything(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)

def get_numpy(x):
    if type(x) == torch.Tensor:
        return x.detach().cpu().numpy()
    else:
        return np.array(x)

def load_pretrained_model(solver_class, config_path, config_name, freeze=True, device='cuda'):
    '''
    Load checkpoint (.ckpt) for pl.LightningModule
    '''
    print("----------------------------")
    print("Pretrained Model is Loading ")
    print("Config_name :", config_name)

    config = OmegaConf.to_object(OmegaConf.load(config_path))[config_name]
    ckpt, model_configs = config['ckpt'], config['model_configs']
    if ckpt is not None:
        pretrained = solver_class.load_from_checkpoint(ckpt, **model_configs)
    else:
        print("no corresponding ckpt")
    if freeze: freeze_params(pretrained)
    pretrained.eval()
    pretrained.to(device)

    print(f"Succesfully Imported")
    print("\n")
    return pretrained

def freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False
