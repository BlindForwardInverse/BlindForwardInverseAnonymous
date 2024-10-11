import math
from random import random
from beartype.typing import List, Union
from beartype import beartype
from tqdm.auto import tqdm
from functools import partial, wraps
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, einsum
import torchvision.transforms as T

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from .imagen.imagen_unet import (
    EfficientUNet,
    EfficientUNet1d,
    EfficientUNetHybrid
    )

from utils.audio_transform import spec_transform, audio_transform

class UnetWrapper(nn.Module):
    def __init__(
        self,
        unet_channels,
        cond_dim,
        use_ref_encoder=True,
        cat_ref=True,
        unet_dim_mults=(1, 1, 2, 2, 4),
        num_resnet_blocks=(2, 2, 4, 4, 8),
        embedding_size=512,
        num_layer_attns=2,
        num_layer_cross_attns=2,
        **kwargs
        ):
        super().__init__()
        num_layers = len(unet_dim_mults)
        num_layer_cross_attns = 0
        layer_attns = [False] * (num_layers - num_layer_attns) + [True] * num_layer_attns
        layer_cross_attns = [False] * (num_layers - num_layer_cross_attns) + [True] * num_layer_cross_attns

        self.model = EfficientUNet(
            unet_channels =     unet_channels,
            cond_dim =          cond_dim if use_ref_encoder else None,
            unet_dim_mults =    unet_dim_mults,
            channels =          4 if cat_ref else 2,
            channels_out =      2,
            embedding_size =    embedding_size if use_ref_encoder else None,
            num_resnet_blocks = num_resnet_blocks,
            layer_attns =       layer_attns,
            layer_cross_attns = layer_cross_attns,
            attend_at_middle =  True if use_ref_encoder else False,
            **kwargs)
        
    def forward(self, dry_tar, t=None, condition=None):
        pred = self.model(dry_tar, t, condition)
        return pred

class UnetWrapper1d(nn.Module):
    def __init__(
        self,
        unet_channels,
        cond_dim,
        use_ref_encoder=True,
        cat_ref=True,
        unet_dim_mults=(1, 1, 2, 2, 4),
        num_resnet_blocks=(2, 2, 4, 4, 8),
        embedding_size=512,
        num_layer_attns=2,
        num_layer_cross_attns=2,
        **kwargs
        ):
        super().__init__()
        num_layers = len(unet_dim_mults)
        num_layer_cross_attns = 0
        layer_attns = [False] * (num_layers - num_layer_attns) + [True] * num_layer_attns
        layer_cross_attns = [False] * (num_layers - num_layer_cross_attns) + [True] * num_layer_cross_attns

        self.model = EfficientUNet1d(
            unet_channels =     unet_channels,
            cond_dim =          cond_dim if use_ref_encoder else None,
            unet_dim_mults =    unet_dim_mults,
            channels =          2 if cat_ref else 1,
            channels_out =      1,
            embedding_size =    embedding_size if use_ref_encoder else None,
            num_resnet_blocks = num_resnet_blocks,
            layer_attns =       layer_attns,
            layer_cross_attns = layer_cross_attns,
            attend_at_middle =  True if use_ref_encoder else False,
            **kwargs)
        
    def forward(self, dry_tar, t=None, condition=None):
        pred = self.model(dry_tar, t, condition)
        return pred

class UnetWrapperHybrid(nn.Module):
    def __init__(
        self,
        unet_channels,
        cond_dim,
        learnable_sum=True,
        use_ref_encoder=True,
        cat_ref=True,
        hybrid_bridge=True,
        unet_dim_mults=(1, 1, 2, 2, 4),
        num_resnet_blocks=(2, 2, 4, 4, 8),
        embedding_size=512,
        num_layer_attns=2,
        num_layer_cross_attns=2,
        **kwargs
        ):
        super().__init__()
        num_layers = len(unet_dim_mults)
        num_layer_cross_attns = 0
        layer_attns = [False] * (num_layers - num_layer_attns) + [True] * num_layer_attns
        layer_cross_attns = [False] * (num_layers - num_layer_cross_attns) + [True] * num_layer_cross_attns

        self.hybrid_bridge = hybrid_bridge # interaction in the bottleneck
        if hybrid_bridge:
            self.model_hybrid = EfficientUNetHybrid(
                use_ref_encoder = use_ref_encoder,
                cat_ref = cat_ref,
                unet_channels = unet_channels,
                cond_dim = cond_dim,
                unet_dim_mults = unet_dim_mults,
                embedding_size = embedding_size,
                num_resnet_blocks = num_resnet_blocks,
                layer_attns = layer_attns,
                layer_cross_attns = layer_cross_attns,
                )
            
        self.model1d = EfficientUNet1d(
            unet_channels =     unet_channels,
            cond_dim =          cond_dim if use_ref_encoder else None,
            unet_dim_mults =    unet_dim_mults,
            channels =          2 if cat_ref else 1,
            channels_out =      1,
            embedding_size =    embedding_size if use_ref_encoder else None,
            num_resnet_blocks = num_resnet_blocks,
            layer_attns =       layer_attns,
            layer_cross_attns = layer_cross_attns,
            attend_at_middle =  True if use_ref_encoder else False,
            **kwargs)

        self.model2d = EfficientUNet(
            unet_channels =     unet_channels,
            cond_dim =          cond_dim if use_ref_encoder else None,
            unet_dim_mults =    unet_dim_mults,
            channels =          4 if cat_ref else 2,
            channels_out =      2,
            embedding_size =    embedding_size if use_ref_encoder else None,
            num_resnet_blocks = num_resnet_blocks,
            layer_attns =       layer_attns,
            layer_cross_attns = layer_cross_attns,
            attend_at_middle =  True if use_ref_encoder else False,
            **kwargs)
        
    def forward(self, dry_tar, dry_tar_wav, t=None, condition=None):
        if self.hybrid_bridge:
            pred_spec, pred_wav = self.model_hybrid(dry_tar, dry_tar_wav, t, condition)
        else:
            pred_spec = self.model2d(dry_tar, t, condition)
            pred_wav = self.model1d(dry_tar_wav, t, condition)
        return pred_spec, pred_wav

class UnetWrapperHybridWithWeight(nn.Module):
    def __init__(
        self,
        unet_channels,
        cond_dim,
        learnable_sum=True,
        use_ref_encoder=True,
        cat_ref=True,
        hybrid_bridge=True,
        unet_dim_mults=(1, 1, 2, 2, 4),
        num_resnet_blocks=(2, 2, 4, 4, 8),
        embedding_size=512,
        num_layer_attns=2,
        num_layer_cross_attns=2,
        **kwargs
        ):
        super().__init__()
        num_layers = len(unet_dim_mults)
        num_layer_cross_attns = 0
        layer_attns = [False] * (num_layers - num_layer_attns) + [True] * num_layer_attns
        layer_cross_attns = [False] * (num_layers - num_layer_cross_attns) + [True] * num_layer_cross_attns

        self.hybrid_bridge = hybrid_bridge # interaction in the bottleneck
        if hybrid_bridge:
            self.model_hybrid = EfficientUNetHybrid(
                use_ref_encoder = use_ref_encoder,
                cat_ref = cat_ref,
                unet_channels = unet_channels,
                cond_dim = cond_dim,
                unet_dim_mults = unet_dim_mults,
                embedding_size = embedding_size,
                num_resnet_blocks = num_resnet_blocks,
                layer_attns = layer_attns,
                layer_cross_attns = layer_cross_attns,
                )
            
        self.model1d = EfficientUNet1d(
            unet_channels =     64,
            cond_dim =          cond_dim if use_ref_encoder else None,
            unet_dim_mults =    unet_dim_mults,
            channels =          2 if cat_ref else 1,
            channels_out =      1,
            embedding_size =    embedding_size if use_ref_encoder else None,
            num_resnet_blocks = num_resnet_blocks,
            layer_attns =       layer_attns,
            layer_cross_attns = layer_cross_attns,
            attend_at_middle =  True if use_ref_encoder else False,
            **kwargs)

        self.model2d = EfficientUNet(
            unet_channels =     unet_channels,
            cond_dim =          cond_dim if use_ref_encoder else None,
            unet_dim_mults =    unet_dim_mults,
            channels =          4 if cat_ref else 2,
            channels_out =      2,
            embedding_size =    embedding_size if use_ref_encoder else None,
            num_resnet_blocks = num_resnet_blocks,
            layer_attns =       layer_attns,
            layer_cross_attns = layer_cross_attns,
            attend_at_middle =  True if use_ref_encoder else False,
            **kwargs)
        
        self.weight = nn.Sequential(nn.Linear(embedding_size, embedding_size),
                                    nn.SiLU(),
                                    nn.Linear(embedding_size, 2))

        self.spec_to_wav = partial(spec_transform, transform_type='unpower_ri', 
                                   n_fft=2046, win_length=2046, hop_length=512,
                                   length=97792)

    def forward(self, dry_tar, dry_tar_wav, t=None, condition=None):
        if self.hybrid_bridge:
            pred_spec, pred_wav = self.model_hybrid(dry_tar, dry_tar_wav, t, condition)
        else:
            pred_spec = self.model2d(dry_tar, t, condition)
            pred_wav = self.model1d(dry_tar_wav, t, condition)
        weight = self.weight(t)
        summed_wav = rearrange(weight[:, 0], 'b -> b 1') * pred_wav +\
                     rearrange(weight[:, 1], 'b -> b 1') * self.spec_to_wav(pred_spec)
        return summed_wav
