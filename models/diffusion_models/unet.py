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

from .imagen.imagen_utils import (
    LearnedSinusoidalPosEmb,
    )

class Diffusion2dUnet(nn.Module):
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

        # Diffusion Timestep Embedding
        time_cond_dim = embedding_size
        learned_sinu_pos_emb_dim = 16
        num_time_tokens = 2

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1
        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
            nn.SiLU()
        )
        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )
        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange('b (r d) -> b r d', r = num_time_tokens)
        )
        self.norm_cond = nn.LayerNorm(cond_dim)

        # Unet
        self.model = EfficientUNet(
            unet_channels =     unet_channels,
            cond_dim =          cond_dim,
            unet_dim_mults =    unet_dim_mults,
            channels =          4 if cat_ref else 2,
            channels_out =      2,
            embedding_size =    embedding_size,
            num_resnet_blocks = num_resnet_blocks,
            layer_attns =       layer_attns,
            layer_cross_attns = layer_cross_attns,
            attend_at_middle = True,
            **kwargs)
        
    def forward(self, x_t, t, global_condition=None, local_condition=None):
        # Diffusion TimeStep Embedding
        time_hiddens = self.to_time_hiddens(t)
        time_condition = self.to_time_cond(time_hiddens)
        time_tokens = self.to_time_tokens(time_hiddens)
        
        t = time_condition
        if global_condition is not None : t += global_condition
        c = time_tokens
        if local_condition is not None : 
            c = torch.cat([time_tokens, local_condition], dim=1)
        c = self.norm_cond(c)

        pred = self.model(x_t, t, c)
        return pred

class Diffusion1dUnet(nn.Module):
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

        # Diffusion Timestep Embedding
        time_cond_dim = embedding_size
        learned_sinu_pos_emb_dim = 16
        num_time_tokens = 2

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1
        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
            nn.SiLU()
        )
        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange('b (r d) -> b r d', r = num_time_tokens)
        )

        self.norm_cond = nn.LayerNorm(cond_dim)

        # Unet
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
        
    def forward(self, x_t, t, global_condition=None, local_condition=None):
        # Diffusion TimeStep Embedding
        time_hiddens = self.to_time_hiddens(t)
        time_condition = self.to_time_cond(time_hiddens)
        time_tokens = self.to_time_tokens(time_hiddens)

        t = time_condition
        if global_condition is not None : t += global_condition
        c = time_tokens
        if local_condition is not None : 
            c = torch.cat([time_tokens, local_condition], dim=1)
        c = self.norm_cond(c)

        pred = self.model(x_t, t, c)
        return pred

class DiffusionHybridUnet(nn.Module):
    def __init__(
        self,
        unet_channels,
        cond_dim,
        learnable_sum=True,
        use_ref_encoder=True,
        cat_ref=True,
        cat_global_condition=False,
        norm_local_condition=True,
        hybrid_bridge=False,
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

        self.norm_local_condition = norm_local_condition
        # Diffusion Timestep Embedding
        time_cond_dim = embedding_size
        learned_sinu_pos_emb_dim = 16
        num_time_tokens = 2

        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinu_pos_emb_dim)
        sinu_pos_emb_input_dim = learned_sinu_pos_emb_dim + 1
        self.cat_global_condition = cat_global_condition
        self.to_time_hiddens = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(sinu_pos_emb_input_dim, time_cond_dim),
            nn.SiLU()
        )
        self.to_time_cond = nn.Sequential(
            nn.Linear(time_cond_dim, time_cond_dim)
        )

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange('b (r d) -> b r d', r = num_time_tokens)
        )

        self.norm_cond = nn.LayerNorm(cond_dim)

        if cat_global_condition : embedding_size = embedding_size * 2
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

    def forward(self, x_t_spec, x_t_wav, t, global_condition=None, local_condition=None):
        # Diffusion TimeStep Embedding
        time_hiddens = self.to_time_hiddens(t)
        time_condition = self.to_time_cond(time_hiddens)
        time_tokens = self.to_time_tokens(time_hiddens)

        t = time_condition
        if global_condition is not None :
            if self.cat_global_condition:
                t = torch.cat([time_condition, global_condition], dim=-1)
            else:
                t += global_condition
        c = time_tokens
        if local_condition is not None : 
            c = torch.cat([time_tokens, local_condition], dim=1)
        if self.norm_local_condition: c = self.norm_cond(c)

        pred_spec = self.model2d(x_t_spec, t, c)
        pred_wav = self.model1d(x_t_wav, t, c)
        return pred_spec, pred_wav
