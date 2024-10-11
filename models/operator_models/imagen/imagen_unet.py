import math
from random import random
from beartype.typing import List, Union
from beartype import beartype
from tqdm.auto import tqdm
from functools import partial, wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum
import torchvision.transforms as T

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from .imagen_modules import *
from .imagen_utils import *

class EfficientUNet(nn.Module):
    def __init__(
        self,
        *,
        unet_channels,
        cond_dim, # dim for c
        unet_dim_mults=(1, 1, 2, 2, 4),
        channels = 4,
        channels_out = 2,
        # global_condition
        embedding_size = 256, # dim for t
        # resnet
        num_resnet_blocks = (2, 2, 4, 4, 8),
        # attention
        attn_dim_head = 32,
        attn_heads = 16,
        ff_mult = 2.,
        layer_attns = (False, False, False, True, True),
        layer_attns_depth = 1,
        layer_mid_attns_depth = 1,
        attend_at_middle = True,            
        layer_cross_attns = (False, False, False, True, True),
        use_linear_attn = False,
        use_linear_cross_attn = False,
        init_dim = None,
        resnet_groups = 8,
        init_conv_kernel_size = 7,          # kernel size of initial conv, if not using cross embed
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        dropout = 0.,
        memory_efficient = True,
        init_conv_to_final_conv_residual = True,
        use_global_context_attn = True,
        scale_skip_connection = True,
        final_resnet_block = True,
        final_conv_kernel_size = 7,
        combine_upsample_fmaps = False,      # combine feature maps from all upsample blocks, used in unet squared successfully
        pixel_shuffle_upsample = True,       # may address checkboard artifacts
        **kwargs
    ):
        super().__init__()
        assert attn_heads > 1, 'you need to have more than 1 attention head, ideally at least 4 or 8'
        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)
        init_channels = channels
        init_dim = default(init_dim, unet_channels)
        dims = [init_dim, *map(lambda m: unet_channels * m, unet_dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_layers = len(in_out)

        # ----------------------------------------------------------
        # Main Layers
        # ----------------------------------------------------------
        # initial convolution
        self.init_conv = CrossEmbedLayer(init_channels,
                                         dim_out = init_dim,
                                         kernel_sizes = init_cross_embed_kernel_sizes,
                                         stride = 1) if init_cross_embed \
                                         else nn.Conv2d(init_channels, init_dim, init_conv_kernel_size,
                                                        padding = init_conv_kernel_size // 2)

        # attention related params
        attn_kwargs = dict(heads = attn_heads, dim_head = attn_dim_head)

        # resnet block klass
        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers)
        resnet_groups = cast_tuple(resnet_groups, num_layers)
        resnet_klass = partial(ResnetBlock, **attn_kwargs)

        layer_attns = cast_tuple(layer_attns, num_layers)
        layer_attns_depth = cast_tuple(layer_attns_depth, num_layers)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_layers)

        use_linear_attn = cast_tuple(use_linear_attn, num_layers)
        use_linear_cross_attn = cast_tuple(use_linear_cross_attn, num_layers)

        assert all([layers == num_layers for layers in list(map(len, (resnet_groups, layer_attns, layer_cross_attns)))])

        # downsample klass
        downsample_klass = Downsample

        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer, kernel_sizes = cross_embed_downsample_kernel_sizes)

        # initial resnet block (for memory efficient unet)
        self.init_resnet_block = resnet_klass(init_dim, init_dim, 
                                 embedding_size = embedding_size,
                                 groups = resnet_groups[0],
                                 use_gca = use_global_context_attn) if memory_efficient else None

        # scale for resnet skip connections
        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        layer_params = [num_resnet_blocks, 
                        resnet_groups,
                        layer_attns,
                        layer_attns_depth,
                        layer_cross_attns,
                        use_linear_attn,
                        use_linear_cross_attn]
        reversed_layer_params = list(map(reversed, layer_params))

        # downsampling layers
        skip_connect_dims = [] # keep track of skip connection dimensions

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            if layer_attn:
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity

            current_dim = dim_in

            # whether to pre-downsample, from memory efficient unet
            pre_downsample = None

            if memory_efficient:
                pre_downsample = downsample_klass(dim_in, dim_out)
                current_dim = dim_out

            skip_connect_dims.append(current_dim)

            # whether to do post-downsample, for non-memory efficient unet
            post_downsample = None
            if not memory_efficient:
                post_downsample = downsample_klass(current_dim, dim_out) if not is_last else Parallel(nn.Conv2d(dim_in, dim_out, 3, padding = 1), nn.Conv2d(dim_in, dim_out, 1))

            self.downs.append(nn.ModuleList([
                pre_downsample,
                resnet_klass(current_dim, current_dim, 
                             cond_dim = layer_cond_dim,
                             linear_attn = layer_use_linear_cross_attn,
                             embedding_size = embedding_size,
                             groups = groups),
                nn.ModuleList([ResnetBlock(current_dim, current_dim,
                                           embedding_size = embedding_size,
                                           groups = groups,
                                           use_gca = use_global_context_attn)
                                           for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = current_dim,
                                        depth = layer_attn_depth,
                                        ff_mult = ff_mult,
                                        context_dim = cond_dim,
                                        **attn_kwargs),
                post_downsample
            ]))

        # middle layers
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim,
                                      cond_dim = cond_dim,
                                      embedding_size = embedding_size,
                                      groups = resnet_groups[-1])
        self.mid_attn = TransformerBlock(mid_dim, 
                                         depth = layer_mid_attns_depth,
                                         **attn_kwargs) if attend_at_middle else None
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim,
                                      cond_dim = cond_dim,
                                      embedding_size = embedding_size,
                                      groups = resnet_groups[-1])

        # upsample klass
        upsample_klass = Upsample if not pixel_shuffle_upsample else PixelShuffleUpsample

        # upsampling layers
        upsample_fmap_dims = []

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            is_last = ind == (len(in_out) - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None
            if layer_attn:
                transformer_block_klass = TransformerBlock
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock
            else:
                transformer_block_klass = Identity

            skip_connect_dim = skip_connect_dims.pop()
            upsample_fmap_dims.append(dim_out)

            self.ups.append(nn.ModuleList([
                resnet_klass(dim_out + skip_connect_dim, dim_out,
                             cond_dim = layer_cond_dim,
                             linear_attn = layer_use_linear_cross_attn,
                             embedding_size = embedding_size,
                             groups = groups),
                nn.ModuleList([ResnetBlock(dim_out + skip_connect_dim, dim_out,
                                           embedding_size = embedding_size,
                                           groups = groups,
                                           use_gca = use_global_context_attn)
                              for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = dim_out,
                                        depth = layer_attn_depth,
                                        ff_mult = ff_mult,
                                        context_dim = cond_dim, **attn_kwargs),
                upsample_klass(dim_out, dim_in) if not is_last or memory_efficient else Identity()
            ]))

        # whether to combine feature maps from all upsample blocks before final resnet block out
        self.upsample_combiner = UpsampleCombiner(
            dim = unet_channels,
            enabled = combine_upsample_fmaps,
            dim_ins = upsample_fmap_dims,
            dim_outs = unet_channels
        )

        # whether to do a final residual from initial conv to the final resnet block out
        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = self.upsample_combiner.dim_out + (unet_channels if init_conv_to_final_conv_residual else 0)

        # final optional resnet block and convolution out
        self.final_res_block = ResnetBlock(final_conv_dim, unet_channels,
                                           embedding_size = embedding_size,
                                           groups = resnet_groups[0], use_gca = True)\
                                           if final_resnet_block else None
        final_conv_dim_in = unet_channels if final_resnet_block else final_conv_dim

        self.final_conv = nn.Conv2d(final_conv_dim_in, self.channels_out, 
                                    final_conv_kernel_size,
                                    padding = final_conv_kernel_size // 2)
        zero_init_(self.final_conv)

    def forward(self, unet_input, t, c):
        '''
        x : (B, F, T, 2)
        fx_embedding : (B, C=embedding_dim=128, F=4, T=512)
        features : list of features coming out from the fx encoder (B, C, F, T=512)
        '''
        batch_size, device = unet_input.shape[0], unet_input.device

        # ------------------------------------------
        # Unet
        # ------------------------------------------

        # initial convolution
        dry_tar = unet_input[..., :2]

        x = rearrange(unet_input, 'b f t c -> b c f t')
        x = self.init_conv(x)

        # init conv residual
        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()

        # initial resnet block (for memory efficient unet)
        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t)

        # go through the layers of the unet, down and up
        # -----------------------------------
        # UnetDown
        hiddens = []
        for idx, (pre_downsample, init_block, resnet_blocks, attn_block, post_downsample) in enumerate(self.downs):
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, t, c)
            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)
                hiddens.append(x)

            x = attn_block(x, c)
            hiddens.append(x)

            if exists(post_downsample):
                x = post_downsample(x)
        # -----------------------------------
        # Bottleneck
        x = self.mid_block1(x, t, c)
        if exists(self.mid_attn):
            x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        # -----------------------------------
        # UnetUp
        add_skip_connection = lambda x: torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim = 1)
        up_hiddens = []
        for idx, (init_block, resnet_blocks, attn_block, upsample) in enumerate(self.ups):
            x = add_skip_connection(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x, t)

            x = attn_block(x, c)
            up_hiddens.append(x.contiguous())
            x = upsample(x)
        # ----------------------------------
        # whether to combine all feature maps from upsample blocks
        x = self.upsample_combiner(x, up_hiddens)

        # final top-most residual if needed
        if self.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual), dim = 1)

        if exists(self.final_res_block):
            x = self.final_res_block(x, t)

        x = self.final_conv(x)
        x = rearrange(x, 'b c f t -> b f t c')
#         x = x + dry_tar
        return x

class EfficientUNet1d(nn.Module):
    def __init__(
        self,
        *,
        unet_channels,
        cond_dim, # dim for c
        unet_dim_mults=(1, 1, 2, 2, 4),
        channels = 2,
        channels_out = 1,
        # global_condition
        embedding_size = 256, # dim for t
        # resnet
        num_resnet_blocks = (2, 2, 4, 4, 8),
        scale_factors = (8, 4, 4, 2, 2),
        # attention
        attn_dim_head = 32,
        attn_heads = 16,
        ff_mult = 2.,
        layer_attns = (False, False, False, True, True),
        layer_attns_depth = 1,
        layer_mid_attns_depth = 1,
        attend_at_middle = True,            
        layer_cross_attns = (False, False, False, True, True),
        use_linear_attn = False,
        use_linear_cross_attn = False,
        init_dim = None,
        resnet_groups = 8,
        init_conv_kernel_size = 7,          # kernel size of initial conv, if not using cross embed
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        dropout = 0.,
        memory_efficient = True,
        init_conv_to_final_conv_residual = True,
        use_global_context_attn = True,
        scale_skip_connection = True,
        final_resnet_block = True,
        final_conv_kernel_size = 7,
        combine_upsample_fmaps = False,      # combine feature maps from all upsample blocks, used in unet squared successfully
        pixel_shuffle_upsample = False,       # may address checkboard artifacts
        **kwargs
    ):
        super().__init__()
        assert attn_heads > 1, 'you need to have more than 1 attention head, ideally at least 4 or 8'
        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)
        init_channels = channels
        init_dim = default(init_dim, unet_channels)
        dims = [init_dim, *map(lambda m: unet_channels * m, unet_dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_layers = len(in_out)

        # Downsample
        self.scale_factors = scale_factors

        # ----------------------------------------------------------
        # Main Layers
        # ----------------------------------------------------------
        # initial convolution
        self.init_conv = CrossEmbedLayer1d(init_channels,
                                         dim_out = init_dim,
                                         kernel_sizes = init_cross_embed_kernel_sizes,
                                         stride = 1) if init_cross_embed \
                                         else nn.Conv1d(init_channels, init_dim, init_conv_kernel_size,
                                                        padding = init_conv_kernel_size // 2)

        # attention related params
        attn_kwargs = dict(heads = attn_heads, dim_head = attn_dim_head)

        # resnet block klass
        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_layers)
        resnet_groups = cast_tuple(resnet_groups, num_layers)
        resnet_klass = partial(ResnetBlock1d, **attn_kwargs)

        layer_attns = cast_tuple(layer_attns, num_layers)
        layer_attns_depth = cast_tuple(layer_attns_depth, num_layers)
        layer_cross_attns = cast_tuple(layer_cross_attns, num_layers)

        use_linear_attn = cast_tuple(use_linear_attn, num_layers)
        use_linear_cross_attn = cast_tuple(use_linear_cross_attn, num_layers)

        assert all([layers == num_layers for layers in list(map(len, (resnet_groups, layer_attns, layer_cross_attns)))])

        # downsample klass
        downsample_klass = Downsample1d

        if cross_embed_downsample:
            downsample_klass = partial(CrossEmbedLayer1d, kernel_sizes = cross_embed_downsample_kernel_sizes)

        # initial resnet block (for memory efficient unet)
        self.init_resnet_block = resnet_klass(init_dim, init_dim, 
                                 embedding_size = embedding_size,
                                 groups = resnet_groups[0],
                                 use_gca = use_global_context_attn) if memory_efficient else None

        # scale for resnet skip connections
        self.skip_connect_scale = 1. if not scale_skip_connection else (2 ** -0.5)

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        layer_params = [num_resnet_blocks, 
                        resnet_groups,
                        layer_attns,
                        layer_attns_depth,
                        layer_cross_attns,
                        use_linear_attn,
                        use_linear_cross_attn]
        reversed_layer_params = list(map(reversed, layer_params))

        # downsampling layers
        skip_connect_dims = [] # keep track of skip connection dimensions

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(in_out, *layer_params)):
            is_last = ind >= (num_resolutions - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None

            if layer_attn:
                transformer_block_klass = TransformerBlock1d
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock1d
            else:
                transformer_block_klass = Identity

            current_dim = dim_in

            # whether to pre-downsample, from memory efficient unet
            pre_downsample = None

            if memory_efficient:
                pre_downsample = downsample_klass(dim_in, dim_out, scale_factors[ind])
                current_dim = dim_out

            skip_connect_dims.append(current_dim)

            # whether to do post-downsample, for non-memory efficient unet
            post_downsample = None
            if not memory_efficient:
                post_downsample = downsample_klass(current_dim, dim_out, scale_factors[ind]) if not is_last else Parallel(nn.Conv2d(dim_in, dim_out, 3, padding = 1), nn.Conv2d(dim_in, dim_out, 1))

            self.downs.append(nn.ModuleList([
                pre_downsample,
                resnet_klass(current_dim, current_dim, 
                             cond_dim = layer_cond_dim,
                             linear_attn = layer_use_linear_cross_attn,
                             embedding_size = embedding_size,
                             groups = groups),
                nn.ModuleList([ResnetBlock1d(current_dim, current_dim,
                                           embedding_size = embedding_size,
                                           groups = groups,
                                           use_gca = use_global_context_attn)
                                           for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = current_dim,
                                        depth = layer_attn_depth,
                                        ff_mult = ff_mult,
                                        context_dim = cond_dim,
                                        **attn_kwargs),
                post_downsample
            ]))

        # middle layers
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock1d(mid_dim, mid_dim,
                                      cond_dim = cond_dim,
                                      embedding_size = embedding_size,
                                      groups = resnet_groups[-1])
        self.mid_attn = TransformerBlock1d(mid_dim, 
                                         depth = layer_mid_attns_depth,
                                         **attn_kwargs) if attend_at_middle else None
        self.mid_block2 = ResnetBlock1d(mid_dim, mid_dim,
                                      cond_dim = cond_dim,
                                      embedding_size = embedding_size,
                                      groups = resnet_groups[-1])

        # upsample klass
        upsample_klass = Upsample1d if not pixel_shuffle_upsample else PixelShuffleUpsample1d

        # upsampling layers
        upsample_fmap_dims = []

        for ind, ((dim_in, dim_out), layer_num_resnet_blocks, groups, layer_attn, layer_attn_depth, layer_cross_attn, layer_use_linear_attn, layer_use_linear_cross_attn) in enumerate(zip(reversed(in_out), *reversed_layer_params)):
            is_last = ind == (len(in_out) - 1)

            layer_cond_dim = cond_dim if layer_cross_attn or layer_use_linear_cross_attn else None
            if layer_attn:
                transformer_block_klass = TransformerBlock1d
            elif layer_use_linear_attn:
                transformer_block_klass = LinearAttentionTransformerBlock1d
            else:
                transformer_block_klass = Identity

            skip_connect_dim = skip_connect_dims.pop()
            upsample_fmap_dims.append(dim_out)

            self.ups.append(nn.ModuleList([
                resnet_klass(dim_out + skip_connect_dim, dim_out,
                             cond_dim = layer_cond_dim,
                             linear_attn = layer_use_linear_cross_attn,
                             embedding_size = embedding_size,
                             groups = groups),
                nn.ModuleList([ResnetBlock1d(dim_out + skip_connect_dim, dim_out,
                                           embedding_size = embedding_size,
                                           groups = groups,
                                           use_gca = use_global_context_attn)
                              for _ in range(layer_num_resnet_blocks)]),
                transformer_block_klass(dim = dim_out,
                                        depth = layer_attn_depth,
                                        ff_mult = ff_mult,
                                        context_dim = cond_dim, **attn_kwargs),
                upsample_klass(dim_out, dim_in, list(reversed(scale_factors))[ind]) if not is_last or memory_efficient else Identity()
            ]))

        # whether to combine feature maps from all upsample blocks before final resnet block out
        self.upsample_combiner = UpsampleCombiner1d(
            dim = unet_channels,
            enabled = combine_upsample_fmaps,
            dim_ins = upsample_fmap_dims,
            dim_outs = unet_channels
        )

        # whether to do a final residual from initial conv to the final resnet block out
        self.init_conv_to_final_conv_residual = init_conv_to_final_conv_residual
        final_conv_dim = self.upsample_combiner.dim_out + (unet_channels if init_conv_to_final_conv_residual else 0)

        # final optional resnet block and convolution out
        self.final_res_block = ResnetBlock1d(final_conv_dim, unet_channels,
                                           embedding_size = embedding_size,
                                           groups = resnet_groups[0], use_gca = True)\
                                           if final_resnet_block else None
        final_conv_dim_in = unet_channels if final_resnet_block else final_conv_dim

        self.final_conv = nn.Conv1d(final_conv_dim_in, self.channels_out, 
                                    final_conv_kernel_size,
                                    padding = final_conv_kernel_size // 2)
        zero_init_(self.final_conv)

    def forward(self, unet_input, t, c):
        '''
        x : (B, C, T)
        fx_embedding : (B, C=embedding_dim=128, F=4, T=512)
        features : list of features coming out from the fx encoder (B, C, F, T=512)
        '''
        batch_size, device = unet_input.shape[0], unet_input.device

        # ------------------------------------------
        # Unet
        # ------------------------------------------

        # initial convolution
        x = unet_input
        x = self.init_conv(x)

        # init conv residual
        if self.init_conv_to_final_conv_residual:
            init_conv_residual = x.clone()

        # initial resnet block (for memory efficient unet)
        if exists(self.init_resnet_block):
            x = self.init_resnet_block(x, t)


        # go through the layers of the unet, down and up
        # -----------------------------------
        # UnetDown
        hiddens = []
        for idx, (pre_downsample, init_block, resnet_blocks, attn_block, post_downsample) in enumerate(self.downs):
            
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, t, c)
            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)
                hiddens.append(x)

            x = attn_block(x, c)
            hiddens.append(x)

            if exists(post_downsample):
                x = post_downsample(x)
        # -----------------------------------
        # Bottleneck
        x = self.mid_block1(x, t, c)
        if exists(self.mid_attn):
            x = self.mid_attn(x)
        x = self.mid_block2(x, t, c)

        # -----------------------------------
        # UnetUp
        add_skip_connection = lambda x: torch.cat((x, hiddens.pop() * self.skip_connect_scale), dim = 1)
        up_hiddens = []
        for idx, (init_block, resnet_blocks, attn_block, upsample) in enumerate(self.ups):
            x = add_skip_connection(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x, t)

            x = attn_block(x, c)
            up_hiddens.append(x.contiguous())
            x = upsample(x)
        # ----------------------------------
        # whether to combine all feature maps from upsample blocks
        x = self.upsample_combiner(x, up_hiddens)

        # final top-most residual if needed
        if self.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual), dim = 1)

        if exists(self.final_res_block):
            x = self.final_res_block(x, t)

        x = self.final_conv(x)
        x = rearrange(x, 'b 1 t -> b t')
        return x

class EfficientUNetHybrid(nn.Module):
    def __init__(
        self,
        use_ref_encoder=True,
        cat_ref=True,
        bridge_first=True,
        *,
        unet_channels,
        cond_dim, # dim for c
        unet_dim_mults=(1, 1, 2, 2, 4),
        # global_condition
        embedding_size = 256, # dim for t
        # resnet
        num_resnet_blocks = (2, 2, 4, 4, 8),
        scale_factors = (8, 4, 4, 2, 2),
        # attention
        attn_dim_head = 32,
        attn_heads = 16,
        ff_mult = 2.,
        layer_attns = (False, False, False, True, True),
        layer_attns_depth = 1,
        layer_mid_attns_depth = 1,
        attend_at_middle = True,            
        layer_cross_attns = (False, False, False, True, True),
        use_linear_attn = False,
        use_linear_cross_attn = False,
        init_dim = None,
        resnet_groups = 8,
        init_conv_kernel_size = 7,          # kernel size of initial conv, if not using cross embed
        init_cross_embed = True,
        init_cross_embed_kernel_sizes = (3, 7, 15),
        cross_embed_downsample = False,
        cross_embed_downsample_kernel_sizes = (2, 4),
        dropout = 0.,
        memory_efficient = True,
        init_conv_to_final_conv_residual = True,
        use_global_context_attn = True,
        scale_skip_connection = True,
        final_resnet_block = True,
        final_conv_kernel_size = 7,
        combine_upsample_fmaps = False,      # combine feature maps from all upsample blocks, used in unet squared successfully
        pixel_shuffle_upsample = False,       # may address checkboard artifacts
        **kwargs
    ):
        super().__init__()
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

        bottleneck_dim = unet_dim_mults[-1] * unet_channels
        self.bridge_first = bridge_first
        self.bridge_2d_to_1d = CrossAttention(dim=bottleneck_dim,
                                               context_dim=bottleneck_dim,
                                               dim_head=64,
                                               heads=8,
                                               norm_context=False,
                                               scale=8)

        self.bridge_1d_to_2d  = CrossAttention(dim=bottleneck_dim,
                                               context_dim=bottleneck_dim,
                                               dim_head=64,
                                               heads=8,
                                               norm_context=False,
                                               scale=8)

    def forward(self, x_2d, x_1d, t, c):
        batch_size, device = x_2d.shape, x_2d.device

        # ------------------------------------------
        # Down2d
        # ------------------------------------------

        # initial convolution
        x = rearrange(x_2d, 'b f t c -> b c f t')
        x = self.model2d.init_conv(x)

        # init conv residual
        if self.model2d.init_conv_to_final_conv_residual:
            init_conv_residual_2d = x.clone()

        # initial resnet block (for memory efficient unet)
        if exists(self.model2d.init_resnet_block):
            x = self.model2d.init_resnet_block(x, t)

        # UnetDown
        hiddens_2d = []
        for idx, (pre_downsample, init_block, resnet_blocks, attn_block, post_downsample) in enumerate(self.model2d.downs):
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, t, c)
            for resnet_block in resnet_blocks:
                x = resnet_block(x, t)
                hiddens_2d.append(x)

            x = attn_block(x, c)
            hiddens_2d.append(x)

            if exists(post_downsample):
                x = post_downsample(x)
        # -----------------------------------

        # ------------------------------------------
        # Down1d
        # ------------------------------------------

        # initial convolution
        y = self.model1d.init_conv(x_1d)

        # init conv residual
        if self.model1d.init_conv_to_final_conv_residual:
            init_conv_residual_1d = y.clone()

        # initial resnet block (for memory efficient unet)
        if exists(self.model1d.init_resnet_block):
            y = self.model1d.init_resnet_block(y, t)


        # go through the layers of the unet, down and up
        # -----------------------------------
        # UnetDown
        hiddens_1d = []
        for idx, (pre_downsample, init_block, resnet_blocks, attn_block, post_downsample) in enumerate(self.model1d.downs):
            
            if exists(pre_downsample):
                y = pre_downsample(y)

            y = init_block(y, t, c)
            for resnet_block in resnet_blocks:
                y = resnet_block(y, t)
                hiddens_1d.append(y)

            y = attn_block(y, c)
            hiddens_1d.append(y)

            if exists(post_downsample):
                y = post_downsample(y)

        # Bottleneck
        # -----------------------------------
        x = self.model2d.mid_block1(x, t, c)
        y = self.model1d.mid_block1(y, t, c)

        F, T = x.shape[-2], x.shape[-1]
        if self.bridge_first:
            x = rearrange(x, 'b c f t -> b (f t) c')
            y = rearrange(y, 'b c t -> b t c')

            x_bridge = self.bridge_1d_to_2d(x, context=y)
            y_bridge = self.bridge_2d_to_1d(y, context=x)

            x = rearrange(x_bridge, 'b (f t) c -> b c f t', f=F, t=T)
            y = rearrange(y_bridge, 'b t c -> b c t')

            if exists(self.model2d.mid_attn):
                x = self.model2d.mid_attn(x)

            if exists(self.model1d.mid_attn):
                y = self.model1d.mid_attn(y)
        else:
            if exists(self.model2d.mid_attn):
                x = self.model2d.mid_attn(x)

            if exists(self.model1d.mid_attn):
                y = self.model1d.mid_attn(y)
            x = rearrange(x, 'b c f t -> b (f t) c')
            y = rearrange(y, 'b c t -> b t c')

            x_bridge = self.bridge_1d_to_2d(x, context=y)
            y_bridge = self.bridge_2d_to_1d(y, context=x)

            x = rearrange(x_bridge, 'b (f t) c -> b c f t', f=F, t=T)
            y = rearrange(y_bridge, 'b t c -> b c t')


        x = self.model2d.mid_block2(x, t, c)
        y = self.model1d.mid_block2(y, t, c)

        # -----------------------------------
        # UnetUp2d
        add_skip_connection = lambda x: torch.cat((x, hiddens_2d.pop() * self.model2d.skip_connect_scale), dim = 1)
        up_hiddens_2d = []
        for idx, (init_block, resnet_blocks, attn_block, upsample) in enumerate(self.model2d.ups):
            x = add_skip_connection(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = add_skip_connection(x)
                x = resnet_block(x, t)

            x = attn_block(x, c)
            up_hiddens_2d.append(x.contiguous())
            x = upsample(x)
        # ----------------------------------
        # whether to combine all feature maps from upsample blocks
        x = self.model2d.upsample_combiner(x, up_hiddens_2d)

        # final top-most residual if needed
        if self.model2d.init_conv_to_final_conv_residual:
            x = torch.cat((x, init_conv_residual_2d), dim = 1)

        if exists(self.model2d.final_res_block):
            x = self.model2d.final_res_block(x, t)

        x = self.model2d.final_conv(x)
        x = rearrange(x, 'b c f t -> b f t c')

        
        # -----------------------------------
        # UnetUp1d
        add_skip_connection = lambda y: torch.cat((y, hiddens_1d.pop() * self.model1d.skip_connect_scale), dim = 1)
        up_hiddens_1d = []
        for idy, (init_block, resnet_blocks, attn_block, upsample) in enumerate(self.model1d.ups):
            y = add_skip_connection(y)
            y = init_block(y, t, c)

            for resnet_block in resnet_blocks:
                y = add_skip_connection(y)
                y = resnet_block(y, t)

            y = attn_block(y, c)
            up_hiddens_1d.append(y.contiguous())
            y = upsample(y)
        # ----------------------------------
        # whether to combine all feature maps from upsample blocks
        y = self.model1d.upsample_combiner(y, up_hiddens_1d)

        # final top-most residual if needed
        if self.model1d.init_conv_to_final_conv_residual:
            y = torch.cat((y, init_conv_residual_1d), dim = 1)

        if exists(self.model1d.final_res_block):
            y = self.model1d.final_res_block(y, t)

        y = self.model1d.final_conv(y)
        y = rearrange(y, 'b 1 t -> b t')

        return x, y
