import torch
import torch.nn as nn

from einops import rearrange
from .mftaa.asa import AxialSoftAttention
from .mftaa.erb import Banks
from .mftaa.f_sampling import FreqDownsampling
from .mftaa.phase_encoder import PhaseEncoder
from .mftaa.stft import STFT
from .mftaa.tfcm import TFConvModule
from .imagen.imagen_modules import PerceiverResampler

class MTFAAEncoder(nn.Module):
    def __init__(self,
                 phase_encoder_channel=4,
                 encoder_channels=64,
                 channel_mult=(1, 2, 4),
                 ds_factors=(4, 4, 4),
                 with_t_attn=True,
                 tfcm_layer=6,
                 nerb=256,
                 sr=44100,
                 causal=False,
                 win_len=2046,
                 win_hop=512,
                 **kwargs
                 ):
        super().__init__()
        self.sr = sr

        encoder_channels = [encoder_channels * c for c in channel_mult]
        channels = [phase_encoder_channel // 2] + encoder_channels  # [2, 32, 64, 128, 256]

        self.PE = PhaseEncoder(phase_encoder_channel, 1)
        self.stft = STFT(win_len, win_hop, win_len, 'hann')
        self.ERB = Banks(nerb, win_len, self.sr)

        self.encoder_fd = nn.ModuleList()
        self.encoder_bn = nn.ModuleList()

        # Encoder
        for idx in range(len(channels) - 1):
            self.encoder_fd.append(
                FreqDownsampling(channels[idx], channels[idx + 1], downsample_factor=ds_factors[idx]),
            )
            self.encoder_bn.append(
                nn.Sequential(
                    TFConvModule(channels[idx + 1], (3, 3), tfcm_layer=tfcm_layer, causal=causal),
                    AxialSoftAttention(channels[idx + 1], causal=causal, with_t_attn=with_t_attn),
                )
            )

    def forward(self, sig):
        spec = self.stft.transform(sig, pad_input=True)
        out = self.ERB.amp2bank(self.PE([spec]))
        for idx in range(len(self.encoder_fd)):
            out = self.encoder_fd[idx](out)
            out = self.encoder_bn[idx](out)

#         fx_latent = self.out(out)
        fx_latent = out
        return fx_latent

class ToAfxToken(nn.Module):
    def __init__(self,
                 cond_dim,
                 num_time_pool=12,
                 # attn_pool
                 use_attn_pool=False,
                 attn_pool_num_latents=16,
                 attn_pool_num_latents_mean_pooled=4,
                 **kwargs
            ):
        super().__init__()
        self.cond_dim = cond_dim
        self.num_time_pool = num_time_pool
        self.use_attn_pool = use_attn_pool
#         self.conv1x1 = nn.Conv2d(256, cond_dim, 1)

        if use_attn_pool:
            self.attn_pool = PerceiverResampler(dim=cond_dim,
                                                depth=2,
                                                num_latents=attn_pool_num_latents,
                                                num_latents_mean_pooled=attn_pool_num_latents_mean_pooled,
                                                max_seq_len=256
                                                )
        else:
            self.mean_pool = nn.AdaptiveAvgPool2d((None, num_time_pool))

    def forward(self, fx_latent):
        # x : (B x C x F x T)
#         fx_latent = self.conv1x1(fx_latent)
        if self.use_attn_pool:
            b, f = fx_latent.shape[0], fx_latent.shape[2]
            x = rearrange(fx_latent, 'b c f t -> (b f) t c')
            x = self.attn_pool(x)
            x = rearrange(x, '(b f) t c -> b (f t) c', b=b, f=f)
        else:
            x = self.mean_pool(fx_latent)
            x = rearrange(x, 'b c f t -> b (f t) c')
        return x # (B x 8*16 x C=128)

class ToAfxEmbedding(nn.Module):
    def __init__(self,
                 cond_dim,
                 embedding_size,
                 partial_mean_pooled=False,
                 **kwargs,
            ):
        super().__init__()
        self.embedding_size = embedding_size
        self.partial_mean_pooled = partial_mean_pooled
        if partial_mean_pooled:
            self.mean_pooling = nn.AdaptiveAvgPool2d((4, 4))
            cond_dim = cond_dim * 16
        self.to_fx_embedding = nn.Sequential(
#                 nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, embedding_size),
                nn.SiLU(),
                nn.Linear(embedding_size, embedding_size)
                )

    def forward(self, fx_latent):
        if self.partial_mean_pooled:
            mean_pooled = self.mean_pooling(fx_latent)
            mean_pooled = rearrange(mean_pooled, 'b c f t -> b (c f t)')
        else:
            mean_pooled = torch.mean(fx_latent, dim=(2, 3))
        emb = self.to_fx_embedding(mean_pooled)
        return emb
