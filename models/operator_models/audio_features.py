import torch
import torch.nn as nn

from einops import rearrange
from .imagen_unet import PerceiverResampler

class ToAcousticEmbedding(nn.Module):
    def __init__(self,
                 cond_dim=128,
                 use_attn_pool=False,
                 **kwargs
            ):
        super().__init__()
        self.cond_dim = cond_dim
        self.use_attn_pool = use_attn_pool
        if use_attn_pool:
            self.attn_pool = PerceiverResampler(dim=cond_dim,
                                                depth = 2,
                                                num_latents = 16,
                                                num_latents_mean_pooled = 4,
                                                )

    def forward(self, emb):
        if self.use_attn_pool:
            emb = self.attn_pool(emb)
        return emb

class ToAcousticToken(nn.Module):
    def __init__(self,
                 cond_dim=128,
                 **kwargs
            ):
        super().__init__()
        self.cond_dim = cond_dim
        self.codebook = nn.Embedding(1024, cond_dim)
        self.attn_pool = PerceiverResampler(dim=cond_dim,
                                            depth = 2,
                                            num_latents = 16,
                                            num_latents_mean_pooled = 4,
                                            )

    def forward(self, audio_codes):
        emb = self.codebook(audio_codes)

        b, q = emb.shape[0], emb.shape[1]
        emb = rearrange(emb, 'b q n c -> (b q) n c')
        emb = self.attn_pool(emb)
        emb = rearrange(emb, '(b q) n c -> b (q n) c', b=b, q=q)
        return emb
