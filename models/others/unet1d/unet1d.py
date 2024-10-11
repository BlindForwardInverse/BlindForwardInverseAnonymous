import torch
import torch.nn as nn
from diffusers import UNet1dModel

class unet_diffusers(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = UNet1dModel(sample_size=40000,
                                 sample_rate=16000,
                                 in_channels=1,
                                 out_channels=1,
                                 time_embedding_type='fourier',
                )
