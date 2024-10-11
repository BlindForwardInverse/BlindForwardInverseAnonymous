import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dag.gblock import GBlock
from models.dag.film import FourierFeature
from einops import rearrange


class DAG(nn.Module):
    def __init__(
        self,
        c_in=1,
        c_out=1,
        stride=[2, 2, 3, 3, 5],
        channel_mult=[1, 2, 2, 4, 8],
        base_channel=128,
        embedding_size=256,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.stride = stride

        self.channels = [base_channel * c for c in channel_mult]
        c_in_down = [c_in] + self.channels[:-1]
        c_out_down = self.channels[:]
        c_in_up = [c for c in reversed(self.channels[:])]
        c_out_up = [c for c in reversed([c_out] + self.channels[:-1])]

        self.fourier_feature = FourierFeature(self.embedding_size)
        down_layers = [
            GBlock(c_in, c_out, s, down=True, embedding_size=embedding_size)
            for c_in, c_out, s in zip(c_in_down, c_out_down, stride)
        ]
        up_layers = [
            GBlock(c_in, c_out, s, down=False, embedding_size=embedding_size)
            for c_in, c_out, s in zip(c_in_up, c_out_up, reversed(stride))
        ]

        self.down_layers = nn.ModuleList(down_layers)
        self.up_layers = nn.ModuleList(up_layers)
        self.middle_layer = nn.GRU(
            input_size=base_channel * channel_mult[-1],
            hidden_size=base_channel * channel_mult[-1] // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, sigma_t):
        x = rearrange(x, "b t -> b 1 t")
        sigma_emb = self.fourier_feature(sigma_t)
        skip_connection = []
        for layer in self.down_layers:
            x = layer(x, sigma_emb)
            skip_connection.append(x)

        x = rearrange(x, "b c t -> b t c")
        x = self.middle_layer(x)[0]
        x = rearrange(x, "b t c -> b c t")

        for layer in self.up_layers:
            skip = skip_connection.pop()
            x = x + skip
            x = layer(x, sigma_emb)

        x = rearrange(x, "b 1 t -> b t")
        return x
