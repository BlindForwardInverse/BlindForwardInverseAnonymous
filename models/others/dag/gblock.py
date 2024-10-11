import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dag.film import FiLM, FourierFeature


class GBlock(nn.Module):
    def __init__(self, c_in, c_out, stride, down=False, embedding_size=256):
        super().__init__()
        self.stride = stride
        self.down = down

        # Four film layers
        film_layers = [
            FiLM(hidden_size=c, embedding_size=embedding_size)
            for c in [c_in, c_out, c_out, c_out]
        ]
        # Four conv layers
        conv_layers = [nn.Conv1d(c_out, c_out, 3, 1, 1) for _ in range(3)]
        if down:
            updown_conv = nn.Conv1d(c_in, c_out, stride, stride)
        else:
            updown_conv = nn.ConvTranspose1d(c_in, c_out, stride, stride)
        conv_layers.insert(0, updown_conv)

        self.layers = nn.ModuleList(
            [
                FiLMActConv(film_layers[i], nn.LeakyReLU(), conv_layers[i])
                for i in range(4)
            ]
        )

        # Resampling Residual conv (1x1)
        self.res_conv = nn.Conv1d(c_in, c_out, 1)

    def forward(self, x, sigma_emb):
        # First Half
        # Residual Path
        res = self.res_conv(x)
        if self.down:
            res = F.avg_pool1d(res, kernel_size=self.stride, stride=self.stride)
        else:
            res = F.interpolate(res, scale_factor=self.stride)

        # Main Path
        for i in range(2):
            x = self.layers[i](x, sigma_emb)
        x = x + res

        # Second Half
        h = self.layers[2](x, sigma_emb)
        h = self.layers[3](h, sigma_emb)
        x = x + h
        return x


class FiLMActConv(nn.Module):
    def __init__(self, film, act, conv):
        super().__init__()
        self.film = film
        self.act = act
        self.conv = conv

    def forward(self, x, sigma_emb):
        x = self.film(x, sigma_emb)
        x = self.act(x)
        x = self.conv(x)
        return x
