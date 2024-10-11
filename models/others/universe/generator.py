import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from audio_utils.audio_transform import audio_transform
from models.universe.convblocks import GeneratorDown, GeneratorUp
from models.universe.film import FourierFeature
from models.universe.module_util import zero_module


class UniverseUnet(nn.Module):
    def __init__(
        self,
        channel_mult=[1, 2, 4, 8, 16, 16],
        base_channel=64,
        stride=[2, 4, 4, 4, 4],
        embedding_size=512,
    ):
        super().__init__()
        self.fourier_feature = FourierFeature(embedding_size=embedding_size)
        self.stem = nn.Sequential(
            nn.ReflectionPad1d(7),
            nn.Conv1d(1, base_channel, kernel_size=15),
        )

        c_in_list = [base_channel * c for c in channel_mult[:-1]]
        c_out_list = [base_channel * c for c in channel_mult[1:]]

        self.down_cnn = nn.ModuleList(
            [
                GeneratorDown(c_in, c_out, k, embedding_size)
                for c_in, c_out, k in zip(c_in_list, c_out_list, stride)
            ]
        )
        self.core = nn.GRU(
            c_in_list[-1],
            c_in_list[-1] // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.up_cnn = nn.ModuleList(
            [
                GeneratorUp(c_in, c_out, k, embedding_size)
                for c_in, c_out, k in zip(
                    reversed(c_out_list), reversed(c_in_list), reversed(stride)
                )
            ]
        )
        # Affine transformation for conditioning signal with RFF

        rev_c_in = [c for c in reversed(c_in_list)]
        self.affine_g = nn.ModuleList(
            [nn.Linear(embedding_size, 2 * c) for c in c_in_list + rev_c_in]
        )

        self.final = nn.Sequential(
            nn.PReLU(),
            zero_module(nn.Conv1d(base_channel, 1, kernel_size=1)),
        )

    def forward(self, x, sigma_t):
        if len(x.shape) == 2:  # [B, T]
            x = rearrange(x, "b t -> b 1 t")
        elif len(x.shape) == 4:  # [B, T, F, C]
            if x.shape[-1] == 3:
                # Magnitude may be included in the 3rd channel
                x_mag = x[:, :, :, 2]
                x = x_mag
            else:
                # Compute magnitude
                x_real = x[:, :, :, 0]  # [B, T, F]
                x_imag = x[:, :, :, 1]  # [B, T, F]
                x_mag = torch.sqrt(x_real**2 + x_imag**2)  # [64, 256, 353]

                # Magnitude averaged over freqs
                x_mag_averaged = x_mag.mean(dim=2).unsqueeze(1)  # [B, 1, T]
                x = x_mag_averaged  # [64, 1, 256]

        sigma_emb = self.fourier_feature(sigma_t)  # [64, 512]
        x = self.stem(x)

        skip_connections = list()
        for i in range(5):
            x, skip = self.down_cnn[i](x, sigma_emb)  ##
            skip_connections.append(skip)

        x = x.transpose(1, 2).contiguous()
        x = self.core(x)[0]  # GRU without residual
        x = x.transpose(1, 2).contiguous()

        for i in range(5):
            skip = skip_connections.pop()
            x = self.up_cnn[i](x, sigma_emb, skip)

        x = self.final(x)
        x = rearrange(x, "b 1 t -> b t")
        return x


if __name__ == "__main__":
    from conditioner import Conditioner

    x = torch.randn(5, 1, 512 * 19)
    mel = torch.randn(5, 128, 19)
    conditioner = Conditioner(128, 3)
    m = Generator()

    a, b, hierarchical_condition = conditioner(x, mel)
    sigma = 0.1
    z = torch.randn_like(x)
    S = m(x + sigma * z, sigma, hierarchical_condition)
