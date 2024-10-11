import torch
import torch.nn as nn
import torch.nn.functional as F

from models.universe.film import FiLM


class GeneratorDown(nn.Module):
    """
    Downsampling ConvBlock for Generator.
    - condition input: affine-transformed RFF (g)
    - condition output: U-Net skip connection (v)
    """

    def __init__(self, c_in, c_out, kernel_size, embedding_size=256):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size

        self.conv1 = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(c_in, c_in, kernel_size=5, padding=2),
        )
        self.film = FiLM(hidden_size=c_in, embedding_size=embedding_size)
        self.conv2 = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(c_in, c_in, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(c_in, c_in, kernel_size=3, padding=1),
        )
        self.stconv = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(c_in, c_out, kernel_size=kernel_size, stride=kernel_size),
        )

    def forward(self, x, sigma_emb):
        h = self.conv1(x)
        h = self.film(h, sigma_emb)
        h = self.conv2(h)
        h = h + x

        skip = h
        h_out = self.stconv(h)  ##
        return h_out, skip


class GeneratorUp(nn.Module):
    """
    Upsampling ConvBlock for Generator.
    - condition input: affine-transformed RFF (g), hierarchical condition (c), U-Net skip connection (v)
    """

    def __init__(self, c_in, c_out, kernel_size, embedding_size=256):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out

        self.trconv = nn.Sequential(
            nn.PReLU(),
            nn.ConvTranspose1d(
                c_in, c_out, kernel_size=kernel_size, stride=kernel_size
            ),
        )
        self.conv1 = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(c_out, c_out, kernel_size=5, padding=2),
        )
        self.film = FiLM(hidden_size=c_out, embedding_size=embedding_size)
        self.conv2 = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(c_out, c_out, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(c_out, c_out, kernel_size=3, padding=1),
        )

    def forward(self, x, sigma_emb, skip):
        x = self.trconv(x)
        x = x + skip

        h = self.conv1(x)
        h = self.film(h, sigma_emb)
        h = self.conv2(h)
        h = h + x
        return h


class PlainConvBlock(nn.Module):
    """
    plain version of ConvBlock.
    no StConv/TrConv. no conditional input/output.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.conv1 = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
        )
        self.conv2 = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return x
