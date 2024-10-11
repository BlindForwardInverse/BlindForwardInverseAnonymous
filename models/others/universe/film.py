import torch
import torch.nn as nn
import numpy as np

from einops import rearrange

class FourierFeature(nn.Module):
    def __init__(self, embedding_size=256):
        super().__init__()
        freq = torch.randn(64) * 4
        self.register_buffer("freq", freq)
        self.linear = nn.Sequential(
            nn.Linear(128, embedding_size),
            nn.PReLU(),
            nn.Linear(embedding_size, embedding_size),
            nn.PReLU(),
        )
        self.device = "cuda"

    def forward(self, sigma_t):
        # log_sigma = (B), self.freq = (64)
        #         log_sigma = torch.log10(sigma_t)
        log_sigma = sigma_t.to(self.device)
        # log_sigma = torch.tensor(log_sigma)
        log_sigma = rearrange(log_sigma, "b -> b 1")
        cos = torch.cos(2.0 * np.pi * self.freq * log_sigma)
        sin = torch.sin(2.0 * np.pi * self.freq * log_sigma)
        fourier_feature = torch.cat((cos, sin), dim=-1)  # (B, 128)
        sigma_embedding = self.linear(fourier_feature)  # (B, 256)
        return sigma_embedding


class FiLM(nn.Module):
    def __init__(
        self,
        hidden_size,
        embedding_size=256,
    ):
        super().__init__()
        self.linear = nn.Sequential(
            nn.SiLU(), nn.Linear(embedding_size, hidden_size * 2)
        )
        self.device = "cuda"

    def forward(self, x, sigma_emb):
        # sigma_emb is c in the paper
        sigma_emb = self.linear(sigma_emb)
        sigma_emb = rearrange(sigma_emb, "b c -> b c 1")
        scale, shift = torch.chunk(
            sigma_emb, chunks=2, dim=1
        )  # (B, 128, 1), (B, 128, 1)
        x = scale * x + shift
        return x


if __name__ == "__main__":
    sigma_t = torch.rand(8)
    ff = FourierFeature(embedding_size=256)
    sigma_embedding = ff(sigma_t)
    print(sigma_embedding.shape)

    x = torch.rand(8, 128, 44100)  # (B, C, T) for waveform unet
    film = FiLM(hidden_size=128, embedding_size=256)
    h = film(x, sigma_embedding)
    print(h.shape)
