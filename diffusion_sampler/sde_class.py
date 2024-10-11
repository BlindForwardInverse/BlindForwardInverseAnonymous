import numpy as np
import torch
import torch.nn as nn
from functools import partial

import math
from einops import rearrange
from utils.torch_utils import reshape_x
from utils.audio_transform import audio_transform, spec_transform

class ProbODEFlow:
    def __init__(self, f_theta, rho, sigma_min, sigma_max, sigma_data, projector=None,**kwargs):
        self.f_theta = f_theta
        self.projector = projector
        self.rho = rho
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

        self.spec_to_wav = partial(spec_transform, transform_type='unpower_ri', 
                                   n_fft=2046, win_length=2046, hop_length=512,
                                   length=97792)

        self.wav_to_spec = partial(audio_transform, transform_type='power_ri', 
                                   n_fft=2046, win_length=2046, hop_length=512,
                                   )
    def get_coeff(self, sigma):
        sigma_data = self.sigma_data
        c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
        c_out = sigma * sigma_data / torch.sqrt(sigma_data ** 2 + sigma ** 2)
        c_in = 1 / torch.sqrt(sigma ** 2 + sigma_data**2)
        c_noise = 0.25 * torch.log(sigma)
        return c_skip, c_out, c_in, c_noise

    def denoiser(self, x, sigma, y=None, t_embedding=None, c_embedding=None):
        # sigma must be expanded to match with the batch size
        # f_theta is network output and converted into a denoiser function
        shape_len = len(x.shape)
        if len(sigma.shape) == 1:
            sigma = reshape_x(sigma, shape_len)
        c_skip, c_out, c_in, c_noise = self.get_coeff(sigma)

        if y is not None:
            model_input = torch.cat([c_in * x, y], dim=-1)
        else:
            model_input = c_int * x
        D_theta = c_skip * x + c_out * self.f_theta(model_input, c_noise.squeeze(), global_condition=t_embedding, local_condition=c_embedding)  ##
        if self.projector :
            y_wav = self.spec_to_wav(y)
            d_wav = self.spec_to_wav(D_theta)
            spec_input = torch.cat([D_theta, y], dim=-1)
            wav_input = torch.stack([d_wav, y_wav], dim=1)
            projected = self.projector(spec_input, wav_input, t_embedding, c_embedding)
        return D_theta

    def sigma_scheduler(self, N):
        rho = self.rho
        A = self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho)
        B = self.sigma_max ** (1 / rho)
        sigmas = [(A * i / (N - 1) + B) ** rho if i < N else 0 for i in range(N + 1)]
        return sigmas

def sigma_scheduler(N, rho, sigma_min, sigma_max):
    A = sigma_min ** (1 / rho) - sigma_max ** (1 / rho)
    B = sigma_max ** (1 / rho)
    sigmas = [(A * i / (N - 1) + B) ** rho if i < N else 0 for i in range(N + 1)]
    return sigmas

class OUVESDE():
    def __init__(self, gamma=1.5, sigma_min=0.05, sigma_max=0.5, transform_type='power_ri', **kwargs):
        self.gamma = gamma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.ratio = sigma_max / sigma_min
        self.transform_type = transform_type

    def forward_process(self, t, x_t, y):
        f = self.gamma * (y - x_t)
        c = self.sigma_min * math.sqrt(2 * math.log(self.ratio))
        k_t = torch.pow(self.ratio, t)
        g = c * k_t
        if self.transform_type == 'power_ri':
            g = rearrange(g, 'b -> b 1 1 1')
        elif self.transform_type == 'waveform':
            g = rearrange(g, 'b -> b 1')
        return f, g

    def reverse_process(self, t, x_t, y, score_model):
        f, g = self.forward_process(t, x_t, y)
        drift = f - torch.square(g) * score_model(x_t, y, t)
        return drift, g

    def forward_sample(self, t, x_0, y):
        z = torch.randn_like(x_0)
        mean, sigma = self.perturb_kernel(t, x_0, y)

        sample = mean + sigma * z
        return sample, sigma, z

    def prior_sample(self, y):
        t = torch.ones(y.shape[0], device = y.device)
        std = self.perturb_kernel_std(t)
        z = torch.randn_like(y)
        return y + std * z

    def perturb_kernel(self, t, x_0, y):
        mean = self.perturb_kernel_mean(t, x_0, y)
        std = self.perturb_kernel_std(t)
        return mean, std

    def perturb_kernel_mean(self, t, x_0, y):
        if self.transform_type == 'power_ri':
            exp_interp = rearrange(torch.exp(-self.gamma * t), 'b -> b 1 1 1')
        
        elif self.transform_type == 'waveform':
            exp_interp = rearrange(torch.exp(-self.gamma * t), 'b -> b 1')

        mean = exp_interp * x_0 + (1 - exp_interp) * y
        return mean

    def perturb_kernel_std(self, t):
        sigma_min, gamma, logsig = self.sigma_min, self.gamma, math.log(self.ratio)
        std = torch.sqrt(
            (sigma_min**2
                * torch.exp(-2 * gamma * t)
                * (torch.exp(2 * (gamma + logsig) * t) - 1)
                * logsig
            )/
            (gamma + logsig)
        )
        if len(std.shape) == 1 and self.transform_type == 'waveform':
            sigma = rearrange(std, 'b -> b 1')
        elif len(std.shape) == 1 and self.transform_type == 'power_ri':
            sigma = rearrange(std, 'b -> b 1 1 1')
        return sigma

    def denoiser(self, x_t, y, t, score):
        variance = self.perturb_kernel_std(t) ** 2
        exp_interp = rearrange(torch.exp(-self.gamma * t), 'b -> b 1 1 1')
        x_hat = (x_t + variance * score - (1 - exp_interp) * y) / exp_interp
        return x_hat


