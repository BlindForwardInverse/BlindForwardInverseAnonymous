import torch
import torch.nn as nn
import torch.nn.functional as F

import soundfile as sf
import numpy as np
import math
from tqdm import tqdm
from einops import rearrange, repeat
from functools import partial
from torch.distributions import Categorical

import os; opj=os.path.join
from utils.audio_transform import spec_transform, audio_transform, stft

class ParticleFiltering(nn.Module):
    def __init__(self, sde,
                 num_particles=8,
                 diffusion_steps=64,
                 S_noise=1.000, S_churn=5, S_tmin=0.05, S_tmax=50,
                 save_intermediate=False,
                 print_progress_bar=True,
                 print_grad_norm=False,
                 **kwargs):
        super().__init__()
        self.sde = sde
        self.steps = diffusion_steps
        self.timesteps = self.sde.sigma_scheduler(self.steps)

        self.num_particles = num_particles
        print("-------------------------------------------------------")
        print(f"Particle Filtering with num_particles = {num_particles}")
        print("-------------------------------------------------------")

        self.S_noise = S_noise
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.device = 'cuda'

        self.save_intermediate = save_intermediate
        self.print_progress_bar = print_progress_bar
        if save_intermediate : self.epoch = 0
        self.print_grad_norm = print_grad_norm

    def forward(self, y=None, y_wav=None, c_embedding=None, t_embedding=None):
        # For tqdm setting
        iterator = enumerate(zip(self.timesteps[:-1], self.timesteps[1:]))
        if self.print_progress_bar:
            iterator = tqdm(iterator, total=len(self.timesteps)-1, desc="Diffusion Steps", leave=False)

        # particle init
        particles = []
        for p in range(self.num_particles):
            particles.append(self.timesteps[0] * torch.randn_like(y_wav))

        # weight init
        weight_particle = []
        for p in range(self.num_particles):
            weight_particle.append(torch.ones(1, device=y.device) / self.num_particles)

        # diffusion steps
        for i, (t, t_next) in iterator:
            for p in range(self.num_particles):
                x = particles[p]

                # Increase noise temporarily
                gamma = self.get_gamma(t)
                z = self.S_noise * torch.randn_like(x)
                t_hat = t + gamma * t
                x_hat = x + np.sqrt(t_hat ** 2 - t ** 2) * z

                # Euler step
                d ,denoised = self.estimate_dx_dt(x_hat, t_hat, y, y_wav, t_embedding=t_embedding, c_embedding=c_embedding)
                x_next = x_hat + (t_next - t_hat) * d

                # 2nd order correction
                if i < self.steps - 1:
                    d_prime, denoised_prime = self.estimate_dx_dt(x_next, t_next,
                                                   y=y, y_wav=y_wav, t_embedding=t_embedding, c_embedding=c_embedding)
                    x_next = x_hat + (t_next - t_hat) * 0.5 * (d + d_prime)

                # update particle
                particles[p] = x_next

            # weight update
            if i < self.steps - 1 :
                log_weights = []
                for p in range(self.num_particles):
                    x_temp = particles[p]
                    d, denoised = self.estimate_dx_dt(x_temp, t_next, y, y_wav, t_embedding=t_embedding, c_embedding=c_embedding)
                    Ax = torch.stack([operator(G, _denoised) for G, _denoised in zip(graph, denoised)], dim=0)
                    log_weight = torch.log(weight_particle[p]) + self.log_normal(y_wav, Ax)
                    log_weights.append(log_weight)
                
                # normalize weight
                log_weights = [lw - max(log_weights) for lw in log_weights]
                unnormalized_weight = [torch.exp(lw) for lw in log_weights]
                weight_particle = [w / sum(unnormalized_weight) for w in unnormalized_weight]

                # resample
                indices = np.arange(self.num_particles)
                weight_resample = np.array([w.detach().cpu().numpy() for w in weight_particle])
                sampled_indices = np.random.choice(indices, size=self.num_particles, replace=True, p=weight_resample[:,0])
                
                for p in range(self.num_particles):
                    particles[p] = particles[sampled_indices[p]]
                weight_particle = [torch.ones(1, device=y.device) / self.num_particles for _ in range(self.num_particles)]
            else:
                x_enhanced = particles[0]        
            
        return x_next

    def forward_with_operator(self, _operator, graph=None,  y=None, y_wav=None, c_embedding=None, t_embedding=None):
        batch_size = y_wav.shape[0]
        device = y.device

        # For tqdm setting
        iterator = enumerate(zip(self.timesteps[:-1], self.timesteps[1:]))
        if self.print_progress_bar:
            iterator = tqdm(iterator, total=len(self.timesteps)-1, desc="Diffusion Steps", leave=False)

        # Operator Setting
        operator = _operator

        # particle init
        particles = []
        for p in range(self.num_particles):
            particles.append(self.timesteps[0] * torch.randn_like(y_wav))

        # weight init
        weight_particle = torch.ones(batch_size, self.num_particles, device=y.device) / self.num_particles # (B x P)

        for i, (t, t_next) in iterator:
            for p in range(self.num_particles):
                x = particles[p]

                # Increase noise temporarily
                gamma = self.get_gamma(t)
                z = self.S_noise * torch.randn_like(x)
                t_hat = t + gamma * t
                x_hat = x + np.sqrt(t_hat ** 2 - t ** 2) * z

                # Euler step
                x_hat.requires_grad = True
                d, denoised = self.estimate_dx_dt(x_hat, t_hat, y, y_wav, t_embedding=t_embedding, c_embedding=c_embedding)
                Ax = torch.stack([operator(G, _denoised) for G, _denoised in zip(graph, denoised)], dim=0)
                loss = F.mse_loss(y_wav, Ax, reduction='sum')
                gradient = torch.autograd.grad(outputs=loss, inputs=x_hat)[0]
                if torch.isnan(gradient).any() or i == self.steps-1:
                    d_guide = torch.zeros_like(x_hat)
                else:
                    d_guide = t * gradient

                x_next = x_hat + (t_next - t_hat) * (d + d_guide)
                x_next = x_next.detach()
                d_guide = d_guide.detach()
                loss = loss.detach()

                # 2nd order correction
                if i < self.steps - 1:
                    x_next.requires_grad=True
                    d_prime, denoised_prime = self.estimate_dx_dt(x_next, t_next,
                                                   y=y, y_wav=y_wav, t_embedding=t_embedding, c_embedding=c_embedding)

                    Ax = torch.stack([operator(G, _denoised) for G, _denoised in zip(graph, denoised_prime)], dim=0)
                    loss = F.mse_loss(y_wav, Ax, reduction='sum')
                    gradient = torch.autograd.grad(outputs = loss, inputs=x_next)[0]
                    if torch.isnan(gradient).any():
                        d_guide_prime = torch.zeros_like(x_next)
                    else:
                        d_guide_prime = t_next * gradient
                    x_next = x_hat + (t_next - t_hat) * (0.5 * (d + d_prime) + 0.5 * (d_guide + d_guide_prime))

                denoised = denoised.detach()
                x_next = x_next.detach()
                # update particle
                particles[p] = x_next

            # weight update
            if i < self.steps - 1 :
                log_weights = torch.zeros_like(weight_particle) # (B x P)
                for p in range(self.num_particles):
                    x_temp = particles[p]
                    d, denoised = self.estimate_dx_dt(x_temp, t_next, y, y_wav, t_embedding=t_embedding, c_embedding=c_embedding)
                    Ax = torch.stack([operator(G, _denoised) for G, _denoised in zip(graph, denoised)], dim=0)
                    log_weights[:, p] = torch.log(weight_particle[:, p]) + self.log_normal(y_wav, Ax)

                # normalize weight
                for b in range(batch_size):
                    log_weights[b, :] = log_weights[b, :] - torch.max(log_weights[b, :])
                unnormalized_weight = torch.exp(log_weights)
                weight_particle = unnormalized_weight / rearrange(torch.sum(unnormalized_weight, dim=1), 'b -> b 1')

                # resample
                indices = np.arange(self.num_particles)
                weight_resample = weight_particle.detach().cpu().numpy()
                for b in range(batch_size):
                    sampled_indices = np.random.choice(indices, size=self.num_particles, replace=True,
                                                       p=weight_resample[b])
                    for idx_resample in sampled_indices:
                        particles[p][b] = particles[idx_resample][b]

                weight_particle = torch.ones(batch_size, self.num_particles, device=y.device) / self.num_particles
                
            else:
                log_weights = torch.zeros_like(weight_particle) # (B x P)
                for p in range(self.num_particles):
                    x_temp = particles[p]
                    Ax = torch.stack([operator(G, _denoised) for G, _denoised in zip(graph, x_temp)], dim=0)
                    log_weights[:, p] = torch.log(weight_particle[:, p]) + self.log_normal(y_wav, Ax)

                # normalize weight
                for b in range(batch_size):
                    log_weights[b, :] = log_weights[b, :] - torch.max(log_weights[b, :])
                unnormalized_weight = torch.exp(log_weights)
                weight_particle = unnormalized_weight / rearrange(torch.sum(unnormalized_weight, dim=1), 'b -> b 1')

                # Taking argmax
                argmax = torch.argmax(weight_particle, dim=1) # (B,)
                x_enhanced = torch.stack([particles[idx][b] for i, idx in enumerate(argmax)], dim=0)
        return x_enhanced

    def forward_with_unknown_operator(self, _operator,  y=None, y_wav=None, c_embedding=None, t_embedding=None):
        batch_size = y_wav.shape[0]
        device = y.device

        # For tqdm setting
        iterator = enumerate(zip(self.timesteps[:-1], self.timesteps[1:]))
        if self.print_progress_bar:
            iterator = tqdm(iterator, total=len(self.timesteps)-1, desc="Diffusion Steps", leave=False)

        # Operator Setting
        operator = partial(_operator, y_spec=y, y_wav=y_wav,
                                     global_condition=t_embedding, local_condition=c_embedding)

        # particle init
        particles = []
        for p in range(self.num_particles):
            particles.append(self.timesteps[0] * torch.randn_like(y_wav))

        # weight init
        weight_particle = torch.ones(batch_size, self.num_particles, device=y.device) / self.num_particles # (B x P)

        for i, (t, t_next) in iterator:
            for p in range(self.num_particles):
                x = particles[p]

                # Increase noise temporarily
                gamma = self.get_gamma(t)
                z = self.S_noise * torch.randn_like(x)
                t_hat = t + gamma * t
                x_hat = x + np.sqrt(t_hat ** 2 - t ** 2) * z

                # Euler step
                x_hat.requires_grad = True
                d, denoised = self.estimate_dx_dt(x_hat, t_hat, y, y_wav, t_embedding=t_embedding, c_embedding=c_embedding)
                Ax = operator(denoised)
                loss = F.mse_loss(y_wav, Ax, reduction='sum')
                gradient = torch.autograd.grad(outputs=loss, inputs=x_hat)[0]
                if torch.isnan(gradient).any() or i == self.steps - 1:
                    d_guide = torch.zeros_like(x_hat)
                else:
                    d_guide = t * gradient

                x_next = x_hat + (t_next - t_hat) * (d + d_guide)
                x_hat = x_hat.detach()
                x_next = x_next.detach()
                d_guide = d_guide.detach()
                loss = loss.detach()

                # 2nd order correction
                if i < self.steps - 1:
                    x_next.requires_grad=True
                    d_prime, denoised_prime = self.estimate_dx_dt(x_next, t_next,
                                                   y=y, y_wav=y_wav, t_embedding=t_embedding, c_embedding=c_embedding)

                    Ax = operator(denoised_prime)
                    loss = F.mse_loss(y_wav, Ax, reduction='sum')
                    gradient = torch.autograd.grad(outputs = loss, inputs= x_next)[0]
                    if torch.isnan(gradient).any():
                        d_guide_prime = torch.zeros_like(x_next)
                    else:
                        d_guide_prime = t_next * gradient
                    loss = loss.detach()
                    d_guide_prime = d_guide_prime.detach()

                    x_next = x_hat + (t_next - t_hat) * (0.5 * (d + d_prime) + 0.5 * (d_guide + d_guide_prime))

                denoised = denoised.detach()
                x_next = x_next.detach()
                # update particle
                particles[p] = x_next

            # weight update
            if i < self.steps - 1 :
                log_weights = torch.zeros_like(weight_particle) # (B x P)
                for p in range(self.num_particles):
                    x_temp = particles[p]
                    d, denoised = self.estimate_dx_dt(x_temp, t_next, y, y_wav, t_embedding=t_embedding, c_embedding=c_embedding)
                    Ax = operator(denoised)
                    log_weights[:, p] = torch.log(weight_particle[:, p]) + self.log_normal(y_wav, Ax)

                # normalize weight
                for b in range(batch_size):
                    log_weights[b, :] = log_weights[b, :] - torch.max(log_weights[b, :])
                unnormalized_weight = torch.exp(log_weights)
                weight_particle = unnormalized_weight / rearrange(torch.sum(unnormalized_weight, dim=1), 'b -> b 1')

                # resample
                indices = np.arange(self.num_particles)
                weight_resample = weight_particle.detach().cpu().numpy()
                for b in range(batch_size):
                    sampled_indices = np.random.choice(indices, size=self.num_particles, replace=True,
                                                       p=weight_resample[b])
                    for idx_resample in sampled_indices:
                        particles[p][b] = particles[idx_resample][b]

                weight_particle = torch.ones(batch_size, self.num_particles, device=y.device) / self.num_particles
                
            else:
                x_enhanced = particles[0]
#                 log_weights = torch.zeros_like(weight_particle) # (B x P)
#                 for p in range(self.num_particles):
#                     x_temp = particles[p]
#                     Ax = operator(x_temp)
#                     log_weights[:, p] = torch.log(weight_particle[:, p]) + self.log_normal(y_wav, Ax)

#                 # normalize weight
#                 for b in range(batch_size):
#                     log_weights[b, :] = log_weights[b, :] - torch.max(log_weights[b, :])
#                 unnormalized_weight = torch.exp(log_weights)
#                 weight_particle = unnormalized_weight / rearrange(torch.sum(unnormalized_weight, dim=1), 'b -> b 1')

#                 # Taking argmax
#                 argmax = torch.argmax(weight_particle, dim=1) # (B,)
#                 x_enhanced = torch.stack([particles[idx][b] for i, idx in enumerate(argmax)], dim=0)
        return x_enhanced

    def estimate_dx_dt(self, x_t, t, y=None, y_wav=None, t_embedding=None, c_embedding=None):
        # Unconditional Score
        t_ones = t * torch.ones(x_t.shape[0], device=self.device, requires_grad=False)
        denoised = self.sde.denoiser(x_t, t_ones,
                                     y, y_wav, t_embedding=t_embedding, c_embedding=c_embedding)
        
        d_uncond = (x_t - denoised) / t
        d_uncond = d_uncond.detach()
        return d_uncond, denoised

    def get_gamma(self, t):
        if t >= self.S_tmin and t <= self.S_tmax:
            gamma = min(self.S_churn / self.steps, np.sqrt(2) - 1)
        else:
            gamma = 0
        return gamma

    @staticmethod
    def normalize_weight(log_weight):
        weight = torch.exp(log_weight)
        normalized_weight = weight / torch.sum(weight)
        return normalized_weight

    @staticmethod
    def log_normal(y_t, Ax):
#         mse_loss = F.mse_loss(y_t, Ax, reduction='sum')
        mse_loss = torch.sum(torch.square(y_t - Ax), dim=1)
        # Compute the log-likelihood
        log_likelihood = -0.5 * mse_loss
        return log_likelihood
#     def forward(self, _operator, y, y_wav, global_condition, local_condition):
#         device = y_wav.device
#         signal_len = y_wav.shape[-1]

#         # For tqdm setting
#         iterator = enumerate(zip(self.timesteps[:-1], self.timesteps[1:]))
#         if self.print_progress_bar:
#             iterator = tqdm(iterator, total=len(self.timesteps)-1, desc="Diffusion Steps", leave=False)

#         # Initialize
#         x = self.timesteps[0] * torch.randn(self.num_particles, signal_len, device=device, requires_grad=True)
#         y = y.squeeze(0)
#         y_wav = y_wav.squeeze(0)
#         global_condition = global_condition.squeeze(0)
#         local_condition = local_condition.squeeze(0)

#         y = repeat(y, ' f t c-> b f t c', b=self.num_particles)
#         y_wav = repeat(y_wav, ' t -> b t', b=self.num_particles)
#         global_condition = repeat(global_condition, ' t -> b t', b=self.num_particles)
#         local_condition = repeat(local_condition, ' c t -> b c t', b=self.num_particles)
#         operator = partial(_operator, y_spec=y, y_wav=y_wav,
#                                      global_condition=global_condition, local_condition=local_condition)
        
#         for i, (t, t_next) in iterator:
#             # Proposal
#             x, x_hat = self.one_step_update(x, i, t, t_next, operator, y, y_wav, global_condition, local_condition)

#             # Observation
# #             y_t = y_wav + t * torch.randn_like(y_wav) # (T,)
#             y_t = y_wav

#             # Weight init
#             weight = torch.ones(self.num_particles, device=device) / self.num_particles
#             log_weight = torch.log(weight)

#             # Weight Update
#             Ax = operator(x_hat)
#             for p in range(self.num_particles):
#                 log_weight[p] += self.log_normal(y_t, Ax, t)

#             weight = self.normalize_weight(log_weight)
#             log_weight = torch.log(weight)
# #             print(weight)

#             # Resample
# #             indices = np.arange(self.num_particles)
# #             sampled_indices = np.random.choice(indices, size=self.num_particles, replace=True, p=weight.detach().cpu().numpy())
# #             print(sampled_indices)
# #             x = torch.stack([x[idx] for idx in resampled], dim=0)

#         return x
