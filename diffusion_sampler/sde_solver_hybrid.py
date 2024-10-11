import torch
import torch.nn as nn
import torch.nn.functional as F

import soundfile as sf
import numpy as np
from functools import partial
from tqdm import tqdm
from einops import rearrange
import os; opj=os.path.join
from utils.audio_transform import spec_transform, audio_transform, stft

class Heun2ndSamplerHybrid(nn.Module):
    def __init__(self, sde,
                 diffusion_steps=256,
                 S_noise=1.003, S_churn=5, S_tmin=0.3, S_tmax=2.0,
                 save_intermediate=False,
                 print_progress_bar=True,
                 print_grad_norm=False,
                 **kwargs):
        super().__init__()
        self.sde = sde
        self.steps = diffusion_steps
        self.timesteps = self.sde.sigma_scheduler(self.steps)

        self.S_noise = S_noise
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.device = 'cuda'
        
        self.num_guidance = 1
        
        self.save_intermediate = save_intermediate
        self.print_progress_bar = print_progress_bar
        if save_intermediate : self.epoch = 0
        self.print_grad_norm = print_grad_norm
        
    def forward(self, y=None, y_wav=None, c_embedding=None, t_embedding=None):
        x_next = self.timesteps[0] * torch.randn_like(y_wav)

        # For tqdm setting
        iterator = enumerate(zip(self.timesteps[:-1], self.timesteps[1:]))
        if self.print_progress_bar:
            iterator = tqdm(iterator, total=len(self.timesteps)-1, desc="Diffusion Steps", leave=False)

        for i, (t, t_next) in iterator:
            x = x_next

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
        return x_next

    def forward_with_operator(self, _operator, graph=None,  y=None, y_wav=None, c_embedding=None, t_embedding=None):
        x_next = self.timesteps[0] * torch.randn_like(y_wav)

        # For tqdm setting
        iterator = enumerate(zip(self.timesteps[:-1], self.timesteps[1:]))
        if self.print_progress_bar:
            iterator = tqdm(iterator, total=len(self.timesteps)-1, desc="Diffusion Steps", leave=False)

        # Operator Setting
        operator = _operator

        for i, (t, t_next) in iterator:
            x = x_next

            # Increase noise temporarily
            gamma = self.get_gamma(t)
            z = self.S_noise * torch.randn_like(x)
            t_hat = t + gamma * t
            x_hat = x + np.sqrt(t_hat ** 2 - t ** 2) * z

            # Euler step
#             print('======first step=========')
            x_hat.requires_grad = True
            d, denoised = self.estimate_dx_dt(x_hat, t_hat, y, y_wav, t_embedding=t_embedding, c_embedding=c_embedding)
            Ax = torch.stack([operator(G, _denoised) for G, _denoised in zip(graph, denoised)], dim=0)
            loss = F.mse_loss(y_wav, Ax, reduction='sum')
            gradient = torch.autograd.grad(outputs=loss, inputs=x_hat)[0]
            if torch.isnan(gradient).any():
                d_guide = torch.zeros_like(x_hat)
            else:
                d_guide = t * gradient

            x_next = x_hat + (t_next - t_hat) * (d + d_guide)
#             print(loss)
#             print(torch.linalg.norm(d))
#             print(torch.linalg.norm(d_guide))

            x_next = x_next.detach()
            d_guide = d_guide.detach()
            loss = loss.detach()

#             print('======second step=========')
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
#                 print(loss)
#                 print(torch.linalg.norm(d_prime))
#                 print(torch.linalg.norm(d_guide_prime))
                x_next = x_hat + (t_next - t_hat) * (0.5 * (d + d_prime) + 0.5 * (d_guide + d_guide_prime))

            denoised = denoised.detach()
            x_next = x_next.detach()

        return x_next

    def forward_with_unknown_operator(self, _operator,  y=None, y_wav=None, c_embedding=None, t_embedding=None):
        x_next = self.timesteps[0] * torch.randn_like(y_wav)

        # For tqdm setting
        iterator = enumerate(zip(self.timesteps[:-1], self.timesteps[1:]))
        if self.print_progress_bar:
            iterator = tqdm(iterator, total=len(self.timesteps)-1, desc="Diffusion Steps", leave=False)

        # Operator Setting
        operator = partial(_operator, y_spec=y, y_wav=y_wav,
                                     global_condition=t_embedding, local_condition=c_embedding)

        for i, (t, t_next) in iterator:
            x = x_next

            # Increase noise temporarily
            gamma = self.get_gamma(t)
            z = self.S_noise * torch.randn_like(x)
            t_hat = t + gamma * t
            x_hat = x + np.sqrt(t_hat ** 2 - t ** 2) * z

            # Euler step
#             print('======first step=========')
            x_hat.requires_grad = True
            d, denoised = self.estimate_dx_dt(x_hat, t_hat, y, y_wav, t_embedding=t_embedding, c_embedding=c_embedding)
#             print(denoised.shape)
            Ax = operator(denoised)
            loss = F.mse_loss(y_wav, Ax, reduction='sum')
            gradient = torch.autograd.grad(outputs=loss, inputs=x_hat)[0]
            if torch.isnan(gradient).any():
                d_guide = torch.zeros_like(x_hat)
            else:
                d_guide = t * gradient

            x_next = x_hat + (t_next - t_hat) * (d + d_guide)
#             print(loss)
#             print(torch.linalg.norm(d))
#             print(torch.linalg.norm(d_guide))

            x_hat = x_hat.detach()
            x_next = x_next.detach()
            d_guide = d_guide.detach()
            loss = loss.detach()

#             print('======second step=========')
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
#                 print(loss)
#                 print(torch.linalg.norm(d_prime))
#                 print(torch.linalg.norm(d_guide_prime))
                loss = loss.detach()
                d_guide_prime = d_guide_prime.detach()

                x_next = x_hat + (t_next - t_hat) * (0.5 * (d + d_prime) + 0.5 * (d_guide + d_guide_prime))

            denoised = denoised.detach()
            x_next = x_next.detach()

        return x_next
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

    def save_intermediate_results(self, x):
        # Plot and Save Intermediate Results
        if i % 10 == 0 or self.steps - i < 5:
            save_dir = '/ssd3/doyo/plot'
            os.makedirs(save_dir, exist_ok=True)
            sf.write(opj(save_dir, f'epoch_{self.epoch}_timestep_{i}.wav'), 
                     x_next[0, :].detach().cpu().numpy(), 44100)

class Heun2ndSamplerHybridV2(nn.Module):
    def __init__(self, sde,
                 diffusion_steps=256,
                 S_noise=1.003, S_churn=5, S_tmin=0.3, S_tmax=2.0,
                 save_intermediate=False,
                 print_progress_bar=True,
                 print_grad_norm=False,
                 **kwargs):
        super().__init__()
        self.sde = sde
        self.steps = diffusion_steps
        self.timesteps = self.sde.sigma_scheduler(self.steps)

        self.S_noise = S_noise
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.device = 'cuda'
        
        self.num_guidance = 1
        
        self.save_intermediate = save_intermediate
        self.print_progress_bar = print_progress_bar
        if save_intermediate : self.epoch = 0
        self.print_grad_norm = print_grad_norm

    def forward(self, y=None, y_wav=None, c_embedding=None, t_embedding=None):
        x_next = self.timesteps[0] * torch.randn_like(y)

        # For tqdm setting
        iterator = enumerate(zip(self.timesteps[:-1], self.timesteps[1:]))
        if self.print_progress_bar:
            iterator = tqdm(iterator, total=len(self.timesteps)-1, desc="Diffusion Steps", leave=False)

        for i, (t, t_next) in iterator:
            x = x_next

            # Increase noise temporarily
            gamma = self.get_gamma(t)
            z = self.S_noise * torch.randn_like(x)
            t_hat = t + gamma * t
            x_hat = x + np.sqrt(t_hat ** 2 - t ** 2) * z

            # Euler step
            d = self.estimate_dx_dt(x_hat, t_hat, y, y_wav, t_embedding=t_embedding, c_embedding=c_embedding)
            x_next = x_hat + (t_next - t_hat) * d

            # 2nd order correction
            if i < self.steps - 1:
                d_prime  = self.estimate_dx_dt(x_next, t_next,
                                               y=y, y_wav=y_wav, t_embedding=t_embedding, c_embedding=c_embedding)
                x_next = x_hat + (t_next - t_hat) * 0.5 * (d + d_prime)
        return x_next

    def estimate_dx_dt(self, x_t, t, y=None, y_wav=None, t_embedding=None, c_embedding=None):
        # Unconditional Score
        t_ones = t * torch.ones(x_t.shape[0], device=self.device, requires_grad=False)
        x_t = x_t.detach()
        denoised = self.sde.denoiser(x_t, t_ones,
                                     y, y_wav, t_embedding=t_embedding, c_embedding=c_embedding)
        
        d_uncond = (x_t - denoised) / t
        d_uncond = d_uncond.detach()
        return d_uncond

    @staticmethod
    def loss_fn(y, pred, loss_mode='magnitude'):
        if loss_mode == 'l2':
            difference = y - pred
            loss = torch.sum(torch.square(difference))
        elif loss_mode == 'mag_l2':
            y_wav = spec_transform(y)
            pred_wav = spec_transform(pred)
            y_mag = stft(y_wav, return_type = 'mag_only')
            pred_mag = stft(pred_wav, return_type = 'mag_only')
            
            difference = y_mag - pred_mag
            print('diffnorm', torch.linalg.norm(difference))
#             loss = torch.sqrt(torch.sum(torch.square(difference)))
            loss = torch.sum(torch.square(difference)) / torch.sum(torch.square(y_mag)) 
        return loss
            

    @staticmethod
    def calc_weight(loss, loss_mode):
        if loss_mode == 'l2': weight = 1
        elif loss_mode == 'mag_l2': weight = 1000
        return weight

    def get_gamma(self, t):
        if t >= self.S_tmin and t <= self.S_tmax:
            gamma = min(self.S_churn / self.steps, np.sqrt(2) - 1)
        else:
            gamma = 0
        return gamma

    def save_intermediate_results(self, x):
        # Plot and Save Intermediate Results
        if i % 10 == 0 or self.steps - i < 5:
            save_dir = '/ssd3/doyo/plot'
            os.makedirs(save_dir, exist_ok=True)
            sf.write(opj(save_dir, f'epoch_{self.epoch}_timestep_{i}.wav'), 
                     x_next[0, :].detach().cpu().numpy(), 44100)

class PredictorCorrector(nn.Module):
    def __init__(self, sde, score_model, predictor_steps, corrector_steps, target_snr, t_min=3e-2):
        super().__init__()
        self.sde = sde
        self.score_model = score_model
        self.predictor_steps = predictor_steps
        self.corrector_steps = corrector_steps
        self.t_min = t_min

        self.dt = 1. / predictor_steps
        self.predictor = EulerMaruyama(sde, score_model, self.dt)
        self.corrector = AnnealedLangevinDynamics(sde, score_model, target_snr)

    def forward(self, y, global_condition, local_condition):
        device = y.device
        b = y.shape[0]
        with torch.no_grad():
            x_t = self.sde.prior_sample(y)
            timesteps = torch.linspace(1., self.t_min, self.predictor_steps, device=device)
            for i in range(self.predictor_steps):
                t = timesteps[i]
                t = torch.ones(b, device=device) * t

                #Corrector Step
                for _ in range(self.corrector_steps):
                    std = self.sde.perturb_kernel_std(t)
                    x_t = self.corrector(x_t, y, t, std)
                
                #Predictor Step
                if i == self.predictor_steps - 1:
                    # Final Denoising following Tweedie's Formula at t = 0
                    x_t = self.predictor(x_t, y, t, global_condition, local_condition, add_noise=False)
                else:
                    x_t = self.predictor(x_t, y, t, global_condition, local_condition,  add_noise=True)
        return x_t

class EulerMaruyama(nn.Module):
    def __init__(self, sde, score_model, dt):
        super().__init__()
        self.sde = sde
        self.dt = dt
        self.score_model = score_model

    def forward(self, x_t, y, t, global_condition=None, local_condition=None,  add_noise=True):
        drift, diffusion = self.sde.forward_process(t, x_t, y)
        f = drift * (self.dt)
        g = diffusion * np.sqrt(self.dt)

        model_input = torch.cat([x_t, y], dim=-1)
        sigma = self.sde.perturb_kernel_std(t)
        if len(sigma.shape) == 4 : sigma = rearrange(sigma, 'b 1 1 1 -> b')
        rev_f = f - torch.square(g) * self.score_model(model_input, sigma, global_condition, local_condition)
        rev_g = g
        z = torch.randn_like(x_t)
        if add_noise:
            return x_t - rev_f + rev_g * z
        else:
            return x_t - rev_f

class AnnealedLangevinDynamics(nn.Module):
    def __init__(self, sde, score_model, target_snr=0.1):
        super().__init__()
        self.sde = sde
        self.score_model = score_model
        self.target_snr = target_snr
        
    def forward(self, x, y, t, std):
        score = self.score_model(x, y, t)
        z = torch.randn_like(x)
        epsilon = torch.square(self.target_snr * std) * 2
        # Here std must be replaced to score_norm / noise_norm
        x = x + epsilon * score + torch.sqrt(2 * epsilon) * z
        return x

