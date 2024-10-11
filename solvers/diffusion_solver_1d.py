import os
import pickle
from glob import glob
import random

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
from einops import rearrange
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from diffusion_sampler.sde_class import ProbODEFlow, sigma_scheduler
from diffusion_sampler.sde_solver import Heun2ndSampler
from models.diffusion_models.imagen.imagen_pytorch import Unet
from models.operator_models.mftaa_encoder import MFTAAEncoder, ToAfxToken, ToAfxEmbedding

from omegaconf import OmegaConf

from loss import EDMLoss
from metric import SpeechEnhancementMetric

from solver_operator import UnetSolver

from plot_module.plot_spectrogram import plot_audio

from utils.tensor_utils import reshape_x, seed_everything
from utils.audio_normalization import absolute_normalize, rms_normalize
from utils.audio_transform import audio_transform, spec_transform

opj = os.path.join

class DiffusionSpecSolver(pl.LightningModule):
    def __init__(
        self, save_dir=None, lr=1e-4, 
        # Signal Spec
        target_sr=44100, n_fft=2046, win_length=2046, hop_length=512,
        # Model
        unet_channels=128, num_resnet_blocks=[2,4,6,8], dim_mults=[1,2,4,8], num_layer_attns=2, num_layer_cross_attns=2,
        cond_dim=256, embedding_size=512, attn_dim_head=32, attn_heads=16,
        # Reference Encoder
        pretrained_encoder = None, use_fx_encoder=False,
        # Diffusion Sampler
        diffusion_steps=64, sigma_min=3e-4, sigma_max=3, train_rho=10, rho=13, sigma_data=0.0126, 
        save_every_n_epochs=1,
        cat_c_embed = True, cat_t_embed = False, ema_decay=0.999,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.save_dir = save_dir
        self.lr = lr

        self.target_sr = target_sr
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.diffusion_steps = diffusion_steps

        self.save_every_n_epochs = save_every_n_epochs

        # Training
        num_layers = len(dim_mults)
        layer_attns = [False] * (num_layers - num_layer_attns) + [True] * num_layer_attns
        layer_cross_attns = [False] * (num_layers - num_layer_cross_attns) + [True] * num_layer_cross_attns
        self.model = Unet(dim=unet_channels,
                          num_resnet_blocks=num_resnet_blocks,
                          dim_mults=dim_mults,
                          cond_dim=cond_dim,
                          embedding_size=embedding_size,
                          layer_attns=layer_attns,
                          layer_cross_attns=layer_cross_attns,
                          attn_dim_head=attn_dim_head,
                          attn_heads=attn_heads,
                          cat_c_embed=cat_c_embed,
                          cat_t_embed=cat_t_embed)

        self.loss_fn = EDMLoss()
        self.metric_fn = SpeechEnhancementMetric(sr=target_sr)

        # SDE
        self.sde = ProbODEFlow(f_theta = self.model,
                               sigma_min = sigma_min,
                               sigma_max = sigma_max,
                               rho = rho,
                               sigma_data = sigma_data,
                               **kwargs)
        self.sampler = Heun2ndSampler(self.sde,
                                      diffusion_steps=diffusion_steps,
                                      **kwargs)
        self.sigma_list = sigma_scheduler(N=128, 
                                          rho=train_rho,
                                          sigma_min=sigma_min,
                                          sigma_max=sigma_max)[:-1]

        # Load Pretrained Operator
        if self.pretrained_encoder is not None:
            self.fx_encoder = pretrained_encoder.fx_encoder
            self.to_global_condition = pretrained_encoder.to_global_condition
            self.to_local_condition = pretrained_encoder.to_local_condition

        # EMA
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(
            self.model.parameters(), decay=self.ema_decay
        )

    def forward(self, y_spec, degraded, degraded_wav=None):
        # y : clean spectrogram data, x : perturbed data (x = spec + n)
        # This follows the convention in 'Elucidating the Design Space~~' paper.
        device = y_spec.device
        batch_size = y_spec.shape[0]

#         '''
#         --------------------------------
#         sample sigma from predefined list
#         --------------------------------
#         '''
#         sigma = random.choices(self.sigma_list, k=batch_size)
#         sigma = torch.tensor(sigma, device=device)
        
        '''
        -----------------------------------
        sample sigma as the proposed paper
        -----------------------------------
        '''
        log_sigma = torch.randn(batch_size, device=device) * 1.5 - 1.5
        sigma = torch.clamp(torch.exp(log_sigma), min=self.sigma_min, max=self.sigma_max)
        sigma = reshape_x(sigma, 4)  # [32, 1, 1, 1]

        n = sigma * torch.randn_like(y_spec)
        c_skip, c_out, c_in, c_noise = self.sde.get_coeff(sigma)

        # Estimate Loss
        preconditioned = c_in * (y_spec + n)
        model_input = torch.cat([preconditioned, degraded], dim=-1)
        c_embedding, t_embedding = self.get_embedding_tokens(degraded_wav)
        model_output = self.model(model_input,
                                  c_noise.squeeze(),
                                  t_embedding = t_embedding,
                                  c_embedding = c_embedding)
        target = (1 / c_out) * (y_spec - c_skip * (y_spec + n))
        loss = self.loss_fn(model_output, target)
        return loss

    def get_embedding_tokens(self, degraded_wav):
        if self.use_fx_encoder:
            if self.pretrained_encoder:
                fx_latent = self.pretrained_operator.fx_encoder(degraded_wav)
                c = self.pretrained_operator.to_afx_tokens(fx_latent)
                t = self.pretrained_operator.to_fx_embedding(fx_latent)
            else:
                fx_latent = self.fx_encoder(degraded_wav)
                c = self.to_afx_tokens(fx_latent)
                t = self.to_fx_embedding(fx_latent)
        else:
            c, t = None, None
        return c, t

#     def get_embedding_tokens(self, degraded_wav):
#         if self.use_fx_encoder:
#             if self.pretrained_encoder:
#                 fx_latent = self.pretrained_operator.fx_encoder(degraded_wav)
#                 c = self.pretrained_operator.to_afx_tokens(fx_latent)
#                 t = self.pretrained_operator.to_fx_embedding(fx_latent)
#             else:
#                 fx_latent = self.fx_encoder(degraded_wav)
#                 c = self.to_afx_tokens(fx_latent)
#                 t = self.to_fx_embedding(fx_latent)
#         else:
#             c, t = None, None
#         return c, t

    def training_step(self, batch, _):
        x_0, degraded, degraded_wav = batch['dry'], batch['wet'], batch['wet_wav']
        loss = self(x_0, degraded, degraded_wav)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_start(self):
        self.metric_history = dict()

    def validation_step(self, batch, idx):
        with self.ema.average_parameters():
            if self.current_epoch % self.save_every_n_epochs == 0 :
                noisy_spec = batch['wet_ref_spec']
                clean, noisy = batch['dry_ref'], batch['wet_ref']
                valid_types, afxs, audio_idxs = batch['valid_type'], batch['afx'], batch['audio_idx']

                shape = noisy_spec.shape
                c_embedding, t_embedding = self.get_embedding_tokens(noisy)
                generated = self.sampler(shape, y=noisy_spec, t_embedding=t_embedding, c_embedding=c_embedding)
                generated_audio = spec_transform(generated,
                                                 n_fft=self.n_fft,
                                                 win_length=self.win_length,
                                                 hop_length=self.hop_length,
                                                 ).detach()

                for i, (valid_type, afx, audio_idx) in enumerate(zip(valid_types, afxs, audio_idxs)):
                    pred = generated_audio[i, ...]
                    clean_audio = clean[i, ...]
                    noisy_audio = noisy[i, ...]

                    metric = self.metric_fn(pred, clean_audio)
                    for k, v in metric.items():
                        v = v.detach().cpu().numpy()
                        if k in self.metric_history: self.metric_history[k].append(v)
                        else: self.metric_history[k] = [v]

                    # Plot and Save
                    tag = f"{valid_type}_{afx}_{audio_idx}"
                    dir_name = opj(
                        self.save_dir, f"epoch_{str(self.current_epoch).zfill(2)}"
                        )
                    os.makedirs(dir_name, exist_ok=True)
                    name = opj(dir_name, tag)
                    self.save_audio(name + "_enhanced.wav", pred)
                    self.save_audio(name + "_clean.wav", clean_audio)
                    self.save_audio(name + "_noisy.wav", noisy_audio)
#                     plot_audio(pred, name + "_spec.png", sr=self.target_sr)

    def on_validation_epoch_end(self):
        if self.current_epoch % self.save_every_n_epochs == 0 :
            for k, v in self.metric_history.items():
                avg = np.sum(v) / len(v)
                highest = np.max(v)
                lowest = np.min(v)

                self.log(k + '_avg', avg)
                self.log(k + '_highest', highest)
                self.log(k + '_lowest', lowest)

    def save_audio(self, name, audio):
        audio = audio.detach().cpu().numpy()
        audio = audio / np.max(np.abs(audio)) # peak normalization
        sf.write(name, audio, self.target_sr)

    """
    ==========
    Optimizers
    ==========
    """

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.model.parameters())

    """
    ==================
    EMA Related Method
    ==================
    """

    def on_load_checkpoint(self, checkpoint):
        print("EMA state_dict will be loaded")
        ema = checkpoint.get("ema", None)
        self.ema.load_state_dict(checkpoint["ema"])

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    # Override .to(), .eval(), .train() methods to adapt ema params
    def to(self, *args, **kwargs):
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def eval(self):
        res = super().train(False)
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())

    """
    ===================
    Pretrained Encoder
    ===================
    """
    @staticmethod
    def load_pretrained_model(solver, ckpt, device='cuda', load_ema=True):
        '''
        Load checkpoint (.ckpt) for pl.LightningModule
        '''
        print("=================================")
        print("Pretrained Model is Loading...")
        pretrained_module = solver.load_from_checkpoint(ckpt)
        pretrained_module.eval()
        pretrained_module.to(device)
        print("Trained Model Imported")
        print("=================================")
        return pretrained_module

    def load_config(self, config_dir, config_file):
        config_name = opj(config_dir, config_file + '.yaml')
        configs = OmegaConf.to_object(OmegaConf.load(config_name))
        return configs
