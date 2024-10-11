import os
import pickle
from glob import glob
from functools import partial
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

from models.diffusion_models.unet import Diffusion2dUnet
from models.operator_models.unet import UnetWrapperHybridWithWeight
from models.operator_models.discriminator import MultiResolutionDiscriminator, PatchGANDiscriminator, DiscriminatorWrapper
from solvers.operator_solver_hybrid import OperatorSolverHybrid

from omegaconf import OmegaConf

from loss import EDMLoss, WeightedSTFTLoss
from loss import (
    AdversarialLoss
    )
from metric import EffectWiseSEMetric

# from solver_operator import UnetSolver
from plot_module.plot_spectrogram import plot_audio

from utils.torch_utils import reshape_x, load_pretrained_model
from utils.audio_normalization import absolute_normalize, rms_normalize
from utils.audio_transform import audio_transform, spec_transform

opj = os.path.join

class DiffusionSpecProj(pl.LightningModule):
    def __init__(
        self, save_dir=None, lr=1e-4, 
        # Signal Spec
        target_sr=44100, audio_len=97792,  n_fft=2046, win_length=2046, hop_length=512,
        # Model
        unet_channels=128, cond_dim=256, embedding_size=512,
        use_discriminator=True,
        # Reference Encoder
        use_ref_encoder=False, config_name=None, config_path=None,
        # Diffusion Sampler
        diffusion_steps=64, sigma_min=3e-4, sigma_max=3, train_rho=10, rho=13, sigma_data=0.0126, 
        # Valid
        full_afx_types = None, save_audio_per_every_n_epochs=1,
        ema_decay=0.999,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_encoder'])
        self.save_dir = save_dir
        self.lr = lr

        self.target_sr = target_sr
        self.audio_len = audio_len
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.use_ref_encoder = use_ref_encoder
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.diffusion_steps = diffusion_steps

        self.save_audio_per_every_n_epochs = save_audio_per_every_n_epochs

        # Training
        self.model = Diffusion2dUnet(
                        unet_channels = unet_channels,
                        cond_dim = cond_dim,
                        use_ref_encoder = use_ref_encoder,
                        unet_dim_mults=(1, 1, 2, 2, 4),
                        num_resnet_blocks=(2, 2, 4, 4, 8),
                        embedding_size=512,
                        num_layer_attns=2,
                        num_layer_cross_attns=2,
                )
    
        self.projector = UnetWrapperHybridWithWeight(unet_channels = unet_channels,
                                               cond_dim = cond_dim,
                                               use_ref_encoder = True,
                                               embedding_size = 512,
                                               cat_ref=True,
                                               hybrid_bridge=False)

        self.loss_fn = EDMLoss()
        self.projection_loss = WeightedSTFTLoss(n_fft=n_fft,
                                    win_length=win_length,
                                    hop_length=hop_length,
                                    factor_sc=1,
                                    factor_mag=1,
                                    factor_phase=0,
                                    mse_weight=50)

        self.effectwise_metric = EffectWiseSEMetric(full_afx_types=full_afx_types, sr=target_sr)

        # SDE
        self.sde = ProbODEFlow(f_theta = self.model,
                               projector=self.projector,
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

        self.use_discriminator = use_discriminator
        if use_discriminator:
            self.discriminator = MultiResolutionDiscriminator()
            self.automatic_optimization = False

            self.adversarial_loss =   AdversarialLoss(use_hinge_loss=False)
            self.discriminator_loss = self.adversarial_loss.discriminator_loss
            self.generator_loss =     self.adversarial_loss.generator_loss

        if use_ref_encoder:
            self.pretrained_encoder = load_pretrained_model(OperatorSolverHybridV2 if 'hb2' in config_name else\
                                                       OperatorSolverHybrid,
                                                       config_path=config_path,
                                                       config_name=config_name,
                                                       freeze=True,
                                                       device='cuda')
            self.fx_encoder = self.pretrained_encoder.fx_encoder
            self.to_local_condition = self.pretrained_encoder.to_local_condition
            self.to_global_condition = self.pretrained_encoder.to_global_condition
        else:
            print("No pretrained_encoder")

        # EMA
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(
            self.collect_model_params(), decay=self.ema_decay
        )

        # transform
        self.spec_to_wav = partial(spec_transform, transform_type='unpower_ri', 
                                   n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
                                   length=self.audio_len)
        self.wav_to_spec = partial(audio_transform, transform_type='power_ri',
                                   n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
                                  )

    def forward(self, dry_spec, wet_spec, global_condition, local_condition):
        # y : clean spectrogram data, x : perturbed data (x = spec + n)
        # This follows the convention in 'Elucidating the Design Space~~' paper.
        device = wet_spec.device
        batch_size = wet_spec.shape[0]
        '''
        -----------------------------------
        sample sigma as the proposed paper
        -----------------------------------
        '''
#         log_sigma = torch.randn(batch_size, device=device) * 1.5 - 1.5
#         sigma = torch.clamp(torch.exp(log_sigma), min=self.sigma_min, max=self.sigma_max)
        sigma = torch.tensor(random.choices(self.sigma_list, k=batch_size), device=device) 
        sigma = reshape_x(sigma, 4)  # [32, 1, 1, 1]
        n = sigma * torch.randn_like(dry_spec)
        perturbed = dry_spec + n
        c_skip, c_out, c_in, c_noise = self.sde.get_coeff(sigma)

        # Estimate Loss
        preconditioned = c_in * perturbed
        model_input = torch.cat([preconditioned, wet_spec], dim=-1)
        model_output = self.model(model_input,
                                  c_noise.squeeze(),
                                  global_condition,
                                  local_condition)

        denoised = c_skip * dry_spec + c_out * model_output
        target = (1 / c_out) * (dry_spec - c_skip * (dry_spec + n))
        return model_output, target, denoised

    def projection(self, denoised, wet_spec, wet_wav, global_condition, local_condition):
        denoised_wav = self.spec_to_wav(denoised)
        spec_input = torch.cat([denoised, wet_spec], dim=-1)
        wav_input = torch.stack([denoised_wav, wet_wav], dim=1)
        projected = self.projector(spec_input, wav_input, global_condition, local_condition)
        return projected

    def get_condition(self, degraded_wav):
        if self.use_ref_encoder:
            fx_latent = self.fx_encoder(degraded_wav)
            local_condition = self.to_local_condition(fx_latent)
            global_condition = self.to_global_condition(fx_latent)
        else:
            local_condition, global_condition = None, None
        return global_condition, local_condition

    def training_step(self, batch, _):
        dry_spec, dry_wav, wet_spec, wet_wav = [batch[key] for key in ['dry_spec', 'dry_wav', 'wet_spec', 'wet_wav']]
        global_condition, local_condition = self.get_condition(wet_wav)

        model_output, target, denoised = self(dry_spec, wet_spec, global_condition, local_condition)
        diffusion_loss = self.loss_fn(model_output, target)
        self.log("train_loss", diffusion_loss, prog_bar=True, on_epoch=True)

        # Projection
        projected_wav = self.projection(denoised.detach(), wet_spec, wet_wav, global_condition, local_condition)
#         projected_wav = self.spec_to_wav(projected)
        projection_loss, loss_dict = self.projection_loss(projected_wav, dry_wav, projected_wav, dry_wav)
        self.log("projection_loss", projection_loss, prog_bar=True, on_epoch=True)
        self.log_dict(loss_dict)

        # Discriminator
        if self.use_discriminator:
            # Call optimizers
            optim_gen, optim_disc = self.optimizers()
            # Discriminator Step
            disc_real_logits = self.discriminator(dry_wav)
            disc_fake_logits = self.discriminator(projected_wav.detach())
            disc_loss, _ = self.discriminator_loss(disc_real_logits, disc_fake_logits)
            
            self.log("disc_loss", disc_loss, prog_bar=True, batch_size=dry_spec.shape[0], on_epoch=True)
            optim_disc.zero_grad()
            self.manual_backward(disc_loss)
            optim_disc.step()

            # Generator Step
            gen_fake_logits = self.discriminator(projected_wav)
            generator_loss = self.generator_loss(gen_fake_logits)
            self.log("generator_loss", generator_loss, prog_bar=True, batch_size=dry_spec.shape[0], on_epoch=True)

            total_gen_loss = diffusion_loss + projection_loss + generator_loss
            optim_gen.zero_grad()
            self.manual_backward(total_gen_loss)
            optim_gen.step()

            self.ema.update()
        else:
            return diffusion_loss

    def validation_step(self, batch, idx):
        with self.ema.average_parameters():
            dry_spec, dry_wav, wet_spec, wet_wav = [batch[key] for key in ['dry_tar_spec', 'dry_tar', 'wet_tar_spec', 'wet_tar']]
            valid_type, afx_name, afx_types, audio_nums = [batch[key] for key in ['valid_type', 'afx_name', 'afx_type', 'audio_idx']]
            shape = wet_spec.shape
            global_condition, local_condition = self.get_condition(wet_wav)
            generated = self.sampler(shape, y=wet_spec, t_embedding=global_condition, c_embedding=local_condition)
            generated_audio = self.spec_to_wav(generated)

            self.effectwise_metric.update(generated_audio, dry_wav, afx_types)

            # Save Audio
            if self.current_epoch % self.save_audio_per_every_n_epochs == 0:
                for _dry, _wet, _pred, _afx, _audio_num in\
                    zip(dry_wav, wet_wav, generated_audio, afx_name, audio_nums):
                        self.save_audio(_dry, _afx, _audio_num, tag='1dry')
                        self.save_audio(_wet, _afx, _audio_num, tag='0noisy')
                        self.save_audio(_pred, _afx, _audio_num, tag='2enhanced')

    def on_validation_epoch_end(self):
        si_sdr, pesq, estoi = self.effectwise_metric.compute()
        self.log_dict(si_sdr)
        self.log_dict(pesq)
#         self.log_dict(estoi)
        self.effectwise_metric.reset()

    def save_audio(self, audio, afx, audio_num, tag):
        audio = audio.detach().cpu().numpy()
        save_name = f'epoch_{self.current_epoch}_{afx}_{audio_num}_{tag}.wav'
        dir_name = opj(self.save_dir, f"epoch_{str(self.current_epoch).zfill(3)}")
        os.makedirs(dir_name, exist_ok=True)

        audio = audio / np.max(np.abs(audio)) # peak normalization
        sf.write(opj(dir_name, save_name), audio, self.target_sr)

    """
    ==========
    Optimizers
    ==========
    """
    def collect_model_params(self):
        return list(self.model.parameters()) + list(self.projector.parameters())

    def configure_optimizers(self):
        optimizers = []
        model_optim = torch.optim.Adam(params=self.collect_model_params(), lr=self.lr)
        optimizers.append(model_optim)
        if self.use_discriminator:
            disc_optim = torch.optim.AdamW(params=self.discriminator.parameters(), lr=self.lr, weight_decay=0.01)
            optimizers.append(disc_optim)
        return optimizers

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.collect_model_params())

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
        self.ema.store(self.collect_model_params())
        self.ema.copy_to(self.collect_model_params())
