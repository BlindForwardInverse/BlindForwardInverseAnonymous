import os
import pickle
from glob import glob
from functools import partial
import random

import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import torch.nn as nn

from einops import rearrange
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm

from diffusion_sampler.sde_class_hybrid import ProbODEFlowHybrid, sigma_scheduler
from diffusion_sampler.sde_solver_hybrid import Heun2ndSamplerHybrid
# from models.operator_models.mftaa_encoder import MFTAAEncoder, ToAfxToken, ToAfxEmbedding
from models.diffusion_models.unet import Diffusion2dUnet, DiffusionHybridUnet
from models.operator_models.discriminator import MultiResolutionDiscriminator, PatchGANDiscriminator, DiscriminatorWrapper

from .operator_solver_hybrid import OperatorSolverHybrid
from omegaconf import OmegaConf

from loss import EDMLoss
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

class DiffusionHybridSolver(pl.LightningModule):
    def __init__(
        self, save_dir=None, lr=1e-4, 
        # Signal Spec
        target_sr=44100, audio_len=97792,  n_fft=2046, win_length=2046, hop_length=512,
        # Model
        unet_channels=128, cond_dim=256, embedding_size=512,
        use_discriminator=True,
        learnable_sum=True, detach_spec=False, cat_global_condition=False, norm_local_condition=True,
        sample_from_log_normal=False,
        # Reference Encoder
        use_ref_encoder=False, config_name=None, config_path=None,
        # Diffusion Sampler
        diffusion_steps=64, sigma_min=3e-4, sigma_max=3, train_rho=10, rho=13, sigma_data=0.0126, 
        # Valid
        full_afx_types = 'dummy', save_audio_per_every_n_epochs=1,
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
        self.detach_spec = detach_spec
        self.sample_from_log_normal = sample_from_log_normal

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.diffusion_steps = diffusion_steps

        self.save_audio_per_every_n_epochs = save_audio_per_every_n_epochs

        # transform
        self.spec_to_wav = partial(spec_transform, transform_type='unpower_ri', 
                                   n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
                                   length=self.audio_len)
        self.wav_to_spec = partial(audio_transform, transform_type='power_ri',
                                   n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
                                  )
        # Training
        self.model = DiffusionHybridUnet(
                        unet_channels = unet_channels,
                        cond_dim = cond_dim,
                        use_ref_encoder = use_ref_encoder,
                        unet_dim_mults=(1, 1, 2, 2, 4),
                        num_resnet_blocks=(2, 2, 4, 4, 8),
                        embedding_size=512,
                        num_layer_attns=2,
                        num_layer_cross_attns=2,
                        cat_global_condition=cat_global_condition,
                        norm_local_condition=norm_local_condition,
                )

        self.learnable_sum = learnable_sum
        self.weight = nn.Sequential(nn.Linear(embedding_size, embedding_size),
                                    nn.SiLU(),
                                    nn.Linear(embedding_size, 2))

        self.loss_fn = EDMLoss()
        self.effectwise_metric = EffectWiseSEMetric(full_afx_types=full_afx_types, sr=target_sr)

        # SDE
        self.sde = ProbODEFlowHybrid(f_theta = self.model, weight=self.weight,
                                   sigma_min = sigma_min,
                                   sigma_max = sigma_max,
                                   rho = rho,
                                   sigma_data = sigma_data,
                                   spec_to_wav = self.spec_to_wav,
                                   wav_to_spec = self.wav_to_spec,
                                   **kwargs)
        self.sampler = Heun2ndSamplerHybrid(self.sde,
                                      diffusion_steps=diffusion_steps,
                                      **kwargs)
        self.sigma_list = sigma_scheduler(N=1024, 
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

        # Load Pretrained Operator
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

    def forward(self, dry_spec, wet_spec, dry_wav, wet_wav):
        # y : clean spectrogram data, x : perturbed data (x = spec + n)
        # This follows the convention in 'Elucidating the Design Space~~' paper.
        device = wet_spec.device
        batch_size = wet_spec.shape[0]
        '''
        -----------------------------------
        sample sigma as the proposed paper
        -----------------------------------
        '''
        if self.sample_from_log_normal:
            log_sigma = torch.randn(batch_size, device=device) * 1.5 - 5.0
            sigma = torch.clamp(torch.exp(log_sigma), min=self.sigma_min, max=self.sigma_max)
        else:
            sigma = torch.tensor(random.choices(self.sigma_list, k=batch_size), device=device) 
        n_wav = reshape_x(sigma, 2) * torch.randn_like(dry_wav)
        perturbed_wav = dry_wav + n_wav

        c_skip, c_out, c_in, c_noise = self.sde.get_coeff(sigma)

        # Estimate Loss
        preconditioned_wav = reshape_x(c_in, 2) * perturbed_wav
        preconditioned_spec = self.wav_to_spec(preconditioned_wav)

        model_input_spec = torch.cat([preconditioned_spec, wet_spec], dim=-1)
        model_input_wav = torch.stack([preconditioned_wav, wet_wav], dim=1)
        global_condition, local_condition = self.get_condition(wet_wav)
        model_output_spec, model_output_wav = self.model(model_input_spec,
                                                         model_input_wav,
                                                         c_noise,
                                                         global_condition,
                                                         local_condition)
        if self.use_ref_encoder and self.learnable_sum:
            weight = self.weight(global_condition.detach())
            model_output_transformed = self.spec_to_wav(model_output_spec)
            if self.detach_spec : model_output_transformed = model_output_transformed.detach()
            model_output_summed = rearrange(weight[:, 0], 'b -> b 1') * model_output_wav +\
                                  rearrange(weight[:, 1], 'b -> b 1') * model_output_transformed
        else:
            model_output_summed = torch.mean(model_output_wav + self.spec_to_wav(model_output_spec))

        target_wav = (1 / reshape_x(c_out, 2)) * (dry_wav - reshape_x(c_skip, 2) * perturbed_wav)
        target_spec = self.wav_to_spec(target_wav)

        denoised = reshape_x(c_skip, 2) * dry_wav + reshape_x(c_out, 2) * model_output_summed
        return (model_output_spec, model_output_summed,
                target_spec, target_wav,
                denoised)

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
        model_output_spec, model_output_summed, target_spec, target_summed, denoised = \
            self(dry_spec, wet_spec, dry_wav, wet_wav)

        diffusion_loss_spec = self.loss_fn(model_output_spec, target_spec)
        diffusion_loss_summed = self.loss_fn(model_output_summed, target_summed)
        diffusion_loss = 0.5 * (diffusion_loss_spec + diffusion_loss_summed)
        self.log("train_loss", diffusion_loss, prog_bar=True, on_epoch=True)

        # Discriminator
        if self.use_discriminator:
            # Call optimizers
            optim_gen, optim_disc = self.optimizers()
            # Discriminator Step
            disc_real_logits = self.discriminator(dry_wav)
            disc_fake_logits = self.discriminator(denoised.detach())
            disc_loss, _ = self.discriminator_loss(disc_real_logits, disc_fake_logits)
            
            self.log("disc_loss", disc_loss, prog_bar=True, batch_size=dry_spec.shape[0], on_epoch=True)
            optim_disc.zero_grad()
            self.manual_backward(disc_loss)
            optim_disc.step()

            # Generator Step
            gen_fake_logits = self.discriminator(denoised)
            generator_loss = self.generator_loss(gen_fake_logits)
            self.log("generator_loss", generator_loss, prog_bar=True, batch_size=dry_spec.shape[0], on_epoch=True)

            total_gen_loss = diffusion_loss + generator_loss
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
            global_condition, local_condition = self.get_condition(wet_wav)
            generated_audio = self.sampler(y=wet_spec, y_wav=wet_wav, t_embedding=global_condition, c_embedding=local_condition)
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
        model_params = list(self.model.parameters())
        model_params += list(self.weight.parameters())
        return model_params

    def configure_optimizers(self):
        optimizers = []
        model_params = self.collect_model_params()
        model_optim = torch.optim.Adam(params=model_params, lr=self.lr)
        optimizers.append(model_optim)
        if self.use_discriminator:
            disc_optim = torch.optim.AdamW(params=self.discriminator.parameters(), lr=self.lr, weight_decay=0.01)
            optimizers.append(disc_optim)
        return optimizers

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update()

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
