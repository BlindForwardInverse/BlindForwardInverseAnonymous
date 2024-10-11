import os

import jax
import jax.numpy as jnp
import numpy as np
import pytorch_lightning as pl
import soundfile as sf
import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm
from functools import partial

from models.operator_models.unet import UnetWrapperHybrid
from models.operator_models.mtfaa_encoder import MTFAAEncoder, ToAfxToken, ToAfxEmbedding
from models.operator_models.discriminator import MultiResolutionDiscriminator, PatchGANDiscriminator, DiscriminatorWrapper

from torch_ema import ExponentialMovingAverage
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

from metric import MetricHandlerOperator, EffectWiseWetMetric
from loss import (
    WeightedSTFTLoss,
    WeightedSTFTLossEachDomain,
    AdversarialLoss
    )

from utils.audio_transform import spec_transform, audio_transform

opj = os.path.join

class OperatorSolverHybrid(pl.LightningModule):
    def __init__(
        self, save_dir=None, clip_grad=True, save_audio_per_n_epochs=5,
        # Signal Spec
        batch_size=6, audio_len=97792, n_fft=2046, win_length=2046, hop_length=512, target_sr=44100, 
        # Training Strategy
        cat_ref=True,
        use_ref_encoder=True,
        hybrid_bridge=False,
        learnable_sum=True,
        partial_mean_pooled=False,
        use_discriminator=False, freeze_disc_steps=30000, 
        detach_global_condition=False,
        # Valid Strategy
        full_afx_types=None,
        # Loss Function
        use_hinge_loss=True, mse_each_domain=False,
        mse_weight=10., factor_sc=1, factor_mag=1, factor_phase=0.1, pred_loss_weight=0.1,
        # Model
        unet_channels=128, cond_dim=512, embedding_size=512,
        discriminator_type='MRD',
        # LR Scheduler
        lr=1e-3, use_lr_schedule=False, warmup_steps=1000, first_cycle_steps=10000,
        max_lr = 5e-4, min_lr=1e-5, gamma=0.9,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.save_dir = save_dir
        self.clip_grad = clip_grad
        self.save_audio_per_n_epochs = save_audio_per_n_epochs

        self.batch_size = batch_size
        self.audio_len = audio_len
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.target_sr = target_sr

        self.cat_ref = cat_ref
        self.use_ref_encoder = use_ref_encoder
        self.detach_global_condition = detach_global_condition

        # For lr schedule
        self.lr = lr
        self.use_lr_schedule = use_lr_schedule
        self.warmup_steps = warmup_steps
        self.first_cycle_steps = first_cycle_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.gamma = gamma

        self.use_discriminator = use_discriminator
        self.discriminator_type = discriminator_type

        # Model
        # -----------------------------------------------------------
        if use_ref_encoder:
            self.fx_encoder = MTFAAEncoder(cond_dim=cond_dim,
                                        embedding_size=embedding_size, 
                                        win_len=win_length,
                                        win_hop=hop_length,
                                        **kwargs)
            self.to_global_condition = ToAfxEmbedding(
                                        cond_dim=cond_dim,
                                        embedding_size=embedding_size,
                                        partial_mean_pooled=partial_mean_pooled
                                        )
            self.to_local_condition= ToAfxToken(
                                        cond_dim=cond_dim,
                                        **kwargs)

        if use_discriminator:
            if self.discriminator_type == 'MRD':
                self.discriminator = MultiResolutionDiscriminator()
            elif self.discriminator_type == 'patch':
                self.discriminator = PatchGANDiscriminator()
            elif self.discriminator_type == 'MRD+MPD':
                self.discriminator = DiscriminatorWrapper()
            self.pred_loss_weight = pred_loss_weight
            self.freeze_disc_steps = freeze_disc_steps
            self.automatic_optimization = False

            self.adversarial_loss =   AdversarialLoss(use_hinge_loss=use_hinge_loss)
            self.discriminator_loss = self.adversarial_loss.discriminator_loss
            self.generator_loss =     self.adversarial_loss.generator_loss

        self.unet = UnetWrapperHybrid(unet_channels = unet_channels,
                                      cond_dim = cond_dim,
                                      learnable_sum = learnable_sum,
                                      use_ref_encoder = use_ref_encoder,
                                      embedding_size=embedding_size,
                                      cat_ref=cat_ref,
                                      hybrid_bridge=hybrid_bridge,
                                      **kwargs)
        # ---------------------------------------------------------------
        # Metric 
        self.effectwise_metric = EffectWiseWetMetric(full_afx_types)

        # Loss
        self.mse_weight = mse_weight
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.mse_each_domain = mse_each_domain
        self.prediction_loss = WeightedSTFTLoss(n_fft=n_fft,
                                                 win_length=win_length,
                                                 hop_length=hop_length,
                                                 factor_sc=factor_sc,
                                                 factor_mag=factor_mag,
                                                 factor_phase=factor_phase,
                                                 mse_weight=mse_weight) if not mse_each_domain else\
                               WeightedSTFTLossEachDomain(
                                                 n_fft=n_fft,
                                                 win_length=win_length,
                                                 hop_length=hop_length,
                                                 factor_sc=factor_sc,
                                                 factor_mag=factor_mag,
                                                 factor_phase=factor_phase,
                                                 mse_weight=mse_weight)

        self.learnable_sum = learnable_sum
        self.weight = nn.Sequential(nn.Linear(embedding_size, embedding_size),
                                    nn.SiLU(),
                                    nn.Linear(embedding_size, 2))

        # EMA
        self.ema_decay = 0.999
        self.ema = ExponentialMovingAverage(
            self.collect_model_params(), decay = self.ema_decay
        )

        # transform
        self.spec_to_wav = partial(spec_transform, transform_type='unpower_ri', 
                                   n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
                                   length=self.audio_len)
        self.wav_to_spec = partial(audio_transform, transform_type='power_ri',
                                   n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
                                  )


    def forward(self, dry_tar_spec, dry_tar_wav, wet_ref, wet_ref_spec=None):
        # Unet Input
        spec_input = torch.cat([dry_tar_spec, wet_ref_spec], dim=-1) if self.cat_ref else dry_tar
        wav_input = torch.stack([dry_tar_wav, wet_ref], dim=1) if self.cat_ref else dry_tar_wav
        global_condition, local_condition = None, None
        if self.use_ref_encoder:
            z = self.fx_encoder(wet_ref)
            local_condition = self.to_local_condition(z)
            global_condition = self.to_global_condition(z)
        pred_spec, pred_wav = self.unet(spec_input, wav_input, t=global_condition, condition=local_condition)

        if self.learnable_sum:
            if self.detach_global_condition:
                weight = self.weight(global_condition.detach())
            else:
                weight = self.weight(global_condition)
            summed_wav = rearrange(weight[:, 0], 'b -> b 1') * pred_wav +\
                         rearrange(weight[:, 1], 'b -> b 1') * self.spec_to_wav(pred_spec)
        else:
            summed_wav = 0.5 * (pred_wav + self.spec_to_wav(pred_spec))
        return pred_spec, pred_wav, summed_wav

    def training_step(self, batch, _):
        dry_tar_wav, wet_tar_wav, dry_tar_spec, wet_tar_spec, wet_ref_wav, wet_ref_spec =\
            [batch.get(key) for key in ["dry_tar", "wet_tar", "dry_tar_spec", "wet_tar_spec", "wet_ref", "wet_ref_spec"]]
        out_spec, out_wav, pred_tar_wav = self(dry_tar_spec, dry_tar_wav, wet_ref_wav, wet_ref_spec)

        # Loss Function
        if self.mse_each_domain : pred_loss, pred_loss_dict = self.prediction_loss(out_spec, wet_tar_spec, pred_tar_wav, wet_tar_wav, tag='_blend')
        else:                     pred_loss, pred_loss_dict = self.prediction_loss(pred_tar_wav, wet_tar_wav, pred_tar_wav, wet_tar_wav)
        self.log("pred_loss", pred_loss, prog_bar=True, batch_size=self.batch_size, on_epoch=True)
        self.log_dict(pred_loss_dict, prog_bar=True, batch_size=self.batch_size, on_epoch=True)

        if self.use_discriminator:
            # Call optimizers
            optim_gen, optim_disc = self.optimizers()

            # Discriminator Step
            if self.discriminator_type in ['MRD', 'MPD']:
                disc_real_logits = self.discriminator(wet_tar_wav)
                disc_fake_logits = self.discriminator(pred_tar_wav.detach())
                disc_loss, disc_loss_dict = self.discriminator_loss(disc_real_logits, disc_fake_logits)
                self.log_dict(disc_loss_dict, prog_bar=True, batch_size=self.batch_size, on_epoch=True)

            elif self.discriminator_type == 'MRD+MPD':
                real_mrd, real_mpd = self.discriminator(wet_tar_wav)
                fake_mrd, fake_mpd = self.discriminator(pred_tar_wav.detach())
                mpd_loss, mpd_loss_dict = self.discriminator_loss(real_mpd, fake_mpd, disc_type='mpd')
                mrd_loss, mrd_loss_dict = self.discriminator_loss(real_mrd, fake_mrd, disc_type='mrd')

                disc_loss = 0.5 * (mpd_loss + mrd_loss)
                self.log_dict(mpd_loss_dict, batch_size=self.batch_size, on_epoch=True)
                self.log_dict(mrd_loss_dict, batch_size=self.batch_size, on_epoch=True)

            self.log("disc_loss", disc_loss, prog_bar=True, batch_size=self.batch_size, on_epoch=True)
            optim_disc.zero_grad()
            self.manual_backward(disc_loss)
            optim_disc.step()
            
            # Generator Step
            generator_loss = 0.0
            if self.global_step > self.freeze_disc_steps:
                if self.discriminator_type in ['MRD', 'MPD']:
                    gen_fake_logits = self.discriminator(pred_tar_wav)
                    generator_loss = self.generator_loss(gen_fake_logits)
                elif self.discriminator_type == 'MRD+MPD':
                    gen_fake_mrd, gen_fake_mpd = self.discriminator(pred_tar_wav)
                    gen_loss_mrd = self.generator_loss(gen_fake_mrd)
                    gen_loss_mpd = self.generator_loss(gen_fake_mpd)
                    generator_loss = 0.5 * (gen_loss_mrd + gen_loss_mpd)

            total_gen_loss = pred_loss * self.pred_loss_weight + generator_loss
            self.log("generator_loss", generator_loss, prog_bar=True, batch_size=self.batch_size, on_epoch=True)
            self.log("total_gen_loss", total_gen_loss, prog_bar=True, batch_size=self.batch_size, on_epoch=True)

            optim_gen.zero_grad()
            self.manual_backward(total_gen_loss)
            if self.clip_grad : self.clip_gradients(optim_gen, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            optim_gen.step()
            self.ema.update()
        else:
            return pred_loss

    def validation_step(self, batch, idx):
        with self.ema.average_parameters():
            dry_tar_wav, wet_tar_wav, dry_tar_spec, wet_tar_spec, dry_ref_wav, wet_ref_wav, wet_ref_spec =\
                [batch.get(key) for key in\
                 ["dry_tar", "wet_tar", "dry_tar_spec", "wet_tar_spec", "dry_ref", "wet_ref", "wet_ref_spec"]]
            valid_types, afx_names, afx_types, audio_nums = \
                [batch[key] for key in ["valid_type", "afx_name", "afx_type", "audio_idx"]]

            _, _, pred_tar_wav = self(dry_tar_spec, dry_tar_wav, wet_ref_wav, wet_ref_spec)
            self.effectwise_metric.update(pred_tar_wav, wet_tar_wav, afx_types)

            if self.current_epoch % self.save_audio_per_n_epochs == 0:
                for _pred, _wet, _dry_ref, _wet_ref, _dry, _afx, _audio_num in\
                    zip(pred_tar_wav, wet_tar_wav, dry_ref_wav, wet_ref_wav, dry_tar_wav, afx_names, audio_nums):
                        self.save_audio(_dry_ref, _afx, _audio_num, tag='0dry_ref')
                        self.save_audio(_wet_ref, _afx, _audio_num, tag='1wet_ref')
                        self.save_audio(_dry, _afx, _audio_num, tag='2dry')
                        self.save_audio(_wet, _afx, _audio_num, tag='3wet')
                        self.save_audio(_pred, _afx, _audio_num, tag='4pred')
                    
    def on_validation_epoch_end(self):
        metric = self.effectwise_metric.compute()
        self.log_dict(metric)
        self.effectwise_metric.reset()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update()

    def save_audio(self, audio, afx, audio_num, tag):
        audio = audio.detach().cpu().numpy()
        save_name = f"epoch_{self.current_epoch}_{afx}_{audio_num}_{tag}.wav"
        dir_name = opj(self.save_dir, f"epoch_{str(self.current_epoch).zfill(3)}")
        os.makedirs(dir_name, exist_ok=True)
        sf.write(opj(dir_name, save_name), audio, self.target_sr)

    def collect_model_params(self):
        """
        Collect model parameters except the discriminator
        """
        model_params = list(self.unet.parameters())
        if self.use_ref_encoder:
            model_params = model_params\
                           + list(self.fx_encoder.parameters())\
                           + list(self.to_local_condition.parameters())\
                           + list(self.to_global_condition.parameters())
        if self.learnable_sum:
            model_params += list(self.weight.parameters())

        return model_params

    def configure_optimizers(self):
        model_params = self.collect_model_params()
        model_optim = torch.optim.AdamW(params=model_params, lr=self.lr, weight_decay=0.01)
        if self.use_discriminator:
            disc_optim = torch.optim.AdamW(params=self.discriminator.parameters(), lr=self.lr, weight_decay=0.01)
            return model_optim, disc_optim
        else:
            if self.use_lr_schedule:
                lr_scheduler = CosineAnnealingWarmupRestarts(optim,
                                                  first_cycle_steps=self.first_cycle_steps,
                                                  cycle_mult=1.0,
                                                  max_lr=self.max_lr,
                                                  min_lr=self.min_lr,
                                                  warmup_steps=self.warmup_steps,
                                                  gamma=self.gamma)
                config = {"scheduler" : lr_scheduler, "interval" : "step"}
                return [optim], [config]

            else:
                return model_optim

    """
    EMA Related Methods
    """
    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint:
            print("EMA state_dict will be loaded")
            ema = checkpoint["ema"]
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            print("No ema checkpoint exists")

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
