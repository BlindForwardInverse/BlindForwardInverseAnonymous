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

from loss import EDMLoss, WeightedSTFTLoss
from metric import SpeechEnhancementMetric

from solver_operator import UnetSolver

from plot_module.plot_spectrogram import plot_audio

from utils.tensor_utils import reshape_x, seed_everything
from utils.audio_normalization import absolute_normalize, rms_normalize
from utils.audio_transform import audio_transform, spec_transform

opj = os.path.join

class DeterministicSolver(pl.LightningModule):
    def __init__(
        self,
        save_dir=None, lr=1e-3, target_sr=44100,
        n_fft=2046, win_length=2046, hop_length=512,
        unet_channels=128, num_resnet_blocks=[2,4,6,8], dim_mults=[1,2,4,8],
        loss_type='mse', mse_weight=50., factor_sc=1, factor_mag=1, factor_phase=0,
        num_layer_attns=2, num_layer_cross_attns=2,
        cond_dim=256, embedding_size=512,
        attn_dim_head=32, attn_heads=16,
        diffusion_steps=64,
        ema_decay=0.999,
        generate_valid_type=['in_dataset_same_mic'],
        save_every_n_epochs=1,
        pretrained_encoder=False,
        use_fx_encoder=False,
        config_dir = 'configs/pretrained_model',
        operator_config_name = 'operator_w_disc_single',
        cat_c_embed = True,
        cat_t_embed = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.save_dir = save_dir
        self.lr = lr
        self.target_sr = target_sr

        self.generate_valid_type = generate_valid_type

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        
        self.save_every_n_epochs = save_every_n_epochs

        # Training
        num_layers = len(dim_mults)
        layer_attns = [False] * (num_layers - num_layer_attns) + [True] * num_layer_attns
        layer_cross_attns = [False] * (num_layers - num_layer_cross_attns) + [True] * num_layer_cross_attns
        self.model = Unet(dim=unet_channels,
                          channels=2,
                          num_resnet_blocks=num_resnet_blocks,
                          dim_mults=dim_mults,
                          cond_dim=cond_dim,
                          layer_attns=layer_attns,
                          layer_cross_attns=layer_cross_attns,
                          attn_dim_head=attn_dim_head,
                          attn_heads=attn_heads,
                          cat_c_embed=cat_c_embed,
                          cat_t_embed=cat_t_embed)

        # Loss Function
        self.loss_type = loss_type
        if self.loss_type == 'mse':
            self.loss_fn = EDMLoss()
        elif self.loss_type == 'mag':
            self.loss_fn = WeightedSTFTLoss(n_fft=n_fft,
                                            win_length=win_length,
                                            hop_length=hop_length,
                                            factor_sc=factor_sc,
                                            factor_mag=factor_mag,
                                            factor_phase=factor_phase,
                                            mse_weight=mse_weight)
        else:
            raise NotImplementedError("Unvalid Error Type")
        self.metric_fn = SpeechEnhancementMetric(sr=target_sr)

        # Load Pretrained Operator
        self.pretrained_encoder = pretrained_encoder
        self.use_fx_encoder = use_fx_encoder
        if use_fx_encoder:
            if pretrained_encoder:
                operator_configs = self.load_config(config_dir, operator_config_name)
                ckpt, model_config = operator_configs['ckpt'], operator_configs['model_params']
                operator_solver = UnetSolver(**model_config)
                self.pretrained_operator = self.load_pretrained_model(operator_solver, ckpt)
                for p in self.pretrained_operator.parameters():
                    p.requires_grad = False
            else:
                self.fx_encoder = MFTAAEncoder(cond_dim=cond_dim,
                                               embedding_size=embedding_size,
                                               win_len=win_length,
                                               win_hop=hop_length,
                                               **kwargs)
                self.to_afx_tokens = ToAfxToken(cond_dim=cond_dim)
                self.to_fx_embedding = ToAfxEmbedding(cond_dim=cond_dim,
                                                      embedding_size=embedding_size,
                                                    )
        # EMA
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(
            self.model.parameters(), decay=self.ema_decay
        )

    def forward(self, clean, degraded, degraded_wav=None):
        # y : clean spectrogram data, x : perturbed data (x = spec + n)
        # This follows the convention in 'Elucidating the Design Space~~' paper.
        device = clean.device
        batch_size = clean.shape[0]

        c_embedding, t_embedding = self.get_embedding_tokens(degraded_wav)
        pred = self.model(degraded,
                          torch.zeros(batch_size, device=device),
                          t_embedding = t_embedding,
                          c_embedding = c_embedding)
        if self.loss_type == 'mse':
            loss = self.loss_fn(pred, clean)
            self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        elif self.loss_type == 'mag':
            pred_wav = spec_transform(pred, transform_type='unpower_ri',
                                      n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
            clean_wav = spec_transform(clean, transform_type='unpower_ri',
                                      n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
            loss, loss_dict = self.loss_fn(pred, clean, pred_wav, clean_wav)
            self.log("train_loss", loss, prog_bar=True, on_epoch=True)
            self.log_dict(loss_dict, prog_bar=True, batch_size=batch_size, on_epoch=True)
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

    def training_step(self, batch, _):
        clean, degraded, degraded_wav = batch['dry'], batch['wet'], batch['wet_wav']
        loss = self(clean, degraded, degraded_wav)
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
                device = noisy_spec.device
                batch_size = shape[0]

                c_embedding, t_embedding = self.get_embedding_tokens(noisy)
                generated = self.model(noisy_spec, torch.zeros(batch_size, device=device),
                                       t_embedding=t_embedding, c_embedding=c_embedding)
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
