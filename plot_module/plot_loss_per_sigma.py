from data.valid_dataset import ValidDataset
from solver.waveform_edm import EDMSolver
from edm_module.sde_class import ProbODEFlow
from loss import ScoreMatchingLoss, EDMLoss
from inference import load_pretrained_model
from audio_utils.audio_processing import audio_preprocessing

from torch.utils.data import DataLoader
from tensor_utils import reshape_x
import torch
import torch.nn as nn

from omegaconf import OmegaConf
from einops import rearrange
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

class Plotter:
    def __init__(
        self,
        task="identity",
        modality="speech",
        diffusion_steps=256,
        rho=13,
        sigma_min=0.0001,
        sigma_max=1,
        sigma_data=0.057,
        t_min=0.05,
    ):
        self.rho = rho
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data

        self.t_min = t_min
        self.diffusion_steps = diffusion_steps
        self.device = "cuda"

        # Preparation
        pretrained = self.get_pretrained_model(task, modality)
        self.model = pretrained.model.to(self.device)
        self.sampler = pretrained.sampler
        self.sde = ProbODEFlow(rho, sigma_min, sigma_max, sigma_data)
        self.sigmas = self.sde.sigma_scheduler(diffusion_steps)

    def get_pretrained_model(self, task, modality):
        ckpt_dir = OmegaConf.load("configs/pretrained_ckpt_path.yaml")
        ckpt = ckpt_dir[modality]
        ckpt = "/ssd3/doyo/ckpt/1121_rho13_schurn1/epoch=27-step=84000.ckpt"
        if task == "identity":
            return load_pretrained_model(EDMSolver, ckpt, self.device, eval_mode=False)

    def plot_sigma_scheduler(self):
        sigmas = self.sigmas
        N = self.diffusion_steps

        plt.plot(range(N + 1), sigmas, marker=",")
        plt.title("Sigma Scheduler Function")
        plt.xlabel("Diffusion steps")
        plt.ylabel("Sigma")
        plt.grid(True)
        plt.savefig("plot/sigma_scheduler.png")

    @torch.no_grad()
    def plot_score_loss_per_sigma(self):
        sigmas = self.sigmas
        N = self.diffusion_steps

        log_sigmas = []
        losses = []
        loss_fn = EDMLoss()
        ref_speech = '/ssd4/inverse_problem/speech/speech16k/voicebank-demand-16k/clean_trainset_wav/p226_001.wav'
        ref_speech, _ = sf.read(ref_speech)
        ref_speech = audio_preprocessing(ref_speech, 45056, 16000, 16000)
        ref_speech = torch.Tensor(ref_speech).to(self.device)
        ref_speech = rearrange(ref_speech, 't -> 1 t')

        for sigma in tqdm(sigmas):
            n = sigma * torch.rand_like(ref_speech)
            x = ref_speech + n
            sigma = torch.tensor(sigma)
            c_skip, c_out, c_in, c_noise = self.sde.get_coeff(sigma)
            f_theta = self.model(c_in * x, c_noise.unsqueeze(0))
            target = 1 / c_out * (ref_speech - c_skip * x)

            loss = loss_fn(f_theta, target)
            losses.append(loss.item())
            log_sigmas.append(torch.log(sigma).cpu().numpy())

        # score loss per sigma
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(sigmas, losses, marker=",")
        plt.title("Score Loss per Sigma")
        plt.xlabel("Sigma")
        plt.ylabel("Loss")
        plt.grid(True)

        # score loss per log sigma
        plt.subplot(1, 2, 2)
        plt.plot(log_sigmas, losses, marker=",")
        plt.title("Score Loss per Log Sigma")
        plt.xlabel("Log Sigma")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("plot/loss_per_sigma.png")
