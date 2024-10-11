import functools

import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utils.spec_utils import compute_loudness, compute_mag
from utils.audio_transform import spec_transform, audio_transform

'''
Loss Function for Diffusion Model
------------------------------------------------------------------------------------
'''
class EDMLoss(nn.Module):
    def __init__(self, mag_loss=True):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mag_loss = mag_loss
        window='hann_window'
        win_length=2048
        self.window = getattr(torch, window)(win_length, device='cuda:0')

    def forward(self, estimate, target):
        loss = self.mse(estimate, target)
        return loss

"""
GAN Loss function
"""
class AdversarialLoss:
    def __init__(self, use_hinge_loss=False):
        self.use_hinge_loss = use_hinge_loss

    def discriminator_loss(self, disc_reals, disc_fakes, disc_type=''):
        loss = 0
        loss_dict = dict()
        for logit_real, logit_fake in zip(disc_reals, disc_fakes):
            if self.use_hinge_loss:
                d_loss_real = torch.mean(F.relu(1. - logit_real))
                d_loss_fake = torch.mean(F.relu(1. + logit_fake))
                loss += 0.5 * (d_loss_real + d_loss_fake)
            else:
                d_loss_real = torch.mean((1 - logit_real) ** 2)
                d_loss_fake = torch.mean(logit_fake) ** 2
                loss += 0.5 * (d_loss_real + d_loss_fake)


            if disc_type != '': disc_type = '_' + disc_type
            loss_dict['disc_loss_real' + disc_type] = d_loss_real
            loss_dict['disc_loss_fake' + disc_type] = d_loss_fake
        return loss, loss_dict

    def generator_loss(self, disc_fakes):
        loss = 0
        for logit_fake in disc_fakes:
            if self.use_hinge_loss:
                d_loss_fake = -torch.mean(logit_fake)
                loss += d_loss_fake
            else:
                d_loss_fake = torch.mean((1 - logit_fake) ** 2)
                loss += d_loss_fake
        return loss

class DiscriminatorLoss(nn.Module):
    def __init__(self,
                 use_hinge_loss=False):
        super().__init__()
        self.use_hinge_loss = use_hinge_loss

    def forward(self, disc_reals, disc_fakes, disc_type=''):
        loss = 0
        loss_dict = dict()
        for logit_real, logit_fake in zip(disc_reals, disc_fakes):
            if self.use_hinge_loss:
                d_loss_real = torch.mean(F.relu(1. - logit_real))
                d_loss_fake = torch.mean(F.relu(1. + logit_fake))
                loss += 0.5 * (d_loss_real + d_loss_fake)
            else:
                d_loss_real = torch.mean((1 - logit_real) ** 2)
                d_loss_fake = torch.mean(logit_fake) ** 2
                loss += 0.5 * (d_loss_real + d_loss_fake)


            if disc_type != '': disc_type = '_' + disc_type
            loss_dict['disc_loss_real' + disc_type] = d_loss_real
            loss_dict['disc_loss_fake' + disc_type] = d_loss_fake
        return loss, loss_dict

class GeneratorLoss(nn.Module):
    def __init__(self,
                 use_hinge_loss=False):
        super().__init__()
        self.use_hinge_loss = use_hinge_loss

    def forward(self, disc_fakes):
        loss = 0
        for logit_fake in disc_fakes:
            if self.use_hinge_loss:
                d_loss_fake = -torch.mean(logit_fake)
                loss += d_loss_fake
            else:
                d_loss_fake = torch.mean((1 - logit_fake) ** 2)
                loss += d_loss_fake
        return loss

"""
Loss Functions for a Operator Model
------------------------------------------------------------------------------------
"""
class WeightedSTFTLossEachDomain(nn.Module):
    def __init__(self,
                 factor_sc=1.,
                 factor_mag=1.,
                 factor_phase=0.1,
                 n_fft=2046,
                 win_length=2046,
                 hop_length=512,
                 mse_weight=50,
                 disc_weight=0.1):
        super().__init__()
#         self.stft_loss = STFTLoss(fft_size=512, shift_size=128, win_length=512)
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.factor_phase = factor_phase

        self.mss_loss = MultiResolutionSTFTLoss(
                 fft_sizes=[4096, 2048, 1024, 512],
                 hop_sizes=[1024, 512, 256, 128],
                 win_lengths=[4096, 2048, 1024, 512],
                 factor_sc=factor_sc,
                 factor_mag=factor_mag,
                 factor_phase=factor_phase)
        self.mse_weight = mse_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, out_spec, y_spec, out_wav, y_wav, tag=''):
        '''
            x : pred, y : gt
        '''
        loss_dict = {}
        if self.mse_weight > 0:
            mse_wav = self.mse_loss(out_wav, y_wav)
            mse_spec = self.mse_loss(out_spec, y_spec)
        
            mse = (mse_wav + mse_spec) * 0.5 * self.mse_weight
            loss_dict['mse_loss'] = mse

        sc, mag, phase = self.mss_loss(out_wav, y_wav)
        if self.factor_sc > 0 : loss_dict['sc_loss' + tag] = sc
        if self.factor_mag > 0 : loss_dict['mag_loss' + tag] = mag

        total_loss = sum([loss_dict[key] for key in loss_dict])
        return total_loss, loss_dict

class WeightedSTFTLoss(nn.Module):
    def __init__(self,
                 factor_sc=1.,
                 factor_mag=1.,
                 factor_phase=0,
                 n_fft=2046,
                 win_length=2046,
                 hop_length=512,
                 mse_weight=50,
                 disc_weight=0.1):
        super().__init__()
#         self.stft_loss = STFTLoss(fft_size=512, shift_size=128, win_length=512)
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.factor_phase = factor_phase

        self.mss_loss = MultiResolutionSTFTLoss(
                 fft_sizes=[4096, 2048, 1024, 512],
                 hop_sizes=[1024, 512, 256, 128],
                 win_lengths=[4096, 2048, 1024, 512],
                 factor_sc=factor_sc,
                 factor_mag=factor_mag,
                 factor_phase=factor_phase)
        self.mse_weight = mse_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred, true, x_wav=None, y_wav=None, tag=''):
        '''
            x : pred, y : gt
        '''
        loss_dict = {}
        if self.mse_weight > 0:
            mse = self.mse_loss(pred, true) * self.mse_weight
            loss_dict['mse_loss'] = mse

        sc, mag, phase = self.mss_loss(x_wav, y_wav)
        if self.factor_sc > 0 : loss_dict['sc_loss' + tag] = sc
        if self.factor_mag > 0 : loss_dict['mag_loss' + tag] = mag
        total_loss = sum([loss_dict[key] for key in loss_dict])

        return total_loss, loss_dict

class PhaseLoss(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x_phase, y_phase):
        return F.mse_loss(x_phase, y_phase)

class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=  [4096, 2048, 1024, 512, 256],
                 hop_sizes=  [1024, 512,  256,  128, 64],
                 win_lengths=[4096, 2048, 1024, 512, 256],
                 window="hann_window", 
                 factor_sc=1,
                 factor_mag=1,
                 factor_phase=0.1):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.factor_phase = factor_phase

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        phase_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l, phase_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
            phase_loss += phase_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)
        phase_loss /= len(self.stft_losses)

        return self.factor_sc*sc_loss, self.factor_mag*mag_loss, self.factor_phase*phase_loss

def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
    real = x_stft.real
    imag = x_stft.imag

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    mag = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)
    phase = torch.angle(x_stft)
    return mag, phase

class SpectralConvergenceLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")

class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))

class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length, device='cuda:0')
        self.spectral_convergence_loss = SpectralConvergenceLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()
        self.phase_loss = PhaseLoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag, x_phase = stft(x, fft_size=self.fft_size, hop_size = self.shift_size, win_length = self.win_length, window = self.window)
        y_mag, y_phase = stft(y, fft_size=self.fft_size, hop_size = self.shift_size, win_length = self.win_length, window = self.window)
        sc_loss = self.spectral_convergence_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)
        phase_loss = self.phase_loss(x_phase, y_phase)
        return sc_loss, mag_loss, phase_loss
