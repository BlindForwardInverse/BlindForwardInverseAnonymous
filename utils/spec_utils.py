import math
import random

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from einops import rearrange, repeat
from omegaconf import OmegaConf

from .audio_processing import power_to_db


def mel_to_wav(mel_spec, target_sr, n_fft, win_length, hop_length, n_mels):
    # Convert Mel spectrogram to linear spectrogram
    mel_spec = mel_spec.cpu().numpy().squeeze()
    mel_to_linear_transform = T.MelScale(
        n_stft=(n_fft // 2) + 1,
        n_mels=n_mels,
        sample_rate=target_sr,
    )
    linear_spec = mel_to_linear_transform(mel_spec)

    # Apply Griffin-Lim algorithm to get the waveform
    griffin_lim_transform = T.GriffinLim(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length
    )
    wav = griffin_lim_transform(linear_spec)

    return wav


def mel_batch_to_wav_batch(mel_batch, target_sr, n_fft, win_length, hop_length):
    if isinstance(mel_batch, np.ndarray):
        mel_batch = torch.from_numpy(mel_batch)

    mel_batch = mel_batch.to("cuda" if torch.cuda.is_available() else "cpu")
    wav_batch = []

    for mel in mel_batch:
        # Assuming mel is a numpy array of shape (1, n_mels, time_frames)
        mel = mel.squeeze(0).cpu().numpy().astype(np.float32)
        waveform = librosa.feature.inverse.mel_to_audio(
            mel,
            sr=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=True,
            pad_mode="reflect",
            power=2.0,
            n_iter=32,
        )
        wav_batch.append(waveform)
    return np.array(wav_batch)


def compute_loudness(
    audio: np.ndarray,
    sample_rate=16000,
    frame_rate=250,  # Rate of loudness frames
    n_fft=512,
    range_db=80.0,  # Dynamic range of loudness dB
    ref_db=0.0,  # Reference maximum perceptual loudness
    padding="valid",  # same, valid, center # TODO change default to center
):
    # TODO make function differentiable?
    lib = np
    reduce_mean = np.mean
    stft_fn = stft_np

    frame_size = n_fft
    hop_size = sample_rate // frame_rate
    audio = pad(audio, frame_size, hop_size, padding=padding)
    audio = np.array(audio)

    # Temporarily a batch dimension for single examples.
    is_1d = len(audio.shape) == 1
    audio = audio[lib.newaxis, :] if is_1d else audio

    # Take STFT.
    overlap = 1 - hop_size / frame_size
    s = stft_fn(audio, frame_size=frame_size, overlap=overlap, pad_end=False)

    # Compute power.
    amplitude = lib.abs(s)
    power = amplitude**2

    # Perceptual weighting.
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    a_weighting = librosa.A_weighting(frequencies)[lib.newaxis, lib.newaxis, :]

    # Perform weighting in linear scale, a_weighting given in decibels.
    weighting = 10 ** (a_weighting / 10)
    power = power * weighting

    # Average over frequencies (weighted power per a bin).
    avg_power = reduce_mean(power, axis=-1)
    loudness = power_to_db(avg_power, ref_db=ref_db, range_db=range_db)

    # Remove temporary batch dimension.
    loudness = loudness[0] if is_1d else loudness

    return loudness  # [B, n_frames] or [n_frames,]


def stft(audio, frame_size=2048, overlap=0.75, pad_end=True):
    """Differentiable stft computed in batch"""
    audio = audio.to(torch.float32)

    if len(audio.shape) == 3:
        audio = torch.squeeze(audio, axis=-1)  # Remove channel dim if present

    # Calculate the frame step
    frame_step = int(frame_size * (1.0 - overlap))

    s = torch.stft(
        audio,
        n_fft=frame_size,
        hop_length=frame_step,
        win_length=frame_size,
        window=torch.hann_window(frame_size, device='cuda'),
        center=True,  # Centers the window at each frame
        pad_mode="reflect",  # Padding
        normalized=False,  # True for normalized STFT
        onesided=True,  # False for a two-sided STFT
        return_complex=True,
    )

    return s


def get_framed_lengths(input_length, frame_size, hop_size, padding="center"):
    """Give a strided framing, gives output lengths."""

    def get_n_frames(length):
        return int(np.floor((length - frame_size) / hop_size)) + 1

    if padding == "valid":
        padded_length = input_length
        n_frames = get_n_frames(input_length)

    elif padding == "center":
        padded_length = input_length + frame_size
        n_frames = get_n_frames(padded_length)

    elif padding == "same":
        n_frames = int(np.ceil(input_length / hop_size))
        padded_length = (n_frames - 1) * hop_size + frame_size

    return n_frames, padded_length


def pad(
    x,
    frame_size,
    hop_size,
    padding="center",  # same, center, valid
    axis=1,
    mode="CONSTANT",
    constant_values=0,
):
    """Pad a tensor for strided framing"""
    x = x.to(torch.float32)

    if padding == "valid":
        return x

    if hop_size > frame_size:
        raise ValueError(
            f"During padding, frame_size ({frame_size})"
            f" must be greater than hop_size ({hop_size})."
        )

    if len(x.shape) <= 1:
        axis = 0

    n_t = x.shape[axis]
    _, n_t_padded = get_framed_lengths(n_t, frame_size, hop_size, padding)
    pads = [(0, 0) for _ in range(len(x.shape))]

    if padding == "same":
        pad_amount = int(n_t_padded - n_t)
        pads[axis] = (0, pad_amount)

    elif padding == "center":  # TODO fix
        pad_amount = int(frame_size // 2)  # Symmetric even padding like librosa
        pads[axis] = (pad_amount, pad_amount)

    else:
        raise ValueError(
            f"padding must be one of ['center', 'same', 'valid'], received ({padding}).",
        )

    return F.pad(x, pads, mode=mode, value=constant_values)


def compute_mag(audio, size=2048, overlap=0.75, pad_end=True):
    mag = torch.abs(stft(audio, frame_size=size, overlap=overlap, pad_end=pad_end))
    return mag.to(torch.float32)


def stft_np(audio, frame_size=2048, overlap=0.75, pad_end=True):
    """Non-differentiable stft using librosa, one example at a time."""
    assert frame_size * overlap % 2.0 == 0.0
    hop_size = int(frame_size * (1.0 - overlap))
    is_2d = len(audio.shape) == 2

    if pad_end:
        audio = pad(audio, frame_size, hop_size, "same", axis=is_2d).numpy()

    def stft_fn(y):
        return librosa.stft(
            y=y, n_fft=int(frame_size), hop_length=hop_size, center=False
        ).T

    s = np.stack([stft_fn(a) for a in audio]) if is_2d else stft_fn(audio)
    return s


if __name__ == "__main__":
    data_config = OmegaConf.load("../configs/input_signal/speech.yaml")

    x_mel = np.random.rand(4, 1, 128, 128)
    y_mel = np.random.rand(4, 1, 128, 128)
    y_pred_mel = np.random.rand(4, 1, 128, 128)

    x, y, y_pred = [
        mel_batch_to_wav_batch(mel, 44100, 1024, 1024, 512) for mel in [x_mel, y_mel, y_pred_mel]
    ]
    print(x.shape)
