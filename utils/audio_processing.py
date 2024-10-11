import math
import random

import librosa
import numpy as np
import torch
import torchaudio.transforms as T
from einops import rearrange, repeat
from omegaconf import OmegaConf

def audio_processing(
    audio,
    target_len,
    sr=44100,
    target_sr=44100,
    mono=True,
    crop_mode="random",
    pad_mode="back",
    pad_type="zero",
    rms_norm=False,
):
    # Audio Preprocessing : Monorize, Resample, Padding or Crop, Normalize
    if mono:
        audio = make_it_mono(audio)
    else:
        audio = make_it_stereo(audio)

    if sr > target_sr:
        audio = resample(audio, sr, target_sr)
    elif sr < target_sr:
        print(f"Warning : Tried to upsample from {sr} to {target_sr}")

    audio_len = audio.shape[0]
    if audio_len < target_len:
        if pad_mode != "none":
            audio = pad_audio(audio, target_len, pad_type=pad_type, pad_mode=pad_mode)
    else:
        audio = crop_audio(audio, target_len, crop_mode=crop_mode)

    if rms_norm:
        audio = rms_normalize_numpy(audio)
    return audio

def ir_preprocessing(
    ir,
    sr=44100,
    target_sr=44100,
    max_ir_sec=5.,
    mono=True,
    rms_norm=True,
    onset_matching=True,
):
    # ir : (T,) or (T x C) shape
    # Mono / Stereo
    if mono: ir = make_it_mono(ir)
    else: ir = make_it_stereo(ir)

    # Resample
    if sr != target_sr:
        ir = resample(ir, sr, target_sr)

    # Onset_matching
    if onset_matching:
        onset = np.argmax(np.abs(ir))
        ir = ir[onset:]

    # Crop
    ir_len = ir.shape[0]
    max_ir_len = int(max_ir_sec * target_sr)
    if ir_len > max_ir_len:
        ir = crop_audio(ir, max_ir_len, crop_mode='front')

    # Rms_norm
    if rms_norm:
        ir = rms_normalize_numpy(ir)
    return ir

def resample(audio, sr=16000, target_sr=44100):
    # warning!  audio : (T,) or (T x C) shape
    return librosa.resample(audio, orig_sr=sr, target_sr=target_sr, axis=0)

def make_it_mono(audio, average_all_channels=False):
    if audio.ndim > 1:
        if audio.shape[-1] == 2 or average_all_channels:
            audio = np.mean(audio, axis=-1)
        else: # Process only top two channels
            audio = np.mean(audio[:, :2], axis=-1)
    return audio

def make_it_stereo(audio, average_all_channels=False):
    if audio.ndim == 1: # Mono
        audio = repeat(audio, "l -> l c", c=2)
    elif audio.ndim == 2 and audio.shape[-1] == 2: # Already stereo
        audio = audio
    elif audio.ndim == 2 and audio.shape[-1] > 2:  # Multi-channel audio
        if average_all_channels:
            # Mix down all channels to stereo
            left_channel = np.mean(audio[:, ::2], axis=1)  # Averages of odd-indexed channels
            if audio.shape[1] % 2 == 0:  # Even number of channels
                right_channel = np.mean(audio[:, 1::2], axis=1)  # Averages of even-indexed channels
            else:  # Odd number of channels, include the last channel in both left and right
                right_channel = np.mean(np.hstack([audio[:, 1::2], audio[:, -1][:, np.newaxis]]), axis=1)
            audio = np.stack([left_channel, right_channel], axis=1)
        else:
            # Process only top two channels
            audio = audio[:, :2]
    else:
        raise ValueError("Input audio must be either mono, stereo, or multi-channel")
    return audio

def pad_audio(wav, len_target, pad_type="zero", pad_mode="back"):
    # Now enables for mono/stereo/multi-channel audio (T,), (T, 2), (T, n)
    len_wav = wav.shape[0]
    if len(wav.shape) == 2:
        num_channels = wav.shape[1]

    len_pad = len_target - len_wav
    if pad_type == "zero":
        if pad_mode == "random":
            st = random.randint(0, len_pad)
            ed = st + len_wav
        elif pad_mode == "front":
            st = len_pad
            ed = len_target
        elif pad_mode == "back":
            st = 0
            ed = len_wav
        else:
            raise Exception("Not supported padding type {}".format(pad_mode))

        if len(wav.shape) == 2:
            padded = np.zeros((len_target, num_channels), dtype=np.float32)
            padded[st:ed, :] = wav
        else:
            padded = np.zeros(len_target, dtype=np.float32)
            padded[st:ed] = wav

    elif pad_type == "repeat":
        repeat = 1 + int(len_pad // len_wav)
        wav = np.tile(wav, repeat)

        rest_pad = len_target - len_wav * repeat
        if len(wav.shape) == 2:
            padded = np.concatenate((wav, wav[:rest_pad, :]), axis=0)
        else:
            padded = np.concatenate((wav, wav[:rest_pad]))

    else:
        raise Exception("Not supported padding mode {}".format(pad_type))

    return padded


def crop_audio(wav, len_target, crop_mode="random", threshold_db=-8.):
    """
    Crop audio with fixed length
    Now support for mono/stereo/multi-channel audio
    """
    len_wav = wav.shape[0]
    if len(wav.shape) == 2:
        num_channels = wav.shape[1]
    len_crop = len_wav - len_target
    assert len_crop >= 0, f"Audio sample of len {len_wav} is shorter than len_target={len_target}."

    if crop_mode == "random":
        st = random.randint(0, len_crop)
    elif crop_mode == "front":
        st = 0
    elif crop_mode == "back":
        st = len_crop
    elif crop_mode == "adaptive":
        threshold_amp = np.power(10., threshold_db/20., dtype=np.float32)
        for i in range(len_crop):
            if wav[i] > threshold_amp : break
        st = i
    else:
        raise Exception("Not supported cropping mode {}".format(crop_mode))

    if len(wav.shape) == 2:
        cropped = wav[st : st + len_target, :]
    else:
        cropped = wav[st : st + len_target]
    return cropped

def rms_normalize_numpy(wav, ref_dBFS=-16.0, return_gain=False):
    eps = np.finfo(np.float32).eps
    ref_linear = np.power(10, (ref_dBFS - 3.0103) / 20.0, dtype=np.float32)
    if isinstance(wav, torch.Tensor):
        device = wav.device
        batch_size = wav.shape[0]
        rms = torch.sqrt(torch.mean(torch.square(wav), dim=-1) + eps)
    else:
        rms = np.sqrt(np.mean(np.square(wav), axis=-1) + eps)

    gain = ref_linear / (rms + 1e-7)
    
    if len(wav.shape) == 2:
        gain = gain.reshape(-1, 1)
    
    wav = gain * wav
    if return_gain:
        return wav, gain
    else:
        return wav


def wav_to_mel(wav, target_sr, n_fft, win_length, hop_length, n_mels):
    mel_transform = T.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel_spec = mel_transform(wav)
    mel_spec = T.AmplitudeToDB()(mel_spec)  # amplitude to db scale
    mel_spec = rearrange(mel_spec, "1 b w h-> b 1 w h")

    return mel_spec  # [n_mels, channel, t frame]


def power_to_db(power, ref_db=0.0, range_db=80.0, use_torch=False):
    """Converts power from linear scale to decibels."""
    # Choose library
    maximum = np.maximum
    log_base10 = np.log10

    # Convert to decibels
    pmin = 10 ** -(range_db / 10.0)
    power = maximum(pmin, power)
    db = 10.0 * log_base10(power)

    # Set dynamic range
    db -= ref_db
    db = maximum(db, -range_db)
    return db

if __name__ == "__main__":
    audio = np.random.rand(44100*2, 4)
    cropped = crop_audio(audio, 44100)
    print(cropped.shape)
    padded = pad_audio(audio, len_target=44100*4)
    print(padded.shape)
    stereo = make_it_stereo(audio)
    print(stereo.shape)
    mono = make_it_mono(audio)
    print(mono.shape)
