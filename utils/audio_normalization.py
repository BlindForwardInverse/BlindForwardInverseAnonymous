import torch
import numpy as np

def rms_normalize(wav, ref_dBFS=-16.0, return_gain=False):
    eps = np.finfo(np.float32).eps
    ref_linear = np.power(10, (ref_dBFS-3.0103)/20.)
    if isinstance(wav, torch.Tensor):
        device = wav.device
        batch_size = wav.shape[0]
        rms = torch.sqrt(torch.mean(torch.square(wav), dim=-1) + eps)
    else:
        rms = np.sqrt(np.mean(np.square(wav), axis=-1) + eps)

    gain = ref_linear / (rms + 1e-7)
    wav = gain * wav
    if return_gain:
        return wav, gain
    else:
        return wav

def absolute_normalize(wav):
    max_amp = torch.max(torch.abs(wav))
    wav = wav / max_amp
    return wav

def peak_normalize(wav):
    max_amp = torch.max(torch.abs(wav))
    wav = wav / max_amp
    return wav
