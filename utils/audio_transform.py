import numpy as np
import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram, Spectrogram


def audio_transform(audio, transform_type="power_ri", 
                    n_fft=2046, win_length=2046, hop_length=512, **kwargs):
    if transform_type == "power_ri":
        spec = power_spectrogram(audio, return_mag=False, n_fft=n_fft, win_length=win_length, hop_length=hop_length,**kwargs)
        # (B, F, T, C) = (B, 256, 512, 2)
    elif transform_type == "power_rim":
        spec = power_spectrogram(audio, return_mag=True, **kwargs)
    elif transform_type == "power_mp":
        spec = power_mag_phase(audio, **kwargs)
    elif transform_type == "stft":
        spec = stft(audio)
    elif transform_type == "cqt":
        pass
    elif transform_type == "waveform":
        return audio
    else:
        assert False
    return spec

def spec_transform(spec, transform_type="unpower_ri",
                   n_fft=2046, win_length=2046, hop_length=512, length=None):
    if transform_type == "unpower_ri":
        audio = unpower_spectrogram(
                spec, 
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                length=length,
                with_mag=False)
    elif transform_type == 'unpower_mp':
        audio = unpower_mag_phase(spec, length)
    return audio

def stft(
    audio,
    n_fft=2048,
    win_length=2048,
    hop_length=512,
    return_type = 'mag_phase'
):
    if isinstance(audio, torch.Tensor):
        device = audio.device
    else:
        audio = torch.Tensor(audio)
        device = "cpu"
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length, device=device),
        return_complex=True,
    )
    mag = spec.abs() 
    phase = spec.angle()
    if return_type == 'mag_phase':
        spec = torch.stack((mag, phase), dim=-1)
    elif return_type == 'real_imag':
        real = scaled_spec.real
        imag = scaled_spec.imag
        spec = torch.stack((real, imag), dim=-1)
    elif return_type == 'mag_only':
        real, imag = spec.real, spec.imag
        spec = torch.sqrt(torch.clamp(real ** 2 + imag **2, min=1e-7))
    return spec

def power_spectrogram(
    audio,
    alpha=0.5,
    beta=0.15,
    n_fft=510,
    win_length=510,
    hop_length=128,
    return_mag=False,
):
    if torch.is_tensor(audio):
        device = audio.device
    else:
        audio = torch.tensor(audio)
        device = "cpu"
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length, device=device),
        return_complex=True,
    )
    mag = torch.pow(spec.abs(), alpha)
    phase = spec.angle()
    scaled_spec = torch.polar(mag, phase) * beta

    real = scaled_spec.real
    imag = scaled_spec.imag
    if return_mag:
        spec = torch.stack((real, imag, mag), dim=-1)
    else:
        spec = torch.stack((real, imag), dim=-1)
    return spec


def unpower_spectrogram(
    spec,
    length=None,
    alpha=0.5,
    beta=0.15,
    n_fft=2046,
    win_length=2046,
    hop_length=512,
    with_mag=False,
):
    if isinstance(spec, torch.Tensor):
        device = spec.device
    else:
        device = "cpu"
    if with_mag:
        complex_spec = torch.view_as_complex(spec[..., :2].contiguous())
    else:
        complex_spec = torch.view_as_complex(spec.contiguous())
    mag = torch.pow(complex_spec.abs(), 1.0 / alpha)
    phase = complex_spec.angle()
    rescaled_spec = torch.polar(mag, phase) / beta ** (1.0 / alpha)

    if length:
        audio = torch.istft(
            rescaled_spec,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            length=length,
            window=torch.hann_window(win_length, device=rescaled_spec.device),
        )
    else:
        audio = torch.istft(
            rescaled_spec,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=torch.hann_window(win_length, device=rescaled_spec.device),
        )
    return audio

def power_mag_phase(
    audio,
    alpha=0.5,
    beta=0.15,
    n_fft=510,
    win_length=510,
    hop_length=128,
):
    if torch.is_tensor(audio):
        device = audio.device
    else:
        audio = torch.tensor(audio)
        device = "cpu"
    spec = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=torch.hann_window(win_length, device=device),
        return_complex=True,
    )
    mag = torch.pow(spec.abs(), alpha)
    phase = spec.angle()
    scaled_spec = torch.polar(mag, phase) * beta

    mag = scaled_spec.abs()
    phase = scaled_spec.angle()
    spec = torch.stack((mag, phase), dim=-1)
    return spec

def unpower_mag_phase(
    spec,
    length=None,
    alpha=0.5,
    beta=0.15,
    n_fft=510,
    win_length=510,
    hop_length=128,
    with_mag=False,
):
    if isinstance(spec, torch.Tensor):
        device = spec.device
    else:
        device = "cpu"
    if with_mag:
        complex_spec = torch.view_as_complex(spec[..., :2].contiguous())
    else:
        complex_spec = torch.view_as_complex(spec.contiguous())

    mag = spec[..., 0]
    phase = spec[...,1]
    rescaled_spec = torch.polar(mag, phase) / beta ** (1.0 / alpha)

    if length:
        audio = torch.istft(
            rescaled_spec,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            length=length,
            window=torch.hann_window(win_length, device=rescaled_spec.device),
        )
    else:
        audio = torch.istft(
            rescaled_spec,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=torch.hann_window(win_length, device=rescaled_spec.device),
        )
    return audio

def audio_to_spec(
    audio,
    n_fft=2046,
    win_length=2046,
    hop_length=512,
    power=None,
    return_mag=True,
):
    """
    Set power = 2 to get power spectrogram,
    power = None to get complex spectrogram
    """
    if isinstance(audio, torch.Tensor):
        device = audio.device
    else:
        device = "cpu"
    spec = Spectrogram(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=power
    ).to(device)
    output = spec(audio)

    if power is None:
        real = output.real
        imag = output.imag
        mag = output.abs()
        if isinstance(audio, torch.Tensor):
            if return_mag:
                complex_spec = torch.stack((real, imag, mag), dim=-1)
            else:
                complex_spec = torch.stack((real, imag), dim=-1)
        return complex_spec
    return output

if __name__ == "__main__":
    # test_spec()
    spec = torch.rand(1024, 192, 2)
    audio = spec_transform(spec)
    print(audio.shape)

#     import soundfile as sf

#     wav, sr = sf.read("operator_learning/models/encodec/test_24k.wav")  # error at 48k?
#     print(power_spectrogram(wav).shape)
