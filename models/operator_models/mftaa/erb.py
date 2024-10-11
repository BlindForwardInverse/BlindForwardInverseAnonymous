import torch
import torch.nn as nn
import torch.nn.functional as F
from spafe.fbanks import linear_fbanks


class Banks(nn.Module):
    """
    linear FBank instead of ERB scale.
    NOTE To to reduce the reconstruction error, the linear fbank is used.
    """

    def __init__(self, nfilters, nfft, fs, low_freq=0, high_freq=None, learnable=False):
        super(Banks, self).__init__()
        self.nfilters, self.nfft, self.fs = nfilters, nfft, fs
        filter, _ = linear_fbanks.linear_filter_banks(
            nfilts=self.nfilters,
            nfft=self.nfft,
            low_freq=low_freq,  #
            high_freq=high_freq,
            fs=self.fs,
        )
        filter = torch.from_numpy(filter).float()
        if not learnable:
            #  30% energy compensation.
            self.register_buffer("filter", filter * 1.3)
            self.register_buffer("filter_inv", torch.pinverse(filter))
        else:
            self.filter = nn.Parameter(filter)
            self.filter_inv = nn.Parameter(torch.pinverse(filter))

    def amp2bank(self, amp):
        amp_feature = torch.einsum("bcft,kf->bckt", amp, self.filter)
        return amp_feature

    def bank2amp(self, inputs):
        return torch.einsum("bckt,fk->bcft", inputs, self.filter_inv)


def test_bank():
    import numpy as np
    import soundfile as sf

    from stft import STFT

    stft = STFT(32 * 48, 8 * 48, 32 * 48, "hann")
    net = Banks(256, 32 * 48, 48000)
    sig_raw, sr = sf.read("path/to/48k.wav")
    sig = torch.from_numpy(sig_raw)[None, :].float()
    cspec = stft.transform(sig)
    mag = torch.norm(cspec, dim=1)
    phase = torch.atan2(cspec[:, 1, :, :], cspec[:, 0, :, :])
    mag = mag.unsqueeze(dim=1)
    outs = net.amp2bank(mag)
    outs = net.bank2amp(outs)
    print(F.mse_loss(outs, mag))
    outs = outs.squeeze(dim=1)
    real = outs * torch.cos(phase)
    imag = outs * torch.sin(phase)
    sig_rec = stft.inverse(real, imag)
    sig_rec = sig_rec.cpu().data.numpy()[0]
    min_len = min(len(sig_rec), len(sig_raw))
    sf.write("res.wav", np.stack([sig_rec[:min_len], sig_raw[:min_len]], axis=1), sr)
    print(np.mean(np.square(sig_rec[:min_len] - sig_raw[:min_len])))


if __name__ == "__main__":
    test_bank()
