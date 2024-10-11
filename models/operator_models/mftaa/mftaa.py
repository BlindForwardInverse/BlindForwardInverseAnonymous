import warnings
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .asa import AxialSoftAttention
from .erb import Banks
from .f_sampling import FreqDownsampling, FreqUpsampling
from .phase_encoder import PhaseEncoder
from .stft import STFT
from .tfcm import TFConvModule

eps = 1e-10


class MTFAA(pl.LightningModule):
    def __init__(
        self,
        afx=None,
        n_sig=1,
        PEc=4,
        Co=[48, 96, 192],
        O=[1, 1, 1],
        causal=True,
        bottleneck_layer=2,
        tfcm_layer=6,
        mag_f_dim=3,
        win_len=32 * 48,  # nfft?
        win_hop=8 * 48,
        nerb=256,  # bank nfilters?
        target_sr=48000,
        win_type="hann",
        beta1=0.5,
        beta2=0.999,
        lr=2.25e-05,
    ):
        super(MTFAA, self).__init__()
        self.target_sr = target_sr
        self.beta1 = beta1
        self.beta2 = beta2
        self.lr = lr

        self.PE = PhaseEncoder(PEc, n_sig)
        # 32ms @ 48kHz
        self.stft = STFT(win_len, win_hop, win_len, win_type)
        self.ERB = Banks(nerb, win_len, self.target_sr)
        self.encoder_fd = nn.ModuleList()
        self.encoder_bn = nn.ModuleList()
        self.bottleneck = nn.ModuleList()
        self.decoder_fu = nn.ModuleList()
        self.decoder_bn = nn.ModuleList()
        C_en = [PEc // 2 * n_sig] + Co  # [2, 48, 96, 192]
        C_de = [4] + Co  # [4, 48, 96, 192]

        # Encoder
        for idx in range(len(C_en) - 1):
            self.encoder_fd.append(
                FreqDownsampling(C_en[idx], C_en[idx + 1]),
            )
            self.encoder_bn.append(
                nn.Sequential(
                    TFConvModule(
                        C_en[idx + 1], (3, 3), tfcm_layer=tfcm_layer, causal=causal
                    ),
                    AxialSoftAttention(C_en[idx + 1], causal=causal),
                )
            )

        # Bottleneck
        for idx in range(bottleneck_layer):
            self.bottleneck.append(
                nn.Sequential(
                    TFConvModule(
                        C_en[-1], (3, 3), tfcm_layer=tfcm_layer, causal=causal
                    ),
                    AxialSoftAttention(C_en[-1], causal=causal),
                )
            )

        # Decoder
        for idx in range(len(C_de) - 1, 0, -1):
            self.decoder_fu.append(
                FreqUpsampling(C_de[idx], C_de[idx - 1], O=(O[idx - 1], 0)),
            )
            self.decoder_bn.append(
                nn.Sequential(
                    TFConvModule(
                        C_de[idx - 1], (3, 3), tfcm_layer=tfcm_layer, causal=causal
                    ),
                    AxialSoftAttention(C_de[idx - 1], causal=causal),
                )
            )
        # MEA is causal, so mag_t_dim = 1.
        self.mag_mask = nn.Conv2d(4, mag_f_dim, kernel_size=(3, 1), padding=(1, 0))
        self.real_mask = nn.Conv2d(4, 1, kernel_size=(3, 1), padding=(1, 0))
        self.imag_mask = nn.Conv2d(4, 1, kernel_size=(3, 1), padding=(1, 0))
        kernel = torch.eye(mag_f_dim)
        kernel = kernel.reshape(mag_f_dim, 1, mag_f_dim, 1)
        self.register_buffer("kernel", kernel)
        self.mag_f_dim = mag_f_dim

        self.afx = afx

    def forward(self, sigs):
        # sigs: list [B N] of len(sigs)
        cspecs = []
        for sig in sigs:
            org_len = len(sig[0])
            stfted_sig = self.stft.transform(sig, pad_input=True)
            cspecs.append(stfted_sig)
        # D / E ?
        D_cspec = cspecs[0]
        mag = torch.norm(D_cspec, dim=1)
        phase = torch.atan2(D_cspec[:, -1, ...], D_cspec[:, 0, ...])
        out = self.ERB.amp2bank(self.PE(cspecs))
        encoder_out = []
        for idx in range(len(self.encoder_fd)):
            out = self.encoder_fd[idx](out)
            encoder_out.append(out)
            out = self.encoder_bn[idx](out)

        for idx in range(len(self.bottleneck)):
            out = self.bottleneck[idx](out)

        for idx in range(len(self.decoder_fu)):
            out = self.decoder_fu[idx](out, encoder_out[-1 - idx])
            out = self.decoder_bn[idx](out)
        out = self.ERB.bank2amp(out)

        # stage 1
        mag_mask = self.mag_mask(out)
        mag_pad = F.pad(
            mag[:, None], [0, 0, (self.mag_f_dim - 1) // 2, (self.mag_f_dim - 1) // 2]
        )
        mag = F.conv2d(mag_pad, self.kernel)
        mag = mag * mag_mask.sigmoid()
        mag = mag.sum(dim=1)
        # stage 2
        real_mask = self.real_mask(out).squeeze(1)
        imag_mask = self.imag_mask(out).squeeze(1)

        mag_mask = torch.sqrt(torch.clamp(real_mask**2 + imag_mask**2, eps))
        pha_mask = torch.atan2(imag_mask + eps, real_mask + eps)
        real = mag * mag_mask.tanh() * torch.cos(phase + pha_mask)
        imag = mag * mag_mask.tanh() * torch.sin(phase + pha_mask)
        return (
            mag,
            torch.stack([real, imag], dim=1),
            self.stft.inverse(real, imag, org_len, unpad_output=True),
        )

    def configure_optimizers(self):
        # TODO fix
        lr = self.lr
        params = (
            list(self.encoder_fd.parameters())
            + list(self.encoder_bn.parameters())
            + list(self.bottleneck.parameters())
            + list(self.decoder_fu.parameters())
            + list(self.decoder_bn.parameters())
        )

        opt = torch.optim.Adam(
            params=params, lr=lr, eps=1e-08, betas=(self.beta1, self.beta2)
        )
        return opt

    def _step(self, batch):
        dry, wet, dry_ref, wet_ref, afx = [
            batch[key] for key in ["dry", "wet", "dry_ref", "wet_ref", "afx"]
        ]
        afx_info = batch["afx_info"] if "afx_info" in batch else None

        pred_mag, pred_cspec, pred = self([dry])

        # Loss
        mse_loss = F.mse_loss(wet, pred)
        self.log("mse_loss", mse_loss, prog_bar=True, on_epoch=True)
        return mse_loss, dry, wet, pred_mag, pred_cspec, pred, afx, afx_info

    def training_step(self, batch, batch_idx):
        mse_loss = self._step(batch)[0]
        self.log("train_loss", mse_loss, on_epoch=True, prog_bar=True)
        return mse_loss

    def validation_step(self, batch, batch_idx):
        mse_loss, dry, wet, pred_mag, pred_cspec, pred, afx, afx_info = self._step(
            batch
        )
        valid_loss = mse_loss
        self.log("valid_loss", valid_loss, on_step=False, on_epoch=True, prog_bar=True)
        return valid_loss, dry, wet, pred_mag, pred_cspec, pred, afx, afx_info


def test_nnet():
    # noise supression (microphone, )
    model = MTFAA(n_sig=1)
    input = torch.randn(3, 48000)
    mag, cspec, wav = model([input])
    # torch.Size([3, 769, 126]) torch.Size([3, 2, 769, 126]) torch.Size([3, 48000])
    print(mag.shape, cspec.shape, wav.shape)

    # echo cancellation (microphone, error, reference,)
    model = MTFAA(n_sig=3)
    mag, cspec, wav = model([input, input, input])
    print(mag.shape, cspec.shape, wav.shape)


def test_mac():
    from thop import clever_format, profile

    model = MTFAA(n_sig=3)
    # hop=8ms, win=32ms@48KHz, process 1s.
    input = torch.randn(1, 48000)
    # inp = th.randn(1, 2, 769, 126)
    macs, params = profile(model, inputs=([input, input, input],), verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print("macs: ", macs)
    print("params: ", params)


if __name__ == "__main__":
    test_nnet()
    # test_mac()
