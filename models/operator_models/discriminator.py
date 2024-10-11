import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

class DiscriminatorWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.MRD = MultiResolutionDiscriminator(
                resolutions = [2048, 1024, 512],
                hop_lengths = [240, 120, 50],
                win_lengths = [1200, 600, 240],
                )
        self.MPD = MultiPeriodDiscriminator(
                mpd_reshapes = [2,  5 , 11]
                )
    def forward(self, wav):
        mrd_logits = self.MRD(wav)
        mpd_logits = self.MPD(wav)
        return mrd_logits, mpd_logits

class MultiResolutionDiscriminator(nn.Module):
    def __init__(self,
                 resolutions = [2048, 1024, 512],
                 hop_lengths = [240, 120, 50],
                 win_lengths = [1200, 600, 240],
                 ):
        super().__init__()
        self.resolutions = resolutions
        self.discs = nn.ModuleList(
            [SpectrogramDiscriminator(res) for res in self.resolutions]
            )
    
    def forward(self, wav):
        logits = []
        for disc, res in zip(self.discs, self.resolutions):
            mag = self.stft(wav, fft_size=res, hop_size=res // 4, win_length=res)
            logit = disc(mag)
            logits.append(logit)
        return logits

    @staticmethod
    def stft(x, fft_size, hop_size, win_length):
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
        window = torch.hann_window(win_length, device=x.device)
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True)
        real = x_stft.real
        imag = x_stft.imag
        mag = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7))
        return mag

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, mpd_reshapes=[2, 3, 5, 7, 11]):
        super(MultiPeriodDiscriminator, self).__init__()
        self.mpd_reshapes = mpd_reshapes
        print("mpd_reshapes: {}".format(self.mpd_reshapes))
        self.discriminators = nn.ModuleList(
                [DiscriminatorP(d_mult=1,
                                period=rs,
                                use_spectral_norm=False)
                for rs in self.mpd_reshapes]
                )

    def forward(self, wav):
        logits = []
        wav = rearrange(wav, 'b t -> b 1 t')
        for i, disc in enumerate(self.discriminators):
            logit = disc(wav)
            logits.append(logit)
        return logits

class SpectrogramDiscriminator(nn.Module):
    def __init__(self,
                 resolutions,
                 d_mult=2,
                 use_spectral_norm=False,
                 ):
        super().__init__()
        self.d_mult = d_mult
        c = 32 * d_mult
        self.lrelu_slope=0.1

        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, c, (3, 9), padding=(1, 4))),
            norm_f(nn.Conv2d(c, c, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(c, c, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(c, c, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm_f(nn.Conv2d(c, c, (3, 9), stride=(1, 2), padding=(1, 4))),
        ])
        self.conv_post = norm_f(nn.Conv2d(c, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        """
        Input : Magnitude spectrogram (B x F x T)
        """
        x = rearrange(x, 'b f t -> b 1 f t')
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, self.lrelu_slope)
        x = self.conv_post(x)
        x = rearrange(x, 'b 1 f t -> b (f t)') # Flatten
        return x


class DiscriminatorP(torch.nn.Module):
    def __init__(self, d_mult, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.d_mult = d_mult
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, int(32*self.d_mult), (kernel_size, 1), (stride, 1), padding=(self.get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(int(32*self.d_mult), int(128*self.d_mult), (kernel_size, 1), (stride, 1), padding=(self.get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(int(128*self.d_mult), int(512*self.d_mult), (kernel_size, 1), (stride, 1), padding=(self.get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(int(512*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), (stride, 1), padding=(self.get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(int(1024*self.d_mult), int(1024*self.d_mult), (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(int(1024*self.d_mult), 1, (3, 1), 1, padding=(1, 0)))

    @staticmethod
    def get_padding(kernel_size, dilation=1):
        return int((kernel_size*dilation - dilation)/2)

    def forward(self, x):
        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.2)
        x = self.conv_post(x)
        x = torch.flatten(x, 1, -1)
        return x

class PatchGANDiscriminator(nn.Module):
    def __init__(self,
                 in_channels=2,
                 num_filters_last=64,
                 n_layers=3):
        super().__init__()

        layers = [nn.Conv2d(in_channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                          2 if i < n_layers else 1, 1, bias=False),
                nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]

        layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = rearrange(x, 'b f t c -> b c f t')
        x = self.model(x)
        x = rearrange(x, 'b 1 f t -> b (f t)')
        logit = [x,]
        return logit
