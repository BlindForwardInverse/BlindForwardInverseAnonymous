import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class STFT(nn.Module):
    """A simple wrapper for torch built-in STFT."""

    def __init__(self, win_len, hop_len, fft_len, win_type):
        super(STFT, self).__init__()
        self.win, self.hop = win_len, hop_len
        self.nfft = fft_len
        window = {
            "hann": torch.hann_window(win_len),
            "hamm": torch.hamming_window(win_len),
        }
        assert win_type in window.keys()
        self.window = window[win_type]

    def transform(self, input, pad_input=False):
        if pad_input:
            input = F.pad(input, (0, STFT.calculate_padding(len(input), self.hop)))
        cspec = torch.stft(
            input,  # [B, N]
            self.nfft,
            self.hop,
            self.win,
            self.window.to(input.device),
            return_complex=True,  # False
        )
        cspec = torch.view_as_real(cspec)  # Convert the complex tensor to a real tensor
        cspec = einops.rearrange(cspec, "b f t c -> b c f t")
        return cspec  # [B, C(R, I), F, T]

    def inverse(self, real, imag, org_len, unpad_output=False):
        # real, imag: [B, F, T]
        # inp = torch.stack([real, imag], dim=-1)
        complex_tensor = torch.view_as_complex(torch.stack([real, imag], dim=-1))

        output = torch.istft(
            complex_tensor, self.nfft, self.hop, self.win, self.window.to(real.device)
        )
        if unpad_output:
            return output[:, :org_len]
        else:
            return output

    @staticmethod
    def calculate_padding(input_len, hop_len):
        pad = (input_len // hop_len + 1) * hop_len - input_len
        return pad

def test_istft(input: torch.Tensor):
    sr = 48000
    signal_length = 132300
    win_len = 32 * 48
    hop_len = 8 * 48

    stft = STFT(win_len, hop_len, win_len, win_type="hann")
    print(input.shape)  # [8, 132300]

    padding = stft.calculate_padding(signal_length, hop_len)  # 180
    input_padded = F.pad(input, (0, padding))

    temp = stft.transform(input_padded)
    print(temp.shape)  # [8, 2, 769, 345] -> [8, 2, 769, 346]

    real = temp[:, 0, :, :]  # [8, 769, 345] -> [8, 769, 346]
    print(real.shape)
    imag = temp[:, 1, :, :]  # [8, 769, 345] -> [8, 769, 346]
    print(imag.shape)

    output_padded = stft.inverse(real, imag, signal_length)  # [8, 132480]
    output = output_padded[:, :signal_length]
    print(output.shape)  # [8, 132300]

    return output


if __name__ == "__main__":
    input = torch.randn(8, 132300)
    output = test_istft(input)
