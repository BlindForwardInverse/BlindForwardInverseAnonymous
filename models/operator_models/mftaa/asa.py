import einops
import torch
import torch.nn as nn


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


class AxialSoftAttention(nn.Module):
    """
    Axial Soft Attention (ASA).
    NOTE I recommend that you remove the t-attention and only keep
    the f-attention when using it, because there is already TFCMs
    to time-modeling, and doing so can greatly increase the batch size.
    """

    def __init__(self, c=64, causal=True, with_t_attn=True):
        super(AxialSoftAttention, self).__init__()
        self.d_c = c // 4
        self.with_t_attn = with_t_attn
        if not with_t_attn:
            print("Will Not perform attention w.r.t. t axis")

        self.f_qkv = nn.Sequential(
            nn.Conv2d(c, self.d_c * 3, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(self.d_c * 3),
            nn.PReLU(self.d_c * 3),
        )
        if with_t_attn:
            self.t_qk = nn.Sequential(
                nn.Conv2d(c, self.d_c * 2, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(self.d_c * 2),
                nn.PReLU(self.d_c * 2),
            )
        self.proj = nn.Sequential(
            nn.Conv2d(self.d_c, c, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.PReLU(c),
        )
        self.causal = causal

    def forward(self, input):
        """
        inp: B C F T
        """
        # f-attention
        f_qkv = self.f_qkv(input)
        qf, kf, v = tuple(einops.rearrange(f_qkv, "b (c k) f t->k b c f t", k=3))
        f_score = torch.einsum("bcft,bcyt->btfy", qf, kf) / (self.d_c**0.5)
        f_score = f_score.softmax(dim=-1)
        f_out = torch.einsum("btfy,bcyt->bcft", [f_score, v])
        # t-attention
        if self.with_t_attn:
            t_qk = self.t_qk(input)
            qt, kt = tuple(einops.rearrange(t_qk, "b (c k) f t->k b c f t", k=2))
            t_score = torch.einsum("bcft,bcfy->bfty", [qt, kt]) / (self.d_c**0.5)
            mask_value = max_neg_value(t_score)
            if self.causal:
                i, j = t_score.shape[-2:]
                mask = torch.ones(i, j, device=t_score.device).triu_(j - i + 1).bool()
                t_score.masked_fill_(mask, mask_value)
            t_score = t_score.softmax(dim=-1)
            t_out = torch.einsum("bfty,bcfy->bcft", [t_score, f_out])
            out = self.proj(t_out)
        else:
            out = self.proj(f_out)
        return out + input


def test_asa():
    block = AxialSoftAttention(c=64)
    input = torch.randn(2, 64, 256, 100)
    out = block(input)
    print("out: ", out.shape)  # torch.Size([2, 64, 256, 100])


if __name__ == "__main__":
    test_asa()
