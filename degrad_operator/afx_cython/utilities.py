"""
utility modules
"""

import jax
import jax.numpy as jnp
from scipy import signal

from functools import partial
from afx.primitives import get_signal, gain_stage
import numpy as np

# @partial(jax.jit, backend='cpu')
def lr_to_ms(x):
    #assert x.ndim == 2 and x.shape[-1] == 2
    if x.ndim == 2: 
        if x.shape[-1] == 2:
            l, r = x[:, :1], x[:, 1:]
            m, s = l + r, l - r
        else:
            m, s = x*2, jnp.zeros_like(x)
        return m, s
    else:
        assert False

# @partial(jax.jit, backend='cpu')
def ms_to_lr(m, s):
    assert m.ndim == 2 and s.ndim == 2
    if m.shape[-1] == 2:
        m = jnp.sum(m, -1, keepdims=True)
    if s.shape[-1] == 2:
        s = jnp.sum(s, -1, keepdims=True)
    l, r = (m + s) / 2, (m - s) / 2
    x = jnp.concatenate([l, r], -1)
    return x

# @partial(jax.jit, backend='cpu')
def panning_mono(x, pan):
    return jnp.concatenate([x * (1 - pan), x * (1 + pan)], -1)


# @partial(jax.jit, backend='cpu')
def panning_stereo(x, pan):
    return jnp.concatenate([x[:, :1] * (1 - pan), x[:, 1:] * (1 + pan)], -1)

def panning(x, pan=0):
    if x.shape[-1] == 1:
        return panning_mono(x, pan=pan)
    if x.shape[-1] == 2:
        return panning_stereo(x, pan=pan)

@partial(jax.jit, backend="cpu")
def imager(x, width=0):
    m, s = lr_to_ms(x)
    m, s = m * (1 - width), s * (1 + width)
    return ms_to_lr(m, s)

def mix(x):
    return x

def convolve(x, h):
    # assert x.shape[-1] == 1 and h.shape[-1] == 1
    x, h = np.array(x[:, 0]), np.array(h[:, 0])
    out = signal.convolve(x, h)[:len(x), None]
    return out

def apply_utility(input_signal, afx_type, gain_staging, **param_dict):
    x = get_signal(input_signal, "main", c=2)

    if afx_type == "panning":
        if "pan" in input_signal:
            if input_signal["pan"].shape[-1] == 2:
                pan = input_signal["pan"][:, :1] + param_dict["pan"]
            elif input_signal["pan"].shape[-1] == 1:
                pan = input_signal["pan"] + param_dict["pan"]
            else:
                print(input_signal["modulation"].shape)
                assert False
        else:
            pan = param_dict["pan"]
        y = panning(x, pan)
        if gain_staging: y = gain_stage(x, y)
        return {"main": y}
    elif afx_type == "imager":
        y = imager(x, **param_dict)
        if gain_staging: y = gain_stage(x, y)
        return {"main": y}
    elif afx_type == "midside":
        mid, side = lr_to_ms(x)
        return {"mid": mid, "side": side}
    elif afx_type == "stereo":
        mid, side = get_signal(input_signal, "mid"), get_signal(input_signal, "side")
        return {"main": ms_to_lr(mid, side)}
    elif afx_type == "mix":
        return {"main": mix(x)}
    elif afx_type == "convolution":
        h = get_signal(input_signal, "ir", c=1)
        y = convolve(x, h)
        if gain_staging: y = gain_stage(x, y)
        return {"main": y}
    elif afx_type == "adder":
        h = get_signal(input_signal, "noise", c=1)
        y = adder(x, h)
        if gain_staging: y = gain_stage(x, y)
        return {"main": y}
        
    else:
        assert False
