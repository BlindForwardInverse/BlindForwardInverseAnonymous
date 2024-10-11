""" 
modulation effects
currently supporting: 
    * phaser
    * chorus
"""
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from functools import partial
from .state_variable_filters import tvsvfilt_step, apply_2nd_order_filter
from .jafx_utils import get_signal, gain_stage, hz_to_G, q_to_twoR
from jax import random

import numpy as np

def apply_modulation_effect(input_signal, afx_type, gain_staging, sr, signal_len, mono, **param_dict):
    x = get_signal(input_signal, "main")
    m = get_signal(input_signal, "modulation")

    if afx_type == "phaser":
        y = phaser(x, m, sr=sr, **param_dict)
    elif afx_type == "chorus":
        y = chorus_const(x, m, sr=sr, **param_dict)
    elif afx_type == "vibrato":
        y = vibrato(x, m, sr=sr, **param_dict)
    elif afx_type == "flanger":
        y = flanger(x, m, sr=sr, **param_dict)
    elif afx_type == "tremolo":
        y = tremolo(x, m, **param_dict)
    if gain_staging: y = gain_stage(x, y)
    
    return {"main": y}

@partial(jax.jit, backend="cpu", static_argnames="sr")
def phaser(x, modulation, depth=1, centre_frequency_hz=1300, feedback=0, mix=0.5, sr=44100):     
    def phaser_step(s, xs):
        (s11, s12), (s21, s22), (s31, s32), y3 = s
        G, twoR, c_hp, c_bp, c_lp, x = xs
        x1 = x + mix * feedback * y3
        (s11, s12), y1 = tvsvfilt_step((s11, s12), (x1, G, twoR, c_hp, c_bp, c_lp))
        (s21, s22), y2 = tvsvfilt_step((s21, s22), (y1, G, twoR, c_hp, c_bp, c_lp))
        (s31, s32), y3 = tvsvfilt_step((s31, s32), (y2, G, twoR, c_hp, c_bp, c_lp))
        y = x * (1 - mix) + y3 * mix
        return ((s11, s12), (s21, s22), (s31, s32), y3), y

    if modulation.shape[-1] == 2 and x.shape[-1] == 1:
        x = jnp.concatenate([x, x], -1)
        
    ones = jnp.ones_like(x)
    frequency_hz = centre_frequency_hz ** (1 + depth * modulation / 10)
    q = 0.5
    G = hz_to_G(frequency_hz, sr)
    twoR = q_to_twoR(q) * ones
    c_hp, c_bp, c_lp = ones, -twoR, ones
    z = np.zeros(x.shape[-1])
    return lax.scan(phaser_step, ((z, z), (z, z), (z, z), z), (G, twoR, c_hp, c_bp, c_lp, x))[1]

def chorus_const(x, modulation, centre_delay_ms=7.0, feedback=0.0, mix=0.5, sr=44100):
    return chorus(x, modulation, feedback=feedback,
                                 mix=mix,
                                 blend=0.7,
                                 depth=0.5,
                                 max_delay_ms=centre_delay_ms,
                                 centre_delay_ms=centre_delay_ms, 
                                 sr=sr)

def vibrato(x, modulation, depth=0.5, sr=44100):
    return chorus(x, modulation, feedback=0, mix=1.0, blend=0, depth=depth, max_delay_ms=3, centre_delay_ms=0, sr=44100)

def flanger(x, modulation, depth=0.5, sr=44100):
    return chorus(x, modulation, feedback=-0.7, mix=0.7, blend=0.7, depth=depth, centre_delay_ms=0, max_delay_ms=2, sr=44100)

@partial(jax.jit, backend="cpu")
def tremolo(x, modulation, depth=1.0):
    return x * 0.5 + x * modulation * depth * 0.5

def chorus(x, modulation, feedback=0, mix=0.5, blend=0, depth=1, centre_delay_ms=0, max_delay_ms=10, sr=44100, mono=True):
    max_delay = 1 + int(sr*max_delay_ms*2/1000)
    buffer = jnp.zeros(max_delay)
    
    delay_ms_lfo = depth * max_delay_ms * modulation
    delay_ms = centre_delay_ms + delay_ms_lfo
    delay = jnp.minimum(jnp.maximum(1.0, delay_ms / 1000 * sr), max_delay)
    delta, alpha = delay.astype(int), delay % 1.0

    FF = mix
    BL = blend
    return _chorus(x, delta, alpha, buffer, FF, BL, feedback, max_delay)

@partial(jax.jit, backend="cpu")
def _chorus(x, delta, alpha, buffer, FF, BL, feedback, max_delay):
    def chorus_step(state, xs):
        buffer, idx = state
        delta, alpha, x = xs
        # Linear Interpolation
        delay_out = (1 - alpha) * buffer[(idx - delta) % max_delay] + \
                    alpha * buffer[(idx - delta - 1) % max_delay]
        y = FF * delay_out + BL * x
        buffer_in = x - feedback * delay_out
        buffer = buffer.at[idx % max_delay].set(buffer_in)
        return (buffer, idx + 1), y

    _, y = lax.scan(chorus_step, (buffer, 0), (delta, alpha, x))
    return y
    
