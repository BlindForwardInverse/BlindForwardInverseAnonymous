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
import flags as FLAGS
from functools import partial
from afx.filters import hz_to_G, q_to_twoR, tvsvfilt_step
from afx.primitives import get_signal, gain_stage

@partial(jax.jit, backend="cpu")
def phaser(x, modulation, depth=1, centre_frequency_hz=1300, feedback=0, mix=0.5):
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
    G = hz_to_G(frequency_hz)
    twoR = q_to_twoR(q) * ones
    c_hp, c_bp, c_lp = ones, -twoR, ones
    z = np.zeros(x.shape[-1])
    return lax.scan(phaser_step, ((z, z), (z, z), (z, z), z), (G, twoR, c_hp, c_bp, c_lp, x))[1]

@partial(jax.jit, backend="cpu")
def chorus(x, modulation, depth=1, centre_delay_ms=7.0, feedback=0, mix=0.5, max_delay_ms=18):
    if modulation.shape[-1] == 2 and x.shape[-1] == 1:
        x = jnp.concatenate([x, x], -1)

    max_delay = 1+int(FLAGS.sr*max_delay_ms/1000)
    c = x.shape[-1]
    buffer = jnp.zeros((max_delay, c))

    delay_ms_lfo = centre_delay_ms * depth * modulation
    delay_ms = centre_delay_ms + delay_ms_lfo
    delay = jnp.minimum(jnp.maximum(1.0, delay_ms / 1000 * FLAGS.sr), max_delay)
    delta, alpha = delay.astype(int), delay % 1.0
    c_arange = jnp.arange(c)

    def chorus_step(state, xs):
        buffer, idx = state
        delta, alpha, x = xs
        delay_out = (1 - alpha) * buffer[(idx - delta) % max_delay, c_arange] + alpha * buffer[
            (idx - delta - 1) % max_delay, c_arange
        ]
        y = mix * delay_out + (1 - mix) * x
        buffer_in = x - feedback * delay_out
        buffer = buffer.at[idx % max_delay].set(buffer_in[0])
        return (buffer, idx + 1), y

    return lax.scan(chorus_step, (buffer, 0), (delta, alpha, x))[1]

def chorus_ste(x, **kwargs):
    zero = x - lax.stop_gradient(x)
    y_stop = lax.stop_gradient(chorus(x, **kwargs))
    return zero + y_stop

def vibrato(x, modulation):
    return chorus(x, modulation, centre_delay_ms=20.0, mix=1.0)

def chorus_const(x, modulation, centre_delay_ms=7.0, feedback=0.0, mix=0.5):
    return chorus(x, modulation, centre_delay_ms=centre_delay_ms, feedback=feedback, mix=mix)

def flanger(x, modulation, centre_delay_ms=7.0, feedback=0.0, mix=0.5):
    return chorus(x, modulation, centre_delay_ms=centre_delay_ms, feedback=feedback, mix=mix)

def tremolo(x, modulation, depth=1.0):
    return x*(1+modulation*depth)

def apply_modulation_effect(input_signal, afx_type, gain_staging, **param_dict):
    x = get_signal(input_signal, "main")
    m = get_signal(input_signal, "modulation")
    if afx_type == "phaser":
        y = phaser(x, m, **param_dict)
    elif afx_type == "chorus":
        y = chorus_const(x, m, **param_dict)
    elif afx_type == "vibrato":
        y = vibrato(x, m, **param_dict)
    elif afx_type == "flanger":
        y = flanger(x, m, **param_dict)
    elif afx_type == "tremolo":
        y = tremolo(x, m, **param_dict)
    if gain_staging: y = gain_stage(x, y)
    return {"main": y}
