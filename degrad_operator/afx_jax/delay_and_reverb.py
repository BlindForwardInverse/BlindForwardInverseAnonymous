from functools import partial
from time import time

import jax
import jax.numpy as jnp
import numpy as np
from .jafx_utils import db_to_amp, gain_stage, get_signal, hz_to_G, q_to_twoR
from .state_variable_filters import ltisvfilt
from jax import lax

"""
delay and reverb
currently supporting:
    * reverb 
    * delay with bandpass filter in the feedback loop
"""


def apply_delay_and_reverb(
    input_signal, afx_type, gain_staging, sr, mono, signal_len, **param_dict
):
    x = get_signal(input_signal, "main")
    if afx_type == "delay":
        y = delay(x, sr=sr, mono=mono, **param_dict)
    elif afx_type == "pingpong":
        y = pingpong(x, sr=sr, mono=mono, **param_dict)
    elif afx_type == "mono_reverb":
        y = mono_reverb(x, mono=mono, **param_dict)
    else:
        assert False
    if gain_staging:
        y = gain_stage(x, y)
    return {"main": y}


def delay(x, delay_seconds=0.4, feedback_gain_db=-1, mix=1, max_dlen=24000, sr=44100, mono=False):
    dlen = int(delay_seconds * sr)
    buffer_shape = (max_dlen,) if mono else (max_dlen, 2)
    return _delay(x, dlen=dlen, feedback_gain_db=feedback_gain_db, mix=mix, sr=sr, buffer_shape=buffer_shape, max_dlen=max_dlen)

@partial(jax.jit, backend="cpu", static_argnames=["sr", "buffer_shape", "max_dlen"])
def _delay(x,dlen=0.4, feedback_gain_db=-1, mix=1, sr=44100, buffer_shape=(24000,2), max_dlen=24000):
    gain = db_to_amp(feedback_gain_db)
    delay_buffer = jnp.zeros(buffer_shape)
    def feedback_loop(state, x):
        delay_buffer, idx = state
        y = x + delay_buffer[idx]
        delay_buffer = delay_buffer.at[(idx + dlen) % max_dlen].set(y * gain)
        idx = (idx + 1) % max_dlen
        return (delay_buffer, idx), y

    out = lax.scan(feedback_loop, (delay_buffer, 0), x)[1]
    return mix * out + (1 - mix) * x

def pingpong(x, sr=44100, mono=False, delay_seconds=0.2, feedback_gain_db=-1, mix=1, lr_ratio=0.5):
    pad_len = int(delay_seconds * (1 - lr_ratio) * sr)
    if mono : l, r = x, x #ordinary delay
    else : l, r = x[:, :1], x[:, 1:] #(T, 1), (T, 1)
    
    x_in = (l + r) / 2
    x_in = jnp.pad(array=x_in, pad_width=((0, pad_len), (0, 0)))
    delay_out = delay(
        x_in,
        delay_seconds=delay_seconds,
        feedback_gain_db=feedback_gain_db,
        mix=1,
        sr=sr,
        mono=False,
    )
    y = jnp.stack([delay_out[pad_len:, 0], delay_out[:-pad_len, 1]], -1)
    return mix * y + (1 - mix) * x

def mono_reverb(x, room_size=0.5, damping=0.5, mix=1, stereo_spread=0, mono=True):
    if mono: return freeverb(x, room_size=room_size, damping=damping, mix=mix, stereo_spread=stereo_spread) 
    else:
        l, r = x[:, 0], x[:, 1]
        reverb_l = freeverb(l, room_size=room_size, damping=damping, mix=mix, stereo_spread=stereo_spread)
        reverb_r = freeverb(r, room_size=room_size, damping=damping, mix=mix, stereo_spread=stereo_spread)
        return jnp.stack([reverb_l, reverb_r], -1)

@partial(jax.jit, backend="cpu", static_argnames="stereo_spread")
def freeverb(x, room_size=0.5, damping=0.5, mix=1, stereo_spread=0):
    damping, feedback = damping * 0.4, 0.28 * room_size + 0.7
    y = (
        lpcomb(
            x, jnp.zeros(1116 + stereo_spread), 1116 + stereo_spread, feedback, damping
        )
        + lpcomb(
            x, jnp.zeros(1188 + stereo_spread), 1188 + stereo_spread, feedback, damping
        )
        + lpcomb(
            x, jnp.zeros(1277 + stereo_spread), 1277 + stereo_spread, feedback, damping
        )
        + lpcomb(
            x, jnp.zeros(1356 + stereo_spread), 1356 + stereo_spread, feedback, damping
        )
        + lpcomb(
            x, jnp.zeros(1422 + stereo_spread), 1422 + stereo_spread, feedback, damping
        )
        + lpcomb(
            x, jnp.zeros(1491 + stereo_spread), 1491 + stereo_spread, feedback, damping
        )
        + lpcomb(
            x, jnp.zeros(1557 + stereo_spread), 1557 + stereo_spread, feedback, damping
        )
        + lpcomb(
            x, jnp.zeros(1617 + stereo_spread), 1617 + stereo_spread, feedback, damping
        )
    )
    y = y * 0.015
    y = freeverb_allpass(y, jnp.zeros(556 + stereo_spread), 556 + stereo_spread)
    y = freeverb_allpass(y, jnp.zeros(441 + stereo_spread), 441 + stereo_spread)
    y = freeverb_allpass(y, jnp.zeros(341 + stereo_spread), 341 + stereo_spread)
    y = freeverb_allpass(y, jnp.zeros(225 + stereo_spread), 225 + stereo_spread)
    y = mix * y + (1 - mix) * x
    return y

def lpcomb(x, buffer, delay_len=1116, feedback=0.5, damping=0.5):
    def step(state, x):
        buffer, u, idx = state
        y = feedback * buffer[idx % delay_len]
        u = y * (1 - damping) + u * damping
        buffer = buffer.at[idx % delay_len].set(x + feedback * u) # 
        return (buffer, u, idx + 1), y

    _, y = lax.scan(step, (buffer, 0.0, 0), x)
    return y
    # return lax.scan(step, (buffer, 0, 0), x)[1]

def freeverb_allpass(x, buffer, delay_len=1116):
    def step(state, x):
        buffer, idx = state
        u = buffer[idx % delay_len]
        y = u - x
        buffer = buffer.at[idx % delay_len].set(x + 0.5 * u)
        return (buffer, idx + 1), y

    return lax.scan(step, (buffer, 0), x)[1]
