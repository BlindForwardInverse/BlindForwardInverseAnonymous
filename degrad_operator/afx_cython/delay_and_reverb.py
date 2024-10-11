import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import flags as FLAGS
from functools import partial
from afx.filters import hz_to_G, q_to_twoR, db_to_amp, ltisvfilt
from afx.primitives import get_signal, gain_stage
from afx.cafx_core import delay as delay_cython
from afx.cafx_core import pingpong as pingpong_cython
from time import time
from afx.augment import apply_pedalboard_effect

"""
delay and reverb
currently supporting:
    * reverb 
    * delay with bandpass filter in the feedback loop
"""


@partial(jax.jit, backend="cpu")
def state_delay(x, delay_seconds=0.4, gain_db=-1, max_dlen=24000):
    c = x.shape[-1]
    dlen = int(delay_seconds * FLAGS.sr)
    gain = db_to_amp(gain_db)
    delay_buffer = jnp.zeros((max_dlen, c))

    def step(state, x):
        delay_buffer, idx = state
        y = x + delay_buffer[idx]
        delay_buffer = delay_buffer.at[(idx + dlen) % max_dlen].set(y * gain)
        idx = (idx + 1) % max_dlen
        return (delay_buffer, idx), y

    return lax.scan(step, (delay_buffer, 0), x)[1]


# from jax lib
def overlap_add(x, step_size):
    *batch_shape, nframes, segment_len = x.shape
    flat_batchsize = np.prod(batch_shape, dtype=np.int64)
    x = x.reshape((flat_batchsize, nframes, segment_len))
    output_size = step_size * (nframes - 1) + segment_len
    nstep_per_segment = 1 + (segment_len - 1) // step_size

    padded_segment_len = nstep_per_segment * step_size
    x = jnp.pad(x, ((0, 0), (0, 0), (0, padded_segment_len - segment_len)))
    x = x.reshape((flat_batchsize, nframes, nstep_per_segment, step_size))

    x = x.transpose((0, 2, 1, 3))  # x: (B, S, N, T)
    x = jnp.pad(x, ((0, 0), (0, 0), (0, nframes), (0, 0)))  # x: (B, S, N*2, T)
    shrinked = x.shape[2] - 1
    x = x.reshape((flat_batchsize, -1))
    x = x[:, : (nstep_per_segment * shrinked * step_size)]
    x = x.reshape((flat_batchsize, nstep_per_segment, shrinked * step_size))

    x = x.sum(axis=1)[:, :output_size]
    return x.reshape(tuple(batch_shape) + (-1,))


#def delay(
#    x,
#    delay_seconds=0.5,
#    feedback_gain_db=-1,
#    mix=0.5,
#    frequency_hz=800,
#    q=2,
#    tail_len=1000,
#):
#    c = x.shape[-1]
#    dlen = int(delay_seconds * FLAGS.sr)
#    num_render = len(x) // dlen
#    buffer = jnp.zeros((dlen + tail_len, c))
#    G, twoR, gain = hz_to_G(frequency_hz), q_to_twoR(q), db_to_amp(feedback_gain_db)
#    print(G, twoR, gain)
#    if len(x) % dlen != 0:
#        num_render += 1
#    delay_outputs = []
#    for i in range(num_render):
#        delay_outputs.append(buffer.T)
#        if i != num_render - 1:
#            x_i = x[i * dlen : (i + 1) * dlen]
#            x_i = jnp.pad(x_i, ((0, dlen - len(x_i) + tail_len), (0, 0)))
#            filter_input = x_i + buffer * gain
#            buffer = ltisvfilt(filter_input, G, twoR, 0, twoR, 0)
#    y = overlap_add(jnp.stack(delay_outputs, -2), dlen).T[: len(x)]
#    return mix * y + (1 - mix) * x
#
def delay(x, **kwargs):
    def apply_cython_delay(x, **kwargs):
        return np.array(delay_cython(np.array(x, dtype=np.float32), sr=FLAGS.sr, **kwargs))
    return np.stack([apply_cython_delay(x[:, c], **kwargs) for c in range(x.shape[-1])], -1)


#def pingpong(x, mix=1, lr_ratio=0.5, delay_seconds=0.2, **param_dict):
#    t0 = time()
#    pad_len = int(delay_seconds * (1 - lr_ratio) * FLAGS.sr)
#    if x.shape[-1] == 1:
#        l, r = x, x
#    elif x.shape[-1] == 2:
#        l, r = x[:, :1], x[:, 1:]
#    else:
#        assert False
#    x_in = (l + r) / 2
#    x_in = jnp.pad(x_in, ((0, pad_len), (0, 0)))
#    delay_out = delay(x_in, delay_seconds=delay_seconds, mix=1, **param_dict)
#    y = jnp.concatenate([delay_out[pad_len:], delay_out[:-pad_len]], -1)
#    return mix * y + (1 - mix) * x

def pingpong(x, mix=1, lr_ratio=0.5, delay_seconds=0.2, **kwargs):
    return np.array(pingpong_cython(np.array(x, dtype=np.float32), mix=mix, delay_seconds=delay_seconds, lr_ratio=lr_ratio, sr=FLAGS.sr, **kwargs))


@partial(jax.jit, backend="cpu")
def lpcomb(x, buffer, delay_len=1116, feedback=0.5, damping=0.5):
    def step(state, x):
        buffer, u, idx = state
        y = feedback * buffer[idx % delay_len]
        u = y * (1 - damping) + u * damping
        buffer = buffer.at[idx % delay_len].set(x + feedback * u)
        return (buffer, u, idx + 1), y

    return lax.scan(step, (buffer, 0, 0), x)[1]


@partial(jax.jit, backend="cpu")
def freeverb_allpass(x, buffer, delay_len=1116):
    def step(state, x):
        buffer, idx = state
        u = buffer[idx % delay_len]
        y = u - x
        buffer = buffer.at[idx % delay_len].set(x + 0.5 * u)
        return (buffer, idx + 1), y

    return lax.scan(step, (buffer, 0), x)[1]


@partial(jax.jit, backend="cpu", static_argnums=(3,))
def mono_reverb_wet(x, room_size=0.5, damping=0.5, stereo_spread=0):
    x = x[:, 0]
    damping, feedback = damping * 0.4, 0.28 * room_size + 0.7
    y = (
        lpcomb(x, jnp.zeros(1116 + stereo_spread), 1116 + stereo_spread, feedback, damping)
        + lpcomb(x, jnp.zeros(1188 + stereo_spread), 1188 + stereo_spread, feedback, damping)
        + lpcomb(x, jnp.zeros(1277 + stereo_spread), 1277 + stereo_spread, feedback, damping)
        + lpcomb(x, jnp.zeros(1356 + stereo_spread), 1356 + stereo_spread, feedback, damping)
        + lpcomb(x, jnp.zeros(1422 + stereo_spread), 1422 + stereo_spread, feedback, damping)
        + lpcomb(x, jnp.zeros(1491 + stereo_spread), 1491 + stereo_spread, feedback, damping)
        + lpcomb(x, jnp.zeros(1557 + stereo_spread), 1557 + stereo_spread, feedback, damping)
        + lpcomb(x, jnp.zeros(1617 + stereo_spread), 1617 + stereo_spread, feedback, damping)
    )
    y = y * 0.015
    y = freeverb_allpass(y, jnp.zeros(556 + stereo_spread), 556 + stereo_spread)
    y = freeverb_allpass(y, jnp.zeros(441 + stereo_spread), 441 + stereo_spread)
    y = freeverb_allpass(y, jnp.zeros(341 + stereo_spread), 341 + stereo_spread)
    y = freeverb_allpass(y, jnp.zeros(225 + stereo_spread), 225 + stereo_spread)
    return y[:, None]


#def mono_reverb(x, mix=0.6, **param_dict):
#    if x.shape[-1] == 1:
#        y = mono_reverb_wet(x, **param_dict)
#    elif x.shape[-1] == 2:
#        y = jnp.concatenate(
#            [
#                mono_reverb_wet(x[:, :1], **param_dict),
#                mono_reverb_wet(x[:, 1:], **param_dict),
#            ],
#            -1,
#        )
#    return (1 - mix) * x + mix * y


#def stereo_reverb(x, width=0.5, mix=0.6, **param_dict):
#    if x.shape[-1] == 1:
#        l_dry, r_dry = x, x
#    elif x.shape[-1] == 2:
#        l_dry, r_dry = x[:, :1], x[:, 1:]
#    l_wet = mono_reverb_wet(l_dry, **param_dict)
#    r_wet = mono_reverb_wet(r_dry, stereo_spread=23, **param_dict)
#    l_mix = (1 - mix) * l_dry + mix * ((1 + width) * l_wet + (1 - width) * r_wet) / 2
#    r_mix = (1 - mix) * r_dry + mix * ((1 + width) * r_wet + (1 - width) * l_wet) / 2
#    return jnp.concatenate([l_mix, r_mix], -1)

def mono_reverb(x, width=0.5, mix=0.6, damping=0.5, room_size=0.5):
    out = apply_pedalboard_effect(np.array(x).T, "reverb", FLAGS.sr, 
            width=width, wet_level=mix*0.015, dry_level=1-mix, damping=damping, room_size=room_size)
    return out.T

def stereo_reverb(x, width=0.5, mix=0.6, damping=0.5, room_size=0.5):
    x = np.array(x)
    if x.shape[-1] == 1: x = np.concatenate([x, x], -1)
    return apply_pedalboard_effect(np.array(x).T, "reverb", FLAGS.sr, 
            width=width, wet_level=mix*0.015, dry_level=1-mix, damping=damping, room_size=room_size).T

# def decorrelate(x):


def apply_delay_and_reverb(input_signal, afx_type, gain_staging, **param_dict):
    x = get_signal(input_signal, "main")
    if afx_type == "delay":
        y = delay(x, **param_dict)
    elif afx_type == "pingpong":
        y = pingpong(x, **param_dict)
    elif afx_type == "mono_reverb":
        y = mono_reverb(x, **param_dict)
    elif afx_type == "reverb":
        y = stereo_reverb(x, **param_dict)
    elif afx_type == "decorrelate":
        pass
    else: 
        assert False
    if gain_staging: y = gain_stage(x, y)
    return {"main": y}
