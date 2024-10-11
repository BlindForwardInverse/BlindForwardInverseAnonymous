from functools import partial

import flags as FLAGS
import jax
import jax.numpy as jnp
import numpy as np
from afx.primitives import get_signal
from jax import lax

""" 
controllers
currently supporting: 
    * envelope follower
    * lfo
"""


@partial(jax.jit, backend="cpu")
def _envfollower(
    x, attack_ms: float = 20.0, release_ms: float = 200.0, rms: bool = False
):
    if x.ndim == 2:
        x = jnp.sum(x, -1)
    exp_factor = -2.0 * np.pi * 1000 / FLAGS.sr
    att = lax.cond(
        attack_ms < 1e-3,
        lambda attack_ms: 0.0,
        lambda attack_ms: jnp.exp(exp_factor / attack_ms),
        attack_ms,
    )
    rel = jnp.exp(exp_factor / release_ms)

    def envfollower_step(yold, x):
        e = lax.cond(rms, lambda: x**2, lambda: jnp.abs(x))
        cte = lax.cond(e > yold, lambda: att, lambda: rel)
        y = e + cte * (yold - e)
        env = lax.cond(rms, lambda: jnp.sqrt(y + 1e-12), lambda: y)
        return y, env

    return lax.scan(envfollower_step, 0, x)[1]


def envfollower(
    x,
    attack_ms: float = 20.0,
    release_ms: float = 200.0,
    rms: bool = False,
    gain=1.0,
    scale: bool = False,
    low_db: float = -60.0,
    high_db: float = -15.0,
):
    env = _envfollower(x, attack_ms, release_ms, rms)
    if scale:
        env = 10 * jnp.log10(env + 1e-6)
        env = jnp.maximum(env, -low_db * jnp.ones_like(env))
        env = jnp.minimum(env, -high_db * jnp.ones_like(env))
        env = (env - low_db) / (high_db - low_db)
        return env[:, None] * gain
    else:
        return env


def modulated_lfo(lfo_frequency_hz: float = 4, offset: float = 0):
    phase_inc = 2 * np.pi * lfo_frequency_hz / FLAGS.sr
    phase = jnp.cumsum(phase_inc)
    return jnp.sin(phase + offset)[:, None]


def lfo(lfo_frequency_hz: float = 4, offset: float = 0, signal_len=FLAGS.signal_len):
    phase = jnp.arange(signal_len) * 2 * np.pi * lfo_frequency_hz / FLAGS.sr
    return jnp.sin(phase + offset)[:, None]


def stereo_lfo(
    lfo_frequency_hz: float = 4,
    offset: float = 0,
    phase_delta: float = 3.14,
    signal_len=FLAGS.signal_len,
):
    phase = jnp.arange(signal_len) * 2 * np.pi * lfo_frequency_hz / FLAGS.sr
    phase = jnp.stack([phase, phase + phase_delta], -1)
    return jnp.sin(phase + offset)


def apply_controller(input_signal, afx_type, gain_staging, **param_dict):
    if afx_type == "envfollower":
        x = get_signal(input_signal, "main")
        return {"main": envfollower(x, **param_dict, scale=True)}
    elif afx_type == "lfo":
        return {"main": lfo(**param_dict)}
    elif afx_type == "stereo_lfo":
        return {"main": stereo_lfo(**param_dict)}
