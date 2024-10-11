import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from functools import partial
from .jafx_utils import get_signal, db_to_amp, hz_to_G, q_to_twoR
from .state_variable_filters import ltisvfilt

""" 
controllers
currently supporting: 
    * envelope follower
    * lfo
"""

def apply_controller(input_signal, afx_type, gain_staging, sr, signal_len, mono, **param_dict):
    if afx_type == "envfollower":
        x = get_signal(input_signal, "main")
        return {"main": envfollower(x, sr=sr, **param_dict, scale=True)}
    elif afx_type == "lfo":
        return {"main": mono_lfo(sr=sr, signal_len=signal_len, **param_dict)}
    elif afx_type == "stereo_lfo":
        return {"main": stereo_lfo(sr=sr, signal_len=signal_len, **param_dict)}
    elif afx_type == "lowpass_noise":
        return {"main": lowpass_noise(sr=sr, signal_len=signal_len, **param_dict)}
    else:
        raise NotImplementedError("Not implemented controller type")

def envfollower(
    x,
    attack_ms: float = 20.0,
    release_ms: float = 200.0,
    rms: bool = False,
    gain=1.0,
    scale: bool = False,
    low_db: float = -60.0,
    high_db: float = -15.0,
    sr : int = 44100,
    **param_dict
):
    env = _envfollower(x, attack_ms, release_ms, rms, sr)
    if scale:
        env = 10 * jnp.log10(env + 1e-6)
        env = jnp.maximum(env, -low_db * jnp.ones_like(env))
        env = jnp.minimum(env, -high_db * jnp.ones_like(env))
        env = (env - low_db) / (high_db - low_db)
        return env[:, None] * gain
    else:
        return env

@partial(jax.jit, backend="cpu", static_argnames='sr')
def _envfollower(x, attack_ms: float = 20.0, release_ms: float = 200.0, rms: bool = False, sr: int = 44100):
    if x.ndim == 2:
        x = jnp.sum(x, -1)
    exp_factor = -2.0 * jnp.pi * 1000 / sr
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

@partial(jax.jit, backend="cpu", static_argnames=['signal_len', 'sr'])
def mono_lfo(lfo_frequency_hz: float = 4, offset: float = 0, signal_len: int = 44100, sr: int = 44100):
    phase = jnp.arange(signal_len) * 2 * np.pi * lfo_frequency_hz / sr
    return jnp.sin(phase + offset)

@partial(jax.jit, backend="cpu", static_argnames=['signal_len', 'sr'])
def stereo_lfo(
    lfo_frequency_hz: float = 4,
    offset: float = 0,
    phase_delta: float = 0,
    signal_len: int = 44100,
    sr: int = 44100,
):
    phase = jnp.arange(signal_len) * 2 * np.pi * lfo_frequency_hz / sr
    phase = jnp.stack([phase, phase + phase_delta], -1)
    return jnp.sin(phase + offset)

def lowpass_noise(
    seed: int = 42,
    cutoff_freq: float = 10,
    signal_len: int = 44100,
    sr: int = 44100
    ):
    key = jax.random.PRNGKey(seed)
    white = jax.random.normal(key, shape = signal_len)

    G = hz_to_G(cutoff_freq, sr)
    twoR = q_to_twoR(q=1)
    lowpassed_white = ltisvfilt(white, G, twoR, c_hp=0, c_bp=0, c_lp=1)
    return lowpassed_white
