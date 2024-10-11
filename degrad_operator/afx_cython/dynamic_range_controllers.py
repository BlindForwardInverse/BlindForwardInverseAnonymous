"""
dynamic range controllers w/ sidechain
currently supporting: 
    * compressor
    * limiter w/ two-stage compression
    * noisegate (which can be also an expander)
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from afx.controllers import envfollower
from afx.utilities import get_signal, gain_stage
from afx.cafx_core import compressor as compressor_cython
from afx.cafx_core import noisegate as noisegate_cython
import numpy as np


@partial(jax.jit, backend="cpu")
def compressor(x, y, threshold_db=-18, ratio=4, attack_ms=20, release_ms=200, rms=False):
    threshold = 10 ** (threshold_db / 20)
    env = envfollower(y, attack_ms, release_ms, rms)

    def compressor_step(_, env):
        return _, lax.cond(
            env < threshold,
            lambda: jnp.exp(0.0).astype(jnp.float32),
            lambda: jnp.power(env / threshold, 1 / ratio - 1).astype(jnp.float32),
        )

    g = lax.scan(compressor_step, None, env)[1]
    return x * g[:, None]


@partial(jax.jit, backend="cpu")
def compressor_with_knee(
    x, y, threshold_db=-18, ratio=4, attack_ms=20, release_ms=200, knee_db=6, rms=False
):
    env = envfollower(y, attack_ms, release_ms, rms)
    env_db = 20 * jnp.log10(env + 1e-7)

    def compressor_with_knee_step(_, env_db):
        return _, lax.cond(
            env_db < (threshold_db - knee_db / 2),
            lambda: env_db,
            lambda: lax.cond(
                env_db < (threshold_db + knee_db / 2),
                lambda: env_db
                + (1 / ratio - 1) * (env_db - threshold_db + knee_db / 2) ** 2 / 2 / knee_db,
                lambda: threshold_db + (env_db - threshold_db) / ratio,
            ),
        )

    env_out_db = lax.scan(compressor_with_knee_step, None, env_db)[1]
    gain_db = env_out_db - env_db
    gain = 10 ** (gain_db / 20) - 1e-7
    return x * gain[:, None]

#@partial(jax.jit, backend="cpu")
#def compressor_with_knee(
#    x, y, threshold_db=-18, ratio=4, attack_ms=20, release_ms=200, knee_db=6, rms=False
#):
#    x, y = np.array(x), np.array(y)
#    if y.shape[-1] == 2:
#        y = (y[:, 0]+y[:, 1])/2
#    else:
#        y = y[:, 0]
#    y = y.astype(np.float32)
#    cython_out = np.array([compressor_cython(x[:, c].astype(np.float32), y, 
#                                             threshold_db=threshold_db, 
#                                             ratio=ratio,
#                                             attack_ms=attack_ms,
#                                             release_ms=release_ms,
#                                             knee_db=knee_db,
#                                             rms=rms,
#                                             sr=44100) 
#        for c in range(x.shape[-1])], np.float32).T
#    return cython_out

@partial(jax.jit, backend="cpu")
def noisegate(x, y, threshold_db=-100, ratio=10, attack_ms=1, release_ms=100):
    threshold = 10 ** (threshold_db / 20)
    env = envfollower(y, attack_ms=0, release_ms=50, rms=True)
    env = envfollower(env, attack_ms=attack_ms, release_ms=release_ms, rms=False)

    def noisegate_step(_, env):
        return _, lax.cond(
            env > threshold,
            lambda: jnp.exp(0.0).astype(jnp.float32),
            lambda: jnp.power(env / threshold, ratio - 1).astype(jnp.float32),
        )

    g = lax.scan(noisegate_step, None, env)[1]
    return x * g[:, None]


@partial(jax.jit, backend="cpu")
def noisegate_with_knee(
    x, y, threshold_db=-40, ratio=4, attack_ms=20, release_ms=200, knee_db=6, rms=False
):
    env = envfollower(y, attack_ms, release_ms, rms)
    env_db = 20 * jnp.log10(env + 1e-7)

    def noisegate_with_knee_step(_, env_db):
        return _, lax.cond(
            env_db < (threshold_db - knee_db / 2),
            lambda: ratio * (env_db - threshold_db) + threshold_db,
            lambda: lax.cond(
                env_db < (threshold_db + knee_db / 2),
                lambda: env_db
                + (1 - ratio) * (env_db - threshold_db - knee_db / 2) ** 2 / 2 / knee_db,
                lambda: env_db,
            ),
        )

    env_out_db = lax.scan(noisegate_with_knee_step, None, env_db)[1]
    gain_db = env_out_db - env_db
    gain = 10 ** (gain_db / 20) - 1e-7
    return x * gain[:, None]

#def noisegate_with_knee(
#    x, y, threshold_db=-40, ratio=4, attack_ms=20, release_ms=200, knee_db=6, rms=False
#):
#    x, y = np.array(x), np.array(y)
#    if y.shape[-1] == 2:
#        y = (y[:, 0]+y[:, 1])/2
#    else:
#        y = y[:, 0]
#    y = y.astype(np.float32)
#    cython_out = np.array([noisegate_cython(x[:, c].astype(np.float32), y, 
#                                            threshold_db=threshold_db, 
#                                            ratio=ratio,
#                                            attack_ms=attack_ms,
#                                            release_ms=release_ms,
#                                            knee_db=knee_db,
#                                            rms=rms,
#                                            sr=44100) 
#        for c in range(x.shape[-1])], np.float32).T
#    return cython_out


@partial(jax.jit, backend="cpu")
def limiter(x, threshold_db=-18, release_ms=100):
    x = compressor(x, x, -10, 4, 2, 200)
    x = compressor(x, x, threshold_db, 1000, 0.001, release_ms)
    return x

@partial(jax.jit, backend="cpu")
def limiter_with_knee(x, threshold_db=-18, release_ms=100):
    x = compressor_with_knee(x, x, -10, 4, 2, 200)
    x = compressor_with_knee(x, x, threshold_db, 1000, 0.001, release_ms)
    return x

@partial(jax.jit, backend="cpu")
def limiter(x, threshold_db=-18, release_ms=100):
    x = compressor(x, x, -10, 4, 2, 200)
    x = compressor(x, x, threshold_db, 1000, 0.001, release_ms)
    return x

def apply_dynamic_range_controller(input_signal, afx_type, gain_staging, **param_dict):
    x = get_signal(input_signal, "main")
    if afx_type in ["compressor", "inverted_compressor"]:
        sidechain_input = input_signal["sidechain"] if "sidechain" in input_signal else x
        y = compressor_with_knee(x, sidechain_input, **param_dict)
    elif afx_type in ["noisegate", "expander"]:
        sidechain_input = input_signal["sidechain"] if "sidechain" in input_signal else x
        y = noisegate_with_knee(x, sidechain_input, **param_dict)
    elif afx_type == "limiter":
        y = limiter(x, **param_dict)
    if gain_staging: y = gain_stage(x, y)
    return {"main": y}
