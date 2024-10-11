"""
memoryless nonlinearities
currently supproting:
    * parametric memoryless distortion w/ antiderivative antialiasing
    * bitcrusher
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial
from scipy.signal import firwin
from .jafx_utils import gain_stage, get_signal, db_to_amp


def apply_memoryless_nonlinearity(input_signal, afx_type, gain_staging, sr, mono, signal_len, **param_dict):
    x = get_signal(input_signal, "main")
    if afx_type == "distortion":
        y = distortion(x, mono, **param_dict)
    elif afx_type == "bitcrush":
        y = bitcrush(x, **param_dict)
    elif afx_type == "hard_clipper":
        y = hard_clip(x, **param_dict)
    elif afx_type == 'soft_clipper':
        y = soft_clip(x, **param_dict)
    else:
        assert False
    if gain_staging: y = gain_stage(x, y)
    return {"main": y}

def distortion(x, mono=False, gain_db=0, hardness=0.5, asymmetry=0, antialiasing=False, upsample=False):
    # Apply distortion with or without upsampling
    if upsample:
        processed = downsample_2(_distortion(upsample_2(x), gain_db, hardness, asymmetry))
    else:
        processed = _distortion(x, gain_db, hardness, asymmetry)
    return processed

@partial(jax.jit, backend="cpu")
def bitcrush(x, bit_depth=16, **param_dict):
    zero = x - lax.stop_gradient(x)
    scale_factor = 2**bit_depth
    y = x * scale_factor
    y = jnp.round(y)
    y = y / scale_factor
    return zero + lax.stop_gradient(y)

@partial(jax.jit, backend="cpu")
def hard_clip(x, gain_db=0):
    x = x*db_to_amp(gain_db)
    x = jnp.maximum(jnp.minimum(x, 1), -1)
#     c = jnp.ones(x.shape
#     x = (jnp.abs(x + c) - jnp.abs(x - c))/2
    return x

@partial(jax.jit, backend="cpu")
def soft_clip(x, factor=1.5):
    return (2/jnp.pi) * jnp.arctan(x * factor)

@partial(jax.jit, backend="cpu")
def upsample_2(x):
    x = jnp.stack([x, jnp.zeros_like(x)], -1)  # t 2
    x = x.reshape(-1)  # (t 2)
    fir_n = 50
    lpfir_2 = firwin(2 * fir_n + 1, 0.5)
    x = jax.scipy.signal.convolve(x, lpfir_2)[fir_n:-fir_n]
    return x


@partial(jax.jit, backend="cpu")
def downsample_2(x):
    fir_n = 50
    lpfir_2 = firwin(2 * fir_n + 1, 0.5)
    x = jax.scipy.signal.convolve(x, lpfir_2)[fir_n:-fir_n]
    x = x.reshape(-1, 2)[:, 0]
    return x


@partial(jax.jit, backend="cpu")
def _distortion(x, gain_db=0, hardness=0.5, asymmetry=0, eps=1e-3, antialiasing=False):
    gain = db_to_amp(gain_db)
    kp, kn, gp_db, gn_db = (
        (1 + asymmetry) / 2,
        (1 - asymmetry) / 2,
        24 * hardness,
        24 * hardness,
    )
    log, tanh, cosh = jnp.log, jnp.tanh, jnp.cosh

    def logcosh(x):
        return lax.cond(
            x > 10,
            lambda _: jnp.abs(x) - jnp.log(2),
            lambda _: jnp.log(jnp.cosh(x)),
            None,
        )

    gp, gn = 10 ** (gp_db / 20), 10 ** (gn_db / 20)
    ap, an = (1 - jnp.tanh(kp)) / gp, (1 - jnp.tanh(kn)) / gn
    bp, bn = jnp.tanh(kp), -jnp.tanh(kn)
    cp, cn = logcosh(kp) - kp * jnp.tanh(kp), logcosh(kn) - kn * jnp.tanh(kn)

    def M(x):
        return lax.cond(
            x > kp,
            lambda _: bp * x + ap / gp * logcosh(gp * (x - kp)) + cp,
            lambda _: lax.cond(
                x < -kn,
                lambda _: bn * x + an / gn * logcosh(gn * (x + kn)) + cn,
                lambda _: logcosh(x),
                None,
            ),
            None,
        )

    def m(_, x):
        return x, lax.cond(
            x > kp,
            lambda: ap * jnp.tanh(gp * (x - kp)) + bp,
            lambda: lax.cond(
                x < -kn, lambda: an * jnp.tanh(gn * (x + kn)) + bn, lambda: jnp.tanh(x)
            ),
        )

    def dM(xold, x):
        return x, lax.cond(
            jnp.abs(x - xold) < eps,
            lambda _: m(None, (x + xold) / 2)[1],
            lambda _: (M(x) - M(xold)) / (x - xold),
            None,
        )

    return lax.cond(
        antialiasing,
        # lambda _: lax.scan(dM, 0, x.astype(jnp.float64)*gain)[1].astype(jnp.float32),
        lambda _: lax.scan(dM, 0, x * gain)[1],
        lambda _: lax.scan(m, 0, x.astype(jnp.float32) * gain)[1].astype(jnp.float32),
        None,
    )
