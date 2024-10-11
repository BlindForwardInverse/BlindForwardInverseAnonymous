from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from .jafx_utils import db_to_amp, gain_stage, get_signal, hz_to_G, q_to_twoR
from jax import lax


def apply_2nd_order_filter(
    input_signal,
    afx_type,
    gain_staging,
    frequency_hz=440,
    q=0.5,
    gain_db=0,
    c_hp=1,
    c_bp=1,
    c_lp=1,
    sr=44100,
    mono=True,
    **param_dict
):
    def get_coefficients(afx_type, twoR=None, gain=None):
        if afx_type in ["lowpass", "bandpass", "highpass", "bandreject"]:
            zero, one = jnp.zeros_like(twoR), jnp.ones_like(twoR)
            if afx_type == "lowpass":
                return zero, zero, one
            elif afx_type == "bandpass":
                return zero, twoR, zero
            elif afx_type == "highpass":
                return one, zero, zero
            elif afx_type == "bandreject":
                return one, zero, one
        elif afx_type in ["lowshelf", "highshelf", "bell"]:
            one = jnp.ones_like(gain)
            gain = one * gain
            if afx_type == "lowshelf":
                return one, twoR * jnp.sqrt(gain), gain
            elif afx_type == "highshelf":
                return gain, twoR * jnp.sqrt(gain), one
            elif afx_type == "bell":
                return one, twoR * gain, one

    x = get_signal(input_signal, "main")
    original_shape = x.shape
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    twoR = q_to_twoR(q)
    tv = False
    
    if "frequency_hz" in input_signal:
        frequency_hz = jnp.power(
            2, jnp.log2(frequency_hz) + input_signal["frequency_hz"]
        )
        frequency_hz = jnp.minimum(frequency_hz, jnp.ones_like(frequency_hz) * sr / 2.5)
        tv = True
    if "gain_db" in input_signal:
        gain_db = gain_db + 12 * input_signal["gain_db"]
        tv = True
        
    G = hz_to_G(frequency_hz, sr)
    gain = db_to_amp(gain_db)

    if tv:
        ones = jnp.ones((len(x), 1))
        twoR = ones * twoR
        if afx_type != "svf":
            c_hp, c_bp, c_lp = get_coefficients(afx_type, twoR, gain)
        else:
            c_hp, c_bp, c_lp = jnp.array(c_hp), jnp.array(c_bp), jnp.array(c_lp)
        if c_hp.ndim == 0:
            c_hp = c_hp * ones
        if c_bp.ndim == 0:
            c_bp = c_bp * ones
        if c_lp.ndim == 0:
            c_lp = c_lp * ones
        if G.ndim == 0:
            G = G * ones
        y = tvsvfilt(x, G, twoR, c_hp, c_bp, c_lp)
    else:
        if afx_type != "svf":
            c_hp, c_bp, c_lp = get_coefficients(afx_type, twoR, gain)
        else:
            c_hp, c_bp, c_lp = jnp.array(c_hp), jnp.array(c_bp), jnp.array(c_lp)
        
        y = ltisvfilt(x, G, twoR, c_hp, c_bp, c_lp)  #
        
    if gain_staging:
        y = gain_stage(x, y)
        
    y = y.reshape(original_shape)
    return {"main": y}


@partial(jax.jit, backend="cpu")
def ltisvfilt(x, G: float, twoR: float, c_hp: float, c_bp: float, c_lp: float):
    original_shape = x.shape
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    b0 = c_hp + c_bp * G + c_lp * G**2
    b1 = -c_hp * 2 + c_lp * 2 * G**2
    b2 = c_hp - c_bp * G + c_lp * G**2
    a0 = 1 + G**2 + twoR * G
    a1 = 2 * G**2 - 2
    a2 = 1 + G**2 - twoR * G
    b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
    
    z = jnp.zeros(x.shape[-1])

    def ltisvfilt_step(s, x):
        s1, s2 = s
        y = b0 * x + s1
        s1 = b1 * x + s2 - a1 * y
        s2 = b2 * x - a2 * y
        return (s1, s2), y
    
    y = lax.scan(ltisvfilt_step, (z, z), x)[1]
    
    return y.reshape(original_shape)
    # return lax.scan(ltisvfilt_step, (z, z), x)[1]  #


@partial(jax.jit, backend="cpu")
def tvsvfilt(x, G, twoR, c_hp, c_bp, c_lp):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[1:] == (1,):
        x = jnp.tile(x, (1, 2))
    
    z = jnp.zeros((2, ))
    # z = jnp.zeros(max(x.shape[-1], G.shape[-1], c_hp.shape[-1])) # works for stereo

    return lax.scan(tvsvfilt_step, ((z), (z)), (x, G, twoR, c_hp, c_bp, c_lp))[1]

def tvsvfilt_step(s, xs):
    s1, s2 = s
    x, G, twoR, c_hp, c_bp, c_lp = xs
    y_bp = (G * (x - s2) + s1) / (1 + G * (G + twoR))
    y_lp = G * y_bp + s2
    y_hp = x - y_lp - twoR * y_bp
    y = c_hp * y_hp + c_bp * y_bp + c_lp * y_lp
    s1 = 2 * y_bp - s1
    s2 = 2 * y_lp - s2
    
    return (s1, s2), y
