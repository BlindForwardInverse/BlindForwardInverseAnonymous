import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from functools import partial
from .jafx_utils import get_signal, gain_stage, hz_to_G

def apply_ladder(input_signal, afx_type, gain_staging, sr, signal_len, mono, frequency_hz=440, k=0, q=None, **param_dict):
    x = get_signal(input_signal, "main")
    if "frequency_hz" in input_signal:
        frequency_hz = jnp.power(2, jnp.log2(frequency_hz) + input_signal["frequency_hz"])
        frequency_hz = jnp.minimum(frequency_hz, jnp.ones_like(frequency_hz) * sr / 2.5)
    G = hz_to_G(frequency_hz, sr)
    if G.shape == ():
        if afx_type == "lowpass_ladder":
            y = lowpass_ladder(x, G, k)
        elif afx_type == "highpass_ladder":
            y = highpass_ladder(x, G, k)
        elif afx_type == "bandpass_ladder":
            y = bandpass_ladder(x, G, k)
        else:
            assert False
    else:
        if afx_type == "lowpass_ladder":
            y = tv_lowpass_ladder(x, G, k)
        elif afx_type == "highpass_ladder":
            y = tv_highpass_ladder(x, G, k)
        elif afx_type == "bandpass_ladder":
            y = tv_bandpass_ladder(x, G, k)
        else:
            assert False
    if gain_staging: y = gain_stage(x, y)
    
    return {"main": y}

@partial(jax.jit, backend="cpu")
def lowpass_ladder(x, G, k):
    z = 0
    G_lp = G/(1+G)
    G_hp = 1/(1+G)
    G_lp2 = G_lp*G_lp; G_lp3 = G_lp2*G_lp; G_lp4 = G_lp3*G_lp
    u_div = 1/(1+4*k*G_lp4)

    def lowpass_ladder_step(s, x):
        (s1, s2, s3, s4) = s
        S1, S2, S3, S4 = s1*G_hp, s2*G_hp, s3*G_hp, s4*G_hp
        S = G_lp3*S1 + G_lp2*S2 + G_lp*S3 + S4
        u = (x-4*k*S)*u_div
        s1, y = onepole_lowpass_step(s1, (G, u))
        s2, y = onepole_lowpass_step(s2, (G, y))
        s3, y = onepole_lowpass_step(s3, (G, y))
        s4, y = onepole_lowpass_step(s4, (G, y))
        return (s1, s2, s3, s4), y

    return lax.scan(lowpass_ladder_step, (z, z, z, z), x)[1]

@partial(jax.jit, backend="cpu")
def tv_lowpass_ladder(x, G, k):
    z = jnp.zeros(max(x.shape[-1], G.shape[-1]))

    def lowpass_ladder_step(s, xs):
        (s1, s2, s3, s4) = s
        G, x = xs
        _G = G / (1 + G)
        _s1, _s2, _s3, _s4 = s1 / (1 + G), s2 / (1 + G), s3 / (1 + G), s4 / (1 + G)
        _G2 = _G*_G; _G3 = _G2*_G; _G4 = _G3*_G
        S = _G3 * _s1 + _G2 * _s2 + _G * _s3 + _s4
        u = (x - 4 * k * S) / (1 + 4 * k * _G4)
        s1, y = onepole_lowpass_step(s1, (G, u))
        s2, y = onepole_lowpass_step(s2, (G, y))
        s3, y = onepole_lowpass_step(s3, (G, y))
        s4, y = onepole_lowpass_step(s4, (G, y))
        return (s1, s2, s3, s4), y

    return lax.scan(lowpass_ladder_step, (z, z, z, z), (G, x))[1]

@partial(jax.jit, backend="cpu")
def highpass_ladder(x, G, k):
    z = 0
    G_hp = 1/(1+G)
    G_hp2 = G_hp*G_hp; G_hp3 = G_hp2*G_hp; G_hp4 = G_hp3*G_hp
    u_div = 1/(1+4*k*G_hp4)

    def highpass_ladder_step(s, x):
        (s1, s2, s3, s4) = s
        S1, S2, S3, S4 = -s1*G_hp, -s2*G_hp, -s3*G_hp, -s4*G_hp
        S = G_hp3*S1 + G_hp2*S2 + G_hp*S3 + S4
        u = (x-4*k*S)*u_div
        s1, y = onepole_highpass_step(s1, (G, u))
        s2, y = onepole_highpass_step(s2, (G, y))
        s3, y = onepole_highpass_step(s3, (G, y))
        s4, y = onepole_highpass_step(s4, (G, y))
        return (s1, s2, s3, s4), y

    return lax.scan(highpass_ladder_step, (z, z, z, z), x)[1]


@partial(jax.jit, backend="cpu")
def tv_highpass_ladder(x, G, k):
    z = jnp.zeros(max(x.shape[-1], G.shape[-1]))

    def highpass_ladder_step(s, xs):
        (s1, s2, s3, s4) = s
        G, x = xs
        _G = 1 / (1 + G)
        _s1, _s2, _s3, _s4 = -s1 / (1 + G), -s2 / (1 + G), -s3 / (1 + G), -s4 / (1 + G)
        S = _G**3 * _s1 + _G**2 * _s2 + _G * _s3 + _s4
        u = (x - 4 * k * S) / (1 + 4 * k * _G**4)
        s1, y = onepole_highpass_step(s1, (G, u))
        s2, y = onepole_highpass_step(s2, (G, y))
        s3, y = onepole_highpass_step(s3, (G, y))
        s4, y = onepole_highpass_step(s4, (G, y))
        return (s1, s2, s3, s4), y

    return lax.scan(highpass_ladder_step, (z, z, z, z), (G, x))[1]


@partial(jax.jit, backend="cpu")
def bandpass_ladder(x, G, k):
    z = 0
    G_lp = G/(1+G)
    G_hp = 1/(1+G)
    G_hplp = G_hp*G_lp; G_hp2lp = G_hplp*G_hp; G_hp2lp2 = G_hp2lp*G_lp
    u_div = 1/(1-4*k*G_hp2lp2)

    def bandpass_ladder_step(s, x):
        (s1, s2, s3, s4) = s
        S1, S2, S3, S4 = s1*G_hp, -s2*G_hp, s3*G_hp, -s4*G_hp
        S = G_hp2lp*S1 + G_hplp*S2 + G_hp*S3 + S4
        u = (x+4*k*S)*u_div
        
        s1, y = onepole_lowpass_step(s1, (G, u))
        s2, y = onepole_highpass_step(s2, (G, y))
        s3, y = onepole_lowpass_step(s3, (G, y))
        s4, y = onepole_highpass_step(s4, (G, y))
        return (s1, s2, s3, s4), y

    return lax.scan(bandpass_ladder_step, (z, z, z, z), x)[1]

@partial(jax.jit, backend="cpu")
def tv_bandpass_ladder(x, G, k):
    z = jnp.zeros(max(x.shape[-1], G.shape[-1]))

    def bandpass_ladder_step(s, xs):
        (s1, s2, s3, s4) = s
        G, x = xs
        G_lp = G / (1 + G)
        G_hp = 1 / (1 + G)
        _s1, _s2, _s3, _s4 = s1 / (1 + G), -s2 / (1 + G), s3 / (1 + G), -s4 / (1 + G)
        S = G_hp**2 * G_lp * _s1 + G_hp * G_lp * _s2 + G_hp * _s3 + _s4
        u = (x + 4 * k * S) / (1 - 4 * k * G_lp**2 * G_hp**2)
        s1, y = onepole_lowpass_step(s1, (G, u))
        s2, y = onepole_highpass_step(s2, (G, y))
        s3, y = onepole_lowpass_step(s3, (G, y))
        s4, y = onepole_highpass_step(s4, (G, y))
        return (s1, s2, s3, s4), y

    return lax.scan(bandpass_ladder_step, (z, z, z, z), (G, x))[1]

def onepole_lowpass_step(s, xs):
    G, x = xs
    v = (x - s) * G / (1 + G)
    y = v + s
    s = y + v
    return s, y

def onepole_highpass_step(s, xs):
    G, x = xs
    v = x - s
    y = v / (1 + G)
    s = s + 2 * y * G
    return s, y
