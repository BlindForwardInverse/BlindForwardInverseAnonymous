import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import flags as FLAGS
from functools import partial
from afx.primitives import get_signal, gain_stage
from scipy import signal
import librosa

""" 
low-order linear filters 
based on time-varying state-variable filters
currently supporting: 
    * lossy state-variable filters: low-, band-, and highpass, bandreject
    * generalized moog ladder filters: low-, band-, and highpass
    * lossless parametric eq components: high- and lowshelf, bell
    * Linkwitz-Riley crossover
    * phaser with 3-stage second-order svf-based allpass filter
"""


def hz_to_G(hz):
    return jnp.tan(np.pi * hz / FLAGS.sr)


def q_to_twoR(q):
    return 1 / q


def db_to_amp(db):
    return 10 ** (db / 20)


@partial(jax.jit, backend="cpu")
def ltisvfilt(x, G: float, twoR: float, c_hp: float, c_bp: float, c_lp: float):
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

    return lax.scan(ltisvfilt_step, (z, z), x)[1]


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


@partial(jax.jit, backend="cpu")
def tvsvfilt(x, G, twoR, c_hp, c_bp, c_lp):
    z = jnp.zeros(max(x.shape[-1], G.shape[-1], c_hp.shape[-1]))
    return lax.scan(tvsvfilt_step, (z, z), (x, G, twoR, c_hp, c_bp, c_lp))[1]


def apply_2nd_order_filter(input_signal, afx_type, gain_staging, frequency_hz=440, q=0.5, gain_db=0, c_hp=1, c_bp=1, c_lp=1):
    def get_cs(afx_type, twoR=None, gain=None):
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
    twoR = q_to_twoR(q)
    tv = False

    if "frequency_hz" in input_signal:
        frequency_hz = jnp.power(2, jnp.log2(frequency_hz) + input_signal["frequency_hz"])
        frequency_hz = jnp.minimum(frequency_hz, jnp.ones_like(frequency_hz) * FLAGS.sr / 2.5)
        tv = True
    if "gain_db" in input_signal:
        gain_db = gain_db + 12 * input_signal["gain_db"]
        tv = True

    G = hz_to_G(frequency_hz)
    gain = db_to_amp(gain_db)

    if tv:
        ones = jnp.ones((len(x), 1))
        twoR = ones * twoR
        if afx_type != "svf":
            c_hp, c_bp, c_lp = get_cs(afx_type, twoR, gain)
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
            c_hp, c_bp, c_lp = get_cs(afx_type, twoR, gain)
        else:
            c_hp, c_bp, c_lp = jnp.array(c_hp), jnp.array(c_bp), jnp.array(c_lp)
        y = ltisvfilt(x, G, twoR, c_hp, c_bp, c_lp)
    if gain_staging: y = gain_stage(x, y)
    return {"main": y}


@partial(jax.jit, backend="cpu")
def modulated_crossover(x, frequency_hz):
    G = hz_to_G(frequency_hz)
    one, zero = jnp.ones_like(G), jnp.zeros_like(G)
    twoR = np.sqrt(2) * one
    lpfed = tvsvfilt(x, G, twoR, zero, zero, one)
    lpfed = tvsvfilt(lpfed, G, twoR, zero, zero, one)
    hpfed = tvsvfilt(x, G, twoR, one, zero, zero)
    hpfed = tvsvfilt(hpfed, G, twoR, one, zero, zero)
    return {"low": lpfed, "high": hpfed}


@partial(jax.jit, backend="cpu")
def crossover(x, frequency_hz):
    G = hz_to_G(frequency_hz)
    lpfed = ltisvfilt(x, G, np.sqrt(2), 0, 0, 1)
    lpfed = ltisvfilt(lpfed, G, np.sqrt(2), 0, 0, 1)
    hpfed = ltisvfilt(x, G, np.sqrt(2), 1, 0, 0)
    hpfed = ltisvfilt(hpfed, G, np.sqrt(2), 1, 0, 0)
    return {"low": lpfed, "high": hpfed}


def apply_crossover(input_signal, gain_staging, frequency_hz=440):
    x = get_signal(input_signal, "main")
    if "frequency_hz" in input_signal:
        frequency_hz = jnp.power(2, jnp.log2(frequency_hz) + input_signal["frequency_hz"])
        frequency_hz = jnp.minimum(frequency_hz, jnp.ones_like(frequency_hz) * FLAGS.sr / 2.5)
        return modulated_crossover(x, frequency_hz)
    return crossover(x, frequency_hz)


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

@partial(jax.jit, backend="cpu")
def lowpass_ladder(x, G, k):
    z = jnp.zeros(x.shape[-1])
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
    z = jnp.zeros(x.shape[-1])
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
    z = jnp.zeros(x.shape[-1])
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

def apply_ladder(input_signal, afx_type, gain_staging, frequency_hz=440, k=0, q=None):
    x = get_signal(input_signal, "main")
    if "frequency_hz" in input_signal:
        frequency_hz = jnp.power(2, jnp.log2(frequency_hz) + input_signal["frequency_hz"])
        frequency_hz = jnp.minimum(frequency_hz, jnp.ones_like(frequency_hz) * FLAGS.sr / 2.5)
    G = hz_to_G(frequency_hz)
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

def butter(x, butter_type="lowpass", frequency_hz=4000, num_sos=2, bandwidth=2, zero_phase=False):
    num_sos = num_sos/2 if zero_phase else num_sos
    if butter_type == "bandpass":
        frequency_hz = [frequency_hz/np.sqrt(bandwidth), min(frequency_hz*np.sqrt(bandwidth), FLAGS.sr*0.45)]
    sos = signal.butter(num_sos, frequency_hz, btype=butter_type, fs=FLAGS.sr, output='sos')
    if zero_phase:
        return signal.sosfilt(sos, np.array(x).T).astype(np.float32).T
    else:
        return signal.sosfiltfilt(sos, np.array(x).T).astype(np.float32).T
    
def downsample(x, frequency_hz=16000, res_type="poly"):
    x = np.array(x)
    x_len = len(x)
    x = librosa.resample(np.array(x).T, orig_sr=FLAGS.sr, target_sr=frequency_hz, res_type=res_type)
    x = librosa.resample(x, orig_sr=frequency_hz, target_sr=FLAGS.sr, res_type=res_type)
    if x.shape[-1] < x_len:
        x = np.pad(x, (0, x_len-x.shape[-1]))
    else:
        x = x[:, :x_len]
    return x.T

def apply_high_order_filter(input_signal, afx_type, gain_staging, **kwargs):
    x = get_signal(input_signal, "main")
    if "butter" in afx_type:
        y = butter(x, afx_type.split("_")[0], **kwargs)
    elif afx_type == "downsample":
        y = gain_stage(x, downsample(x, **kwargs))
    else:
        assert False
    if gain_staging: y = gain_stage(x, y)
    return {"main": y}

#def apply_downsample(input_signal, **kwargs):
#    x = get_signal(input_signal, "main")
#    y = downsample(x, afx_type, **kwargs)
#    if gain_staging: y = gain_stage(x, y)
#    return {"main": y}
