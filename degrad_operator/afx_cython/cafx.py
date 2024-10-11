from functools import partial

import flags as FLAGS
import jax
import jax.numpy as jnp
import numpy as np
from afx.cafx_core import *
from afx.primitives import get_signal
from jax import lax


def modulated_lfo(lfo_frequency_hz=4, offset=0.0):
    phase_inc = 2 * np.pi * lfo_frequency_hz / FLAGS.sr
    phase = jnp.cumsum(phase_inc)
    return jnp.sin(phase + offset)


def lfo(lfo_frequency_hz=4.0, offset=0.0):
    phase = jnp.arange(FLAGS.signal_len) * 2 * np.pi * lfo_frequency_hz / FLAGS.sr
    return jnp.sin(phase + offset)


def apply_controller(input_signal, afx_type, **param_dict):
    if afx_type == "envfollower":
        x = np.ascontiguousarray(input_signal["main"][:, 0]).astype(np.float32)
        return {"main": envfollower(x, **param_dict)}
    elif afx_type == "lfo":
        return {"main": jax.jit(lfo, backend="cpu")(**param_dict)}


def apply_2nd_order_filter(input_signal, afx_type, frequency_hz=440, q=0.5, gain_db=0):
    def get_cs(afx_type, twoR=None, gain=None):
        if afx_type == "lowpass":
            return 0, 0, 1
        elif afx_type == "bandpass":
            return 0, twoR, 0
        elif afx_type == "highpass":
            return 1, 0, 0
        elif afx_type == "bandreject":
            return 1, 0, 1
        elif afx_type == "lowshelf":
            return 1, twoR * jnp.sqrt(gain), gain
        elif afx_type == "highshelf":
            return gain, twoR * jnp.sqrt(gain), 1
        elif afx_type == "bell":
            return 1, twoR * gain, 1

    twoR = q_to_twoR(q)
    tv = False
    if "frequency_hz" in input_signal:
        frequency_hz = frequency_hz ** (1 + input_signal["frequency_hz"])
        tv = True
    if "gain_db" in input_signal:
        gain_db = gain_db + 12 * inpuut_signal["gain_db"]
        tv = True

    G = hz_to_G(frequency_hz)
    gain = db_to_amp(gain_db)

    if tv:
        ones = jnp.ones_like(input_signal["main"])
        c_hp, c_bp, c_lp = get_cs(afx_type, twoR, gain)
        c_hp, c_bp, c_lp = c_hp * ones, c_bp * ones, c_lp * ones
        return {"main": tvsvfilt(input_signal["main"], G, twoR, c_hp, c_bp, c_lp)}
    else:
        c_hp, c_bp, c_lp = get_cs(afx_type, twoR, gain)
        return {"main": ltisvfilt(input_signal["main"], G, twoR, c_hp, c_bp, c_lp)}


@partial(jax.jit, backend="cpu")
def crossover(x, frequency_hz):
    G = hz_to_G(frequency_hz)
    lpfed = ltisvfilt(x, G, np.sqrt(2), 0, 0, 1)
    lpfed = ltisvfilt(lpfed, G, np.sqrt(2), 0, 0, 1)
    hpfed = ltisvfilt(x, G, np.sqrt(2), 1, 0, 0)
    hpfed = ltisvfilt(hpfed, G, np.sqrt(2), 1, 0, 0)
    return {"low": lpfed, "high": hpfed}


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


def apply_crossover(input_signal, frequency_hz=440):
    if "frequency_hz" in input_signal:
        frequency_hz = frequency_hz ** (1 + input_signal["frequency_hz"])
        return modulated_crossover(input_signal["main"], frequency_hz)
    return crossover(input_signal["main"], frequency_hz)


def apply_ladder(input_signal, afx_type, frequency_hz=440, k=0, q=None):
    if "frequency_hz" in input_signal:
        frequency_hz = frequency_hz ** (1 + input_signal["frequency_hz"])
    else:
        frequency_hz = frequency_hz * jnp.ones_like(input_signal["main"])
    G = hz_to_G(frequency_hz)
    if afx_type == "lowpass_ladder":
        return {"main": lowpass_ladder(input_signal["main"], G, k)}
    if afx_type == "highpass_ladder":
        return {"main": highpass_ladder(input_signal["main"], G, k)}
    if afx_type == "bandpass_ladder":
        return {"main": bandpass_ladder(input_signal["main"], G, k)}


@partial(jax.jit, backend="cpu")
def bitcrush(x, bit_depth=16):
    zero = x - lax.stop_gradient(x)
    scale_factor = 2**bit_depth
    y = x * scale_factor
    y = jnp.round(y)
    y = y / scale_factor
    return zero + lax.stop_gradient(y)


def apply_memoryless_nonlinearity(input_signal, afx_type, **param_dict):
    if afx_type == "distortion":
        return {"main": distortion(input_signal["main"], **param_dict)}
    elif afx_type == "bitcrush":
        return {"main": bitcrush(input_signal["main"], **param_dict)}


@partial(jax.jit, backend="cpu")
def limiter(x, threshold_db=-18, release_ms=100):
    x = compressor(x, x, -10, 4, 2, 200)
    x = compressor(x, x, threshold_db, 1000, 0.001, release_ms)
    return x


def apply_dynamic_range_controller(input_signal, afx_type, **param_dict):
    if afx_type == "compressor":
        sidechain_input = (
            input_signal["sidechain"]
            if "sidechain" in input_signal
            else input_signal["main"]
        )
        return {"main": compressor(input_signal["main"], sidechain_input, **param_dict)}
    elif afx_type in ["noisegate", "expander"]:
        sidechain_input = (
            input_signal["sidechain"]
            if "sidechain" in input_signal
            else input_signal["main"]
        )
        return {"main": noisegate(input_signal["main"], sidechain_input, **param_dict)}
    elif afx_type == "limiter":
        return {"main": limiter(input_signal["main"], **param_dict)}


def apply_modulation_effect(input_signal, afx_type, **param_dict):
    if afx_type == "phaser":
        return {
            "main": phaser(
                input_signal["main"], input_signal["modulation"], **param_dict
            )
        }
    elif afx_type == "chorus":
        return {
            "main": chorus(
                input_signal["main"], input_signal["modulation"], **param_dict
            )
        }


@partial(jax.jit, backend="cpu")
def pitchshift(x, semitones=0.0):
    _, _, X = jax.scipy.signal.stft(x, window="hann", nperseg=2048, noverlap=512)
    mag, phase = jnp.abs(X), jnp.angle(X)
    init_phase, phase_inc = phase[:, :1], jnp.unwrap(phase[:, 1:] - phase[:, :-1], -1)
    ratio = 2 ** (semitones / 12)
    phase_inc = ratio * phase_inc
    phase = jnp.cumsum(jnp.concatenate([init_phase, phase_inc], -1), -1)
    X_shifted = mag * jnp.exp(1j * phase)
    _, x_shifted = jax.scipy.signal.istft(
        X_shifted, window="hann", nperseg=2048, noverlap=512
    )
    x_shifted = x_shifted[: FLAGS.signal_len]
    return x_shifted


def apply_pitchshift(input_signal, **param_dict):
    return {"main": pitchshift(input_signal["main"], **param_dict)}
