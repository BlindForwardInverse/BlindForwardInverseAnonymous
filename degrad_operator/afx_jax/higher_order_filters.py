from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from .jafx_utils import gain_stage, get_signal
from jax import lax
from scipy import signal

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


def apply_higher_order_filter(
    input_signal, afx_type, gain_staging, sr, signal_len, mono, **kwargs
):
    x = get_signal(input_signal, "main")
    if "butter" in afx_type:
        y = butter(x, afx_type.split("_")[0], sr=sr, **kwargs)
    else:
        assert False
    if gain_staging:
        y = gain_stage(x, y)
    return {"main": y}

# def downsample(x, original_sr, target_sr):
#    if original_sr == target_sr:
#         return audio

#     assert original_sr % target_sr == 0, "Target sampling rate must be a divisor of original sampling rate"

#     ratio = original_sr // target_sr
#     downsampled_audio = audio[:, ::ratio]
#     return downsampled_audio

def butter(
    x,
    butter_type="lowpass",
    frequency_hz=4000,
    num_sos=2,
    bandwidth=2,
    zero_phase=False,
    sr=44100,
    **param_dict
):
    num_sos = num_sos / 2 if zero_phase else num_sos
    if butter_type == "bandpass":
        frequency_hz = [
            frequency_hz / np.sqrt(bandwidth),
            min(frequency_hz * np.sqrt(bandwidth), sr * 0.45),
        ]
    sos = signal.butter(num_sos, frequency_hz, btype=butter_type, fs=sr, output="sos")
    if zero_phase:
        return sosfilt(sos, x)

    else:
        return sosfiltfit(sos, x)


@partial(jax.jit, backend="cpu")
def sosfilt_section(b, a, x, zi):
    init_carry = (zi, b, a)

    def tdf2_step(carry, x_curr):
        zi, b, a = carry
        y_curr = b[0] * x_curr + zi[0]
        z0_temp = b[1] * x_curr + zi[1] - a[1] * y_curr
        new_zi = jnp.array([z0_temp, b[2] * x_curr - a[2] * y_curr])
        return (new_zi, b, a), y_curr

    zi_final, y = lax.scan(tdf2_step, init_carry, x)
    return y, zi_final


def sosfilt(sos, x):
    """Process data x with a second-order sections filter defined by sos."""
    n_sections = sos.shape[0]
    for section in range(n_sections):
        b = sos[section, :3]
        a = sos[section, 3:]
        x, _ = sosfilt_section(b, a, x, jnp.zeros(2))
    return
