import flags as FLAGS
import jax.numpy as jnp
import soundfile as sf
import numpy as np


def get_signal(signal_dict, key, c=1):
    return signal_dict[key] if key in signal_dict else jnp.zeros((FLAGS.signal_len, c))

def calculate_rms(x):
    x = jnp.clip(x, a_min=-10, a_max=10)
#     x_abs = jnp.max(jnp.abs(x))
#     if x_abs > 10: 
#         pass
    return jnp.sqrt(jnp.mean(x**2) + 1e-7)

def rms_normalize(x, ref_dBFS=-23.0, return_gain=False):
    rms = calculate_rms(x)
    ref_linear = jnp.power(10, (ref_dBFS - 3.0103) / 20.0)
    gain = ref_linear / (rms + 1e-7)
    x = gain * x
    if return_gain:
        return x, gain
    else:
        return x


def gain_stage(x, y):
    x_rms, y_rms = calculate_rms(x), calculate_rms(y)
    return y * x_rms / y_rms
