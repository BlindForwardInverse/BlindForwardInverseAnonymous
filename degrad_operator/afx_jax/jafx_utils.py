import math
from functools import partial

import jax
import jax.numpy as jnp

def get_signal(signal_dict, key):
    # print(f"{key} : {signal_dict[key].shape}")
    return signal_dict[key]

def calculate_rms(x):
    return jnp.sqrt(jnp.mean(x**2) + 1e-7)


def rms_normalize(x, ref_dBFS=-16.0, return_gain=False):
    rms = calculate_rms(x)
    ref_linear = jnp.power(10, (ref_dBFS - 3.0103) / 20.0)
    gain = ref_linear / (rms + 1e-7)
    x = gain * x
    if return_gain:
        return x, gain
    else:
        return x

@partial(jax.jit, backend="cpu")
def gain_stage(x, y):
    x_rms, y_rms = calculate_rms(x), calculate_rms(y)
    val = y * x_rms / y_rms
    return val

# Functions for state_variable_filters
@partial(jax.jit, backend="cpu")
def db_to_amp(gain):
    return jnp.exp(gain / 20 * jnp.log(10))


@partial(jax.jit, backend="cpu", static_argnames="sr")
def hz_to_G(hz, sr):
    return jnp.tan(jnp.pi * hz / sr)


@partial(jax.jit, backend="cpu")
def q_to_twoR(q):
    return 1 / q
