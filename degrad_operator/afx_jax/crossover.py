import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from functools import partial
from .state_variable_filters import tvsvfilt, ltisvfilt
from .jafx_utils import get_signal, gain_stage, hz_to_G

def apply_crossover(input_signal, gain_staging, frequency_hz=440, sr=44100, **param_dict):
    x = get_signal(input_signal, "main")
    if "frequency_hz" in input_signal:
        frequency_hz = jnp.power(2, jnp.log2(frequency_hz) + input_signal["frequency_hz"])
        frequency_hz = jnp.minimum(frequency_hz, jnp.ones_like(frequency_hz) * sr / 2.5)
        return modulated_crossover(x, frequency_hz, sr)
    else:
        val = crossover(x, frequency_hz, sr)
        return val

@partial(jax.jit, backend="cpu", static_argnames="sr")
def modulated_crossover(x, frequency_hz, sr):
    original_shape = x.shape
    
    if x.ndim == 1: #?
        x = x.reshape(-1, 1) #?
        
    G = hz_to_G(frequency_hz, sr)
    one, zero = jnp.ones_like(G), jnp.zeros_like(G)
    twoR = jnp.sqrt(2) * one
    lpfed = tvsvfilt(x, G, twoR, zero, zero, one)
    lpfed = tvsvfilt(lpfed, G, twoR, zero, zero, one)
    hpfed = tvsvfilt(x, G, twoR, one, zero, zero)
    hpfed = tvsvfilt(hpfed, G, twoR, one, zero, zero)
    
    if len(original_shape) == 1:
        lpfed = jnp.mean(lpfed, axis=1)
        hpfed = jnp.mean(hpfed, axis=1)
    
    return {"low": lpfed, "high": hpfed}


@partial(jax.jit, backend="cpu", static_argnames="sr")
def crossover(x, frequency_hz, sr):
    G = hz_to_G(frequency_hz, sr)
    lpfed = ltisvfilt(x, G, jnp.sqrt(2), 0, 0, 1)
    lpfed = ltisvfilt(lpfed, G, jnp.sqrt(2), 0, 0, 1)
    hpfed = ltisvfilt(x, G, jnp.sqrt(2), 1, 0, 0)
    hpfed = ltisvfilt(hpfed, G, jnp.sqrt(2), 1, 0, 0)
    
    return {"low": lpfed, "high": hpfed}




