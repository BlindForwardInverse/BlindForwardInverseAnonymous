""" 
jafx: afx library with jax
only render for differentiable processors.

"""
from .controllers import apply_controller
from .delay_and_reverb import apply_delay_and_reverb
from .dynamic_range_controllers import apply_dynamic_range_controller
from .state_variable_filters import apply_2nd_order_filter
from .ladder_filters import apply_ladder
from .crossover import apply_crossover
from .higher_order_filters import apply_higher_order_filter
from .memoryless_nonlinearities import apply_memoryless_nonlinearity
from .modulation_effects import apply_modulation_effect
from .utilities import apply_utility
from .audio_codec import apply_codec

import jax
import jax.numpy as jnp

def apply_jafx(input_signal, afx_type, afx_class, gain_staging, param_dict):
    if afx_class == "controller":
        return apply_controller(input_signal, afx_type, gain_staging, **param_dict)
    elif afx_class == "2nd_order_filter":
        return apply_2nd_order_filter(input_signal, afx_type, gain_staging, **param_dict)
    elif afx_class == "ladder":
        return apply_ladder(input_signal, afx_type, gain_staging, **param_dict)
    elif afx_class == "higher_order_filter":
        return apply_higher_order_filter(input_signal, afx_type, gain_staging, **param_dict)
    elif afx_class == "crossover":
        return apply_crossover(input_signal, gain_staging, **param_dict)
    elif afx_class == "memoryless_nonlinearity":
        return apply_memoryless_nonlinearity(input_signal, afx_type, gain_staging, **param_dict)
    elif afx_class == "dynamic_range_controller":
        return apply_dynamic_range_controller(input_signal, afx_type, gain_staging, **param_dict)
    elif afx_class == "delay_and_reverb":
        return apply_delay_and_reverb(input_signal, afx_type, gain_staging, **param_dict)
    elif afx_class == "modulation_effect":
        return apply_modulation_effect(input_signal, afx_type, gain_staging, **param_dict)
    elif afx_class == "utility":
        return apply_utility(input_signal, afx_type, gain_staging, **param_dict)
    elif afx_class == "codec":
        return apply_codec(input_signal, afx_type, gain_staging, **param_dict)
    else:
        raise NotImplementedError("Not Supporting afx_class")
