""" 
jafx: afx library with jax
"""
from afx.controllers import apply_controller
from afx.delay_and_reverb import apply_delay_and_reverb
from afx.dynamic_range_controllers import apply_dynamic_range_controller
from afx.filters import apply_2nd_order_filter, apply_ladder, apply_crossover, apply_high_order_filter
from afx.memoryless_nonlinearities import apply_memoryless_nonlinearity
from afx.modulation_effects import apply_modulation_effect
from afx.phase_vocoders import apply_phase_vocoder
from afx.utilities import apply_utility
from afx.augment import apply_codec
import flags as FLAGS
import jax; jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp
import numpy as np

from jax.config import config

def enable_debug():
    config.update("jax_debug_nans", True)

def apply_jafx(input_signal, afx_type, param_dict, gain_staging):
    if "main" in input_signal:
        if not (input_signal["main"].ndim == 2 and input_signal["main"].shape[-1] < 3):
            # print(input_signal["main"].shape, afx_type)
            assert False
    jafx_class = FLAGS.afx_config[afx_type]["class"]
    if jafx_class == "controller":
        return apply_controller(input_signal, afx_type, gain_staging, **param_dict)
    elif jafx_class == "2nd_order_filter":
        return apply_2nd_order_filter(input_signal, afx_type, gain_staging, **param_dict)
    elif jafx_class == "ladder":
        return apply_ladder(input_signal, afx_type, gain_staging, **param_dict)
    elif jafx_class == "higher_order_filter":
        return apply_high_order_filter(input_signal, afx_type, gain_staging, **param_dict)
    elif jafx_class == "crossover":
        return apply_crossover(input_signal, gain_staging, **param_dict)
    elif jafx_class == "memoryless_nonlinearity":
        return apply_memoryless_nonlinearity(input_signal, afx_type, gain_staging, **param_dict)
    elif jafx_class == "dynamic_range_controller":
        return apply_dynamic_range_controller(input_signal, afx_type, gain_staging, **param_dict)
    elif jafx_class == "phase_vocoder":
        return apply_phase_vocoder(input_signal, afx_type, gain_staging, **param_dict)
    elif jafx_class == "delay_and_reverb":
        return apply_delay_and_reverb(input_signal, afx_type, gain_staging, **param_dict)
    elif jafx_class == "modulation_effect":
        return apply_modulation_effect(input_signal, afx_type, gain_staging, **param_dict)
    elif jafx_class == "utility":
        return apply_utility(input_signal, afx_type, gain_staging, **param_dict)
    elif jafx_class == "codec":
        return apply_codec(input_signal, afx_type, gain_staging, **param_dict)



if __name__ == "__main__":
    x = np.random.uniform(size=(96000, 2))
    y = apply_jafx({"main": x, "modulation": x}, "chorus", {})
    # print(y["main"].shape)
    #    x = np.random.uniform(size=(96000))
    #    y = apply_jafx({'main': x, 'modulation': x}, afx, {})
