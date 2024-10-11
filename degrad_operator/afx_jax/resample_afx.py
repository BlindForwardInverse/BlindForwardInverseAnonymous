import jax.numpy as jnp
import jax
from jax import grad, jit, lax, jacrev
from functools import partial

def differentiable_resample(audio_signal, original_rate, target_rate):
    """
    Differentiable audio downsampling using simple linear interpolation.
    """
    ratio = (original_rate / target_rate)
    indices = jnp.arange(0, len(audio_signal), ratio)

    # Linear interpolation using lax
    lower_indices = jnp.floor(indices).astype(int)
    upper_indices = jnp.clip(jnp.ceil(indices).astype(int), 0, len(audio_signal) - 1)

    def interpolate(i):
        alpha = indices[i] - lower_indices[i]
        lower_val = lax.dynamic_index_in_dim(audio_signal, lower_indices[i], axis=0, keepdims=False)
        upper_val = lax.dynamic_index_in_dim(audio_signal, upper_indices[i], axis=0, keepdims=False)
        return (1 - alpha) * lower_val + alpha * upper_val

    return jnp.vectorize(interpolate)(jnp.arange(len(indices)))

# Create a toy audio signal for testing
audio_signal = jnp.sin(jnp.linspace(0, 2*jnp.pi*10, 44100))
print(audio_signal.shape)
original_rate = 44100
target_rate = 16000

resampled_audio = differentiable_resample(audio_signal, original_rate, target_rate)
print(resampled_audio.shape)

# Check the gradient with respect to the audio signal
resample_grad_fn = jacrev(differentiable_resample, argnums=0)
resample_gradient = resample_grad_fn(audio_signal, original_rate, target_rate)
print(resample_gradient)

