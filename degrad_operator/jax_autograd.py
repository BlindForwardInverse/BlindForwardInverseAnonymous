import torch
import jax
from jax import jit, pmap
import jax.numpy as jnp
import numpy as np
from functools import partial
from time import time

def jax_to_autograd(processors):
    class JaxFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, _self, _G, input):
            # Move input to CPU and convert PyTorch tensor to JAX array
            input_cpu = input.detach().cpu()
            input_jax = jnp.array(input_cpu.numpy())

            # Compute the forward pass using JAX on CPU
            func = partial(processors, _self, _G)
#             output_jax = processors(_self, _G, input_jax)
            output_jax = func(input_jax)

            # Convert JAX output back to PyTorch tensor and move to the original device
            output = torch.tensor(np.array(output_jax), dtype=input.dtype).pin_memory().to(input.device)

            # Save input and device for backward pass
            ctx.save_for_backward(input)
            ctx._self = _self
            ctx._G = _G
            ctx.device = input.device

            return output

        @staticmethod
        def backward(ctx, grad_output):
            # Retrieve saved input tensor and device
            input, = ctx.saved_tensors
            _self, _G = ctx._self, ctx._G
            device = ctx.device

            # Move input to CPU and convert PyTorch tensor to JAX array
            input_cpu = input.detach().cpu()
            input_jax = jnp.array(input_cpu.numpy())

            jax_grad_output = jnp.array(grad_output.detach().cpu().numpy())
            func = partial(processors, _self, _G)

            primals, tangents = jax.jvp(func, (input_jax,), (jax_grad_output,))
            torch_grad = torch.tensor(np.array(tangents), dtype=input.dtype).pin_memory().to(device)
            return None, None, torch_grad

    def wrapped(self, G, input_signals):
        return JaxFunction.apply(self, G, input_signals)
    return wrapped

