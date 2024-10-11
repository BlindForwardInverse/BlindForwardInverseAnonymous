from functools import partial
from typing import Dict, List, Tuple, Union
from pprint import pprint

import os;opj=os.path.join

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .afx_jax.jafx import apply_jafx
from .afx_jax.jafx_utils import db_to_amp, rms_normalize

from .grafx import Grafx
from .jax_autograd import jax_to_autograd

from utils.audio_processing import make_it_mono, make_it_stereo, audio_processing
from einops import rearrange, repeat

class DifferentiableRenderGrafx:
    def __init__(
        self,
        sr=44100,
        backend='jax',
        gain_staging=True,
        batched_processing=False,
        mono_processing=True, 
        differentiable=True,
        output_format=torch.tensor,
        config_path='configs/degrad_operator',
    ):
        '''
        - If mono_processing : True, any stereo signals are converted into mono signals for processing,
        and vice versa.
        - RIR, MicIR, Noises signals are stored as parameters of each node
        - differenetiable : torch -> torch with autograd
        '''
        self.sr = sr
        self.backend = backend
        self.gain_staging = gain_staging
        self.batched_processing = batched_processing
        self.mono_processing = mono_processing
        self.differentiable = differentiable
        self.output_format = output_format

        # Afx parameter information
        self.afx_module_config = OmegaConf.load(opj(config_path, "afx_module_configs.yaml"))

    def __call__(self, G: Grafx, input_signals):
        """
        G (class Grafx): signal processing graph to render
        input_signal (dict or audio(torch.tensor or np.ndarray)): input_signals
        """
        if self.differentiable:
            rendered_audio = self.render_differentiable(G, input_signals)
        else:
            rendered_audio = self.render(G, input_signals)
            rendered_audio = self.jnp_to_signal(rendered_audio)
        return rendered_audio

    @jax_to_autograd
    def render_differentiable(self, G, dry_audio):
        return self._main_processing(G, self.dict_to_jnp(self.signal_to_dict(dry_audio)))

    def render(self, G, dry_audio):
        audio_signal = self.dict_to_jnp(self.signal_to_dict(dry_audio))
        return self._main_processing(G, audio_signal)

    def _main_processing(self, G, input_signals):
        """
        Now support for the single audio setting
        """
        graph_configs = self.get_graph_configs(G)
        signal_configs = self.get_signal_configs(input_signals)

        parameters, input_signal_idx, topo_order, outgoing_signals = \
            (graph_configs[key] for key in ["parameters", "input_signal_idx", "topo_order", "outgoing_signals"])
        batch_size, signal_shape, signal_len = \
            (signal_configs[key] for key in ['batch_size', 'signal_shape', 'signal_len'])

        # Processing stage
        for idx in topo_order:
            # rank 0 nodes (i.e. no predecessors)
            if idx in input_signal_idx:
                external_input = input_signals.get(idx, jnp.zeros(signal_shape))
                outgoing_signals[idx] = {"main": external_input}

            elif G.nodes[idx]["afx_type"] in ["rir", "micir"]:
                ir = parameters[idx]["ir"]
                if self.batched_processing : ir = repeat(ir, 't -> b t', b=batch_size)
                if self.backend == 'jax':
                    ir = jnp.array(ir)
                outgoing_signals[idx] = {"main" : ir}

            elif G.nodes[idx]["afx_type"] == "noise":
                noise = parameters[idx]["noise"]
                noise = audio_processing(noise,
                                         target_len=signal_len,
                                         mono=self.mono_processing,
                                         crop_mode="front",
                                         pad_type="repeat",
                                         rms_norm=True,)
                snr_db = parameters[idx]["snr"]
                snr = 10 ** (snr_db / 20)
                gain = (1 / snr)
                scaled_noise = gain * noise
                if self.batched_processing : scaled_noise = repeat(scaled_noise, 't -> b t', b=batch_size)
                if self.backend == 'jax':
                    scaled_noise = jnp.array(scaled_noise)
                outgoing_signals[idx] = {"main" : scaled_noise}
                
            else:
                if G.nodes[idx]["afx_type"] in ['lfo', 'stereo_lfo', 'lowpass_noise']:
                    incoming_signals = None
                else:
                    # After the topological sort, all predecessors have outgoing_signals
                    incoming_signals = self._collect_incoming_signals(G, idx, outgoing_signals)

                processed_signal = self._processor(incoming_signals, idx, graph_configs, signal_configs)
                outgoing_signals[idx] = processed_signal

        rendered_audio = outgoing_signals["out"]["main"]
        return rendered_audio

    def _collect_incoming_signals(self, G: Grafx, idx, outgoing_signals):
        """
        [pred] outlet ---(gain)--- inlet [idx]
        [pred2] outlet ----(gain)----|
        """
        def sum_signals(x, y):
            if self.backend == 'jax':
                return jnp.add(x, y)

        incoming_signals = {}  # ex) {main=..., modulation=...}
        for edge in G.in_edges(nbunch=idx, data=True):
            predecessor, _, config = edge  # config = {outlet=..., inlet=..., gain=...}
            outlet, inlet, gain = [
                config.get(key) for key in ["outlet", "inlet", "gain"]
            ]
            # Load outgoing signals coming out from the predecessor
            if self.backend == 'jax':
                gain_amp = db_to_amp(gain)
            outgoing_signal = (
                gain_amp * outgoing_signals[predecessor][outlet]
            )

            if inlet in incoming_signals.keys():
                incoming_signals[inlet] = sum_signals(
                    incoming_signals[inlet], outgoing_signal
                )
            else:
                incoming_signals[inlet] = outgoing_signal
        return incoming_signals

    def _processor(self, input_signal, afx, graph_configs, signal_configs):
        afx_type = graph_configs['afx_types'][afx]
        afx_class = self.afx_module_config[afx_type]["class"]
        gain_staging = self.gain_staging
        configs = {key : signal_configs[key] for key in ['sr', 'mono', 'signal_len']}
        if afx_class == "output":
            return input_signal
        else:
            params = graph_configs['parameters'][afx]
            param_dict = {**params, **configs}
            jafx = partial(
                apply_jafx,
                afx_type=afx_type,
                afx_class=afx_class,
                gain_staging=gain_staging,
                param_dict=param_dict,
            )
            if self.batched_processing:  # Batched processing
                if self.backend == 'jax':
                    return jax.vmap(jafx, in_axes=0)(input_signal)
            else:
                return jafx(input_signal)

# Utility Methods
# -----------------------------------------------------------------------------------------------------------
    def dict_to_jnp(self, input_signals):
        # Make jnp array audio
        for signal_name, audio in input_signals.items():
            if not isinstance(audio, jnp.ndarray):
                input_signals[signal_name] = rms_normalize(jnp.array(audio))
        return input_signals

    def dict_to_torch(self, input_signals):
        for signal_name, audio in input_signals.items():
            if not isinstance(audio, torch.Tensor):
                input_signals[signal_name] = rms_normalize_torch(torch.tensor(audio))
        return input_signals

    def jnp_to_signal(self, output_audio):
        output_signal = np.array(output_audio)
        output_signal = self.output_format(output_signal)
        return output_signal

    @staticmethod
    def signal_to_dict(input_signals):
        if not isinstance(input_signals, dict):
            input_signals = {'speech' : input_signals}
        return input_signals

    @staticmethod
    def get_graph_configs(G: Grafx):
        parameters = G.parameters
        afx_types = G.afx_types
        input_signal_idx = G.input_signal_idx
        topo_order = list(nx.topological_sort(G))
        outgoing_signals = {node: None for node in G.nodes()}
        graph_configs = {
            'parameters' : parameters,
            'afx_types' : afx_types,
            'input_signal_idx' : input_signal_idx,
            'topo_order' : topo_order,
            'outgoing_signals' : outgoing_signals
        }
        return graph_configs

    def get_signal_configs(self, input_signals):
        """
        input_signals = dict()
        audio must be np.ndarray or torch.tensor
        """
        # 1) Check if the signal shapes are consistent.
        # -------------------------------------------------------------------------------------------
        signal_shapes = [audio.shape for name, audio in input_signals.items()]
        assert all(shape == signal_shapes[0] for shape in signal_shapes), f"Inconsistent Audio Shape"
        signal_shape = signal_shapes[0]


        # 2) Signal config (batch_size, signal_len, mono/stereo input)
        # -------------------------------------------------------------------------------------------
        if self.batched_processing:
            assert len(signal_shape) in [2, 3], f"The renderer is set to batched processing"
            batch_size, signal_len = signal_shape[0], signal_shape[1] # (B, T) or (B, T, 2)
            mono = True if len(signal_shape) == 2 else False
        else:
            assert len(signal_shape) in [1, 2], f"The renderer gets a single audio"
            batch_size, signal_len = None, signal_shape[0]  # (T) or (T, 2)
            mono = True if len(signal_shape) == 1 else False
        assert mono == self.mono_processing, f"The renderer is set to {self.mono_processing} mode"

        signal_config = {
            'sr' : self.sr,
            'signal_len' : signal_len,
            'signal_shape': signal_shape,
            'mono' : mono,
            'batch_size' : batch_size,
            'signal_shpe' : signal_shape,
        }
        if self.backend == 'torch':
            devices = [audio.device for name, audio in input_signals.items()]
            assert all(device == devices[0] for device in devices), f"Inconsistent Devices"
            signal_config['device'] = devices[0]

        return signal_config
