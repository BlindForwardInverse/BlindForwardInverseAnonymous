from functools import partial
from typing import Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from jax import vmap
from omegaconf import OmegaConf

from .afx_jax.jafx import apply_jafx
from .afx_jax.jafx_utils import db_to_amp, rms_normalize
from .grafx import Grafx

from utils.audio_processing import make_it_mono, make_it_stereo, audio_processing

class RenderGrafx:
    def __init__(
        self,
        sr=44100,
        backend="jax",
        gain_staging=True,
        batched_processing=False,
        mono_processing=True, 
        output_format=torch.tensor,
    ):
        '''
        - If mono_processing : True, any stereo signals are converted into mono signals for processing,
        and vice versa.
        - RIR, MicIR, Noises signals are stored as parameters of each node
        '''
        self.sr = sr
        self.backend = backend
        self.gain_staging = gain_staging
        self.batched_processing = batched_processing
        self.mono_processing = mono_processing
        self.output_format = output_format

        # Afx parameter information
        self.afx_module_config = OmegaConf.load("configs/degrad_operator/afx_module_configs.yaml")

    def __call__(self, G: Grafx, input_signals):
        """
        G (class Grafx): signal processing graph to render
        input_signal (dict): input_signals

        Render the input signal w.r.t. the given signal processing graph.
        It processes the inputs one by one according to the topological order.
        """

        # Initialize Stage
        self.initialize_renderer(G)
        input_signals = self.preprocess_input_signals(input_signals)

        # Processing stage
        for idx in self.topo_order:
            # rank 0 nodes (i.e. no predecessors)
            if idx in self.input_signal_idx:
                external_input = input_signals.get(idx, jnp.zeros(self.signal_shape))
                self.outgoing_signals[idx] = {"main": external_input}

            elif G.nodes[idx]["afx_type"] in ["rir", "micir"]:
                ir = self.parameters[idx]["ir"]
                self.outgoing_signals[idx] = {"main" : ir}

            elif G.nodes[idx]["afx_type"] == "noise":
                noise = self.parameters[idx]["noise"]
                noise = audio_processing(noise,
                                         target_len=self.signal_len,
                                         mono=self.mono_processing,
                                         crop_mode="front",
                                         pad_type="repeat",
                                         rms_norm=True,)
                snr_db = self.parameters[idx]["snr"]
                snr = 10 ** (snr_db / 20)
                gain = (1 / snr)
                scaled_noise = gain * noise
                self.outgoing_signals[idx] = {"main" : scaled_noise}
                
            else:
                if G.nodes[idx]["afx_type"] in ['lfo', 'stereo_lfo', 'lowpass_noise']:
                    incoming_signals = None
                else:
                    # After the topological sort, all predecessors have outgoing_signals
                    incoming_signals = self.collect_incoming_signals(G, idx)
                processed_signal = self.render_audio_processor(incoming_signals, idx)
                self.outgoing_signals[idx] = processed_signal
            
        rendered_audio = self.finalize_renderer()
        return rendered_audio

    def initialize_renderer(self, G: Grafx):
        self.parameters = G.parameters
        self.afx_types = G.afx_types
        self.input_signal_idx = G.input_signal_idx
        self.topo_order = list(nx.topological_sort(G))
        self.outgoing_signals = {node: None for node in G.nodes()}

    def preprocess_input_signals(self, input_signals):
        # Force each signal to have the same shape.
        # -------------------------------------------------------------------------------------------
        signal_shape = None
        for signal_name, audio in input_signals.items():
            if signal_shape is None:
                signal_shape = audio.shape  # initialization
            else: 
                assert (
                    audio.shape == signal_shape
                ), f"Inconsistent Audio shape ({signal_shape} of {signal_name} != {self.signal_shape})"
        self.signal_shape = signal_shape

        # Signal config (batch_size, signal_len, mono/stereo input)
        # -------------------------------------------------------------------------------------------
        if self.batched_processing:
            batch_size, signal_len = signal_shape[0], signal_shape[1] # (B, T) or (B, T, 2)
            mono_input = True if len(signal_shape) == 2 else False
                
        else:
            batch_size, signal_len = None, signal_shape[0]  # (T) or (T, 2)
            mono_input = True if len(signal_shape) == 1 else False

        self.batch_size, self.signal_len, self.mono_input = batch_size, signal_len, mono_input

        # Input signals must be converted to the mono/stereo mode of the renderer.
        # -------------------------------------------------------------------------------------------
        if mono_input and not self.mono_processing:
            for signal_name, audio in input_signals.items():
                input_signals[signal_name] = make_it_stereo(audio)

        if not mono_input and mono_processing:
            for signal_name, audio in input_signals.items():
                input_signals[signal_name] = make_it_mono(audio)

        # Make jnp array audio
        # -------------------------------------------------------------------------------------------
        for signal_name, audio in input_signals.items():
            if self.backend == "jax" and not isinstance(audio, jnp.ndarray):
                input_signals[signal_name] = rms_normalize(jnp.array(audio))

        self.signal_config = {
            "sr": self.sr,
            "signal_len": self.signal_len,
            "mono": self.mono_processing
        }
        return input_signals

    def collect_incoming_signals(self, G: Grafx, idx):
        """
        [pred] outlet ---(gain)--- inlet [idx]
        [pred2] outlet ----(gain)----|
        """

        def sum_signals(x, y):
            if self.backend == "jax":
                return jnp.add(x, y)
            else:
                return np.add(x, y)

        incoming_signals = {}  # ex) {main=..., modulation=...}
        for edge in G.in_edges(nbunch=idx, data=True):
            predecessor, _, config = edge  # config = {outlet=..., inlet=..., gain=...}
            outlet, inlet, gain = [
                config.get(key) for key in ["outlet", "inlet", "gain"]
            ]
            # Load outgoing signals coming out from the predecessor
            outgoing_signal = (
                db_to_amp(gain) * self.outgoing_signals[predecessor][outlet]
            )
            if inlet in incoming_signals.keys():
                incoming_signals[inlet] = sum_signals(incoming_signals[inlet], outgoing_signal)
            else:
                incoming_signals[inlet] = outgoing_signal

        for inlet in incoming_signals:
            if inlet != 'modulation':
                incoming_signals[inlet] = rms_normalize(incoming_signals[inlet])

        return incoming_signals

    def render_audio_processor(self, input_signal, afx):
        afx_type = self.afx_types[afx]
        afx_class = self.afx_module_config[afx_type]["class"]
        if afx_class == "output":
            return input_signal
        else:
            gain_staging = self.gain_staging
            params = self.parameters[afx]
            param_dict = {**params, **self.signal_config}
            jafx = partial(
                apply_jafx,
                afx_type=afx_type,
                afx_class=afx_class,
                gain_staging=gain_staging,
                param_dict=param_dict,
            )
            if self.batched_processing:  # Batched processing
                return vmap(jafx, in_axes=0)(input_signal)
            else:
                return jafx(input_signal)

    def finalize_renderer(self):
        output_signal = np.array(self.outgoing_signals["out"]["main"])
        output_signal = self.output_format(output_signal)
        return output_signal
