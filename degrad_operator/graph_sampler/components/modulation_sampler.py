import numpy as np
import random
from omegaconf import OmegaConf
from .sample_functions import get_random_values, true_or_false
import os; opj = os.path.join
from functools import partial

class ModulationSampler():
    def __init__(self,
                 lfo_prob=0,
                 mono_processing=True,
                 config_path='configs/degrad_operator'):
        self.lfo_prob = lfo_prob
        self.mono_processing = mono_processing

        self.modulation_config = OmegaConf.load(opj(config_path, 'modulation_sampler.yaml'))
        self.afx_config =        OmegaConf.load(opj(config_path, 'afx_module_configs.yaml'))

    def __call__(self, G, mono=True, *args, **kwargs):
        nodes = [node for node in G.nodes(data=True)]
        for idx, data in nodes:
            afx_type, chain, structure_node= data['afx_type'], data['chain'], data['structure_node']
            afx_class = self.afx_config[afx_type]['class']

            if afx_type in self.modulation_config.keys():
                if afx_class == 'modulation_effect':
                    self.attach_controller(G, idx, afx_type, chain, structure_node)
                else:
                    attach_controller = true_or_false(p=self.lfo_prob)
                    if attach_controller: self.attach_controller(G, idx, afx_type, chain, structure_node)

    def attach_controller(self, G, idx, afx_type, chain, structure_node):
        # Load configs for modulation parameter
        config = self.modulation_config[afx_type]
        param_config = config['parameters']
        modulation_type = config['modulation_type']
        inlet = config['inlet']
        
        # Get params
        param_dict = dict()
        for param in param_config:
            dist = param_config[param]['distribution']
            sampling_range = list(param_config[param]['sampling_range'])
            value = get_random_values(dist, *sampling_range)
            param_dict[param] = value

        # Add
        add = partial(G.add, chain=chain, structure_node=structure_node)
        if modulation_type == 'lowpassed_noise':
            modulation = add("lowpass_noise", **param_dict)
        elif modulation_type == 'lfo':
            modulation = add("lfo" if self.mono_processing else "stereo_lfo",
                             **param_dict)

        # Connect
        G.connect(modulation, idx, inlet=inlet, gain=0)
