import torch
import numpy as np
from functools import partial
from omegaconf import OmegaConf
from scipy.stats import loguniform

from degrad_operator.grafx import Grafx
from .sample_functions import choice, choices, true_or_false

class ChainSampler():
    def __init__(self,
                 mono_processing=True,
                 chain_len=None,
                 min_chain_len=1,
                 max_chain_len=3,
                 chain_randomization = 'full'):
        self.chain_sampler_config = OmegaConf.load('configs/degrad_operator/chain_sampler.yaml')
        self.afx_sampler_config = OmegaConf.load('configs/degrad_operator/afx_sampler.yaml')
        self.afx_module_config = OmegaConf.load('configs/degrad_operator/afx_module_configs.yaml')
        assert chain_randomization in ['full', 'fixed_order', 'param_only']

        self.mono_processing = mono_processing
        self.chain_len = chain_len
        self.min_chain_len = min_chain_len
        self.max_chain_len = max_chain_len
        self.chain_randomization = chain_randomization # TODO

    def __call__(self,
                 G: Grafx,
                 last_node=None,
                 chain_type='default',
                 structure_node=None,
                 single_afx_name=None, # Set this None for a random afx
                 *args, **kwargs):

        # Sample sequential chain of afx
        if single_afx_name is None:
            perceptual_attributes = self.sample_attributes(chain_type)
            afx_chain = self.sample_afx_chain(perceptual_attributes, self.mono_processing)
        else:
            afx_chain = [single_afx_name,]

        # Add afx to G
        for afx in afx_chain:
            if afx in self.afx_module_config.keys():
                last_node = G.add_and_connect(afx, last_node, chain=chain_type, structure_node=structure_node)
            else:
                last_node = add_predefined_chain(G, last_node, afx, chain_type, structure_node=structure_node)
        return last_node

    def sample_attributes(self, chain_type):
        # Sample the sequential list of the perceptual attributes
        chain_config = self.chain_sampler_config[chain_type]
        if self.chain_len is None:
            chain_len = choice(list(chain_config['chain_len']['keys']),
                               list(chain_config['chain_len']['weights']))
            if chain_len < self.min_chain_len: chain_len = self.min_chain_len
            if chain_len > self.max_chain_len: chain_len = self.max_chain_len
        else:
            chain_len = self.chain_len
        perceptual_attributes = choices(chain_config['perceptual_attribute']['keys'],
                                chain_config['perceptual_attribute']['weights'],
                                size = chain_len,
                                replace=False) # True
        return perceptual_attributes

    def sample_afx_chain(self, perceptual_attributes, mono=False):
        # Sample the sequential list of afx
        afx_chain = []
        for attribute in perceptual_attributes:
            afx_config = self.afx_sampler_config[attribute]
            afx = choice(list(afx_config['keys']),
                         list(afx_config['weights']))
            if afx == "reverb" and mono:
                afx = "mono_reverb"
            elif afx == "reverb" and not mono:
                afx = "chorus"
            afx_chain.append(afx)
        return afx_chain

    def get_supporting_afx(self, chain_type):
        chain_config = self.chain_sampler_config[chain_type]
        supporting_afx = []
        for perceptual_attributes in list(chain_config['perceptual_attribute']['keys']):
            for afx in list(self.afx_sampler_config[perceptual_attributes]['keys']):
                supporting_afx.append(afx)

        return supporting_afx

    def change_single_afx_name(self, updated_afx):
        self.single_afx_name = updated_afx

def add_predefined_chain(G: Grafx, last_node, afx, chain, structure_node):
    add_and_connect = partial(G.add_and_connect, chain=chain, structure_node=structure_node)
    add = partial(G.add, chain=chain, structure_node=structure_node)

    if afx in ['plosive', 'deesser']:
        crossover = add_and_connect('crossover', i_from=last_node)
        comp_type = 'inverted_compressor' if afx =='plosive' else 'compressor'
        compressor = add_and_connect(comp_type, i_from=crossover, outlet='high')
        mix = add('mix')
        G.connect(crossover, mix, outlet='low')
        G.connect(compressor, mix)
        last_node = mix

    elif afx == 'parametric_eq':
        num_eq = np.random.randint(6)
        eqs = choices(['lowshelf', 'bell', 'highshelf'], size=num_eq, replace=True)
        for eq in eqs:
            last_node = add_and_connect(eq, i_from=last_node, q=loguniform.rvs(0.5, 4))

    elif afx == 'rir_conv':
        conv = add_and_connect('convolution', i_from=last_node)
        rir = add('rir', name=f'rir_{G.num_rir}')
        G.connect(rir, conv, inlet='ir')

        G.num_rir += 1
        last_node = conv

    elif afx == 'micir_conv':
        conv = add_and_connect('convolution', i_from=last_node)
        micir = add('micir', name=f'micir_{G.num_micir}')
        G.connect(micir, conv, inlet='ir')

        G.num_micir += 1
        last_node = conv
        
    elif afx == "add_noise":
        if G.num_noise < 4:
            mix = add_and_connect('mix', i_from=last_node)
            noise = add('noise', name=f'noise_{G.num_noise}')
            G.connect(noise, mix, inlet="main")
            
            G.num_noise += 1
            last_node = mix
    else:
        raise Exception(f"{afx} is not defined")
    return last_node
