from time import time
from pprint import pprint

from degrad_operator.grafx import Grafx
from .components.chain_sampler import ChainSampler
from .components.modulation_sampler import ModulationSampler
from .components.parameter_sampler import ParameterSampler
from .components.structure_sampler import StructureSampler

class GraphSampler:
    def __init__(
        self,
        # Structure Params
        fork_prob=0.7,
        merge_prob=0.7,
        min_node_num=1,
        max_node_num=6,
        force_merge_num=6,
        default_chain_type='full_afx',
        # Chain Sampler
        mono_processing=True,
        chain_randomization='full',
        chain_len=None,
        min_chain_len=1,
        max_chain_len=2,
        # Modulation Sampler
        lfo_prob=0,
        # Parameter Sampler
        randomize_params=True,
        perceptual_intensity="default",
        target_sr=44100,
        max_ir_seconds=1.5,
        max_noise_seconds=2.5,
        unseen_noise=False,
        # Graph Sampler
        verbose=False,
        **kwargs
        ):

        self.verbose = verbose
        self.default_chain_type = default_chain_type
        self.structure_sampler = StructureSampler(fork_prob=fork_prob,
                                                  merge_prob=merge_prob,
                                                  min_node_num=min_node_num,
                                                  max_node_num=max_node_num,
                                                  force_merge_num=force_merge_num,
                                                  default_chain_type=default_chain_type)
        self.chain_sampler = ChainSampler(mono_processing=mono_processing,
                                          chain_randomization=chain_randomization,
                                          chain_len = chain_len,
                                          min_chain_len = min_chain_len,
                                          max_chain_len = max_chain_len,
                                          )
        self.modulation_sampler = ModulationSampler(lfo_prob=lfo_prob)
        self.parameter_sampler = ParameterSampler(randomize_params=randomize_params,
                                                  perceptual_intensity=perceptual_intensity,
                                                  target_sr=target_sr,
                                                  max_ir_seconds=max_ir_seconds,
                                                  max_noise_seconds=max_noise_seconds,
                                                  unseen_noise=unseen_noise,
                                                  )

    def __call__(self, *args, **kwargs):
        if self.verbose:
            start_time = time()
        S = self.structure_sampler()

        G = Grafx()
        G.structure = S
        last_node = G.add(afx_type="in", name='speech', chain='default', structure_node=0)
        G.chain_last_nodes['in'] = last_node

        for node, node_config in S.nodes(data=True):
            chain_type = node_config["chain_type"]
            last_nodes_pred = [S.nodes[pred]["last_node"] for pred in S.predecessors(node)]
            if "merge" in node_config:
                mix = G.add('mix', chain=chain_type, structure_node=node)
                for node_to_connect in last_nodes_pred:
                    G.connect(node_to_connect, mix)
                last_node = mix
            elif len(last_nodes_pred) == 1: # if there is the only one predecessor
                last_node = last_nodes_pred[0]
                
            last_node = self.chain_sampler(G, last_node, chain_type, structure_node=node, **kwargs)
            node_config["last_node"] = last_node

        G.add_and_connect("out", last_node, chain='default', structure_node=G.nodes[last_node]["structure_node"])
        self.modulation_sampler(G, *args, **kwargs)
        G.finalize()
        self.parameter_sampler(G, *args, **kwargs)
        
        if self.verbose:
            end_time = time()
            print(G)
            pprint(list(S.nodes(data=True)))
            print(f"Making a graph from structure took {(end_time - start_time):.3f}")
        return G
