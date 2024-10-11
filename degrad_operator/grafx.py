"""
Grafx: an audio processor graph.
"""
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Union
from omegaconf import OmegaConf

from .afx_jax import jafx
from .afx_jax.jafx_utils import rms_normalize

IGNORE_INVALID_OPERATIONS = False
VERBOSE = True


def raise_warning(raise_str):
    if IGNORE_INVALID_OPERATIONS and VERBOSE:
        print("Ignoring following operation: " + raise_str)
    else:
        raise Exception(raise_str)


class Grafx(nx.MultiDiGraph):
    def __init__(self):
        super().__init__()
        self.afx_configs = OmegaConf.load("configs/degrad_operator/afx_module_configs.yaml")
        self.chain_last_nodes = dict()
        self.num_rir = 0
        self.num_micir = 0
        self.num_noise = 0
        self.chain_types = list()

    @property
    def parameters(self):
        return {node : data['parameters'] for node, data in self.nodes(data=True)}
    
    @property
    def afx_types(self):
        return {node : data['afx_type'] for node, data in self.nodes(data=True)}

    @property
    def input_signal_idx(self):
        return [idx for idx in self.nodes() if self.afx_types[idx] == 'in']

    def __str__(self):
        _str = ""
        nodes = self.nodes(data=True)
        num_nodes = self.number_of_nodes()
        num_edges = self.number_of_edges()
        _str += "Grafx with %d processors & %d connections\n" % (num_nodes, num_edges)
        for node in nodes:
            i, afx_type, parameters = (
                node[0],
                f"{node[1]['afx_type'] if 'afx_type' in node else ''}",
                f"{node[1]['parameters'] if 'parameters' in node else ''}",
            )
            _str += f"{i} [{afx_type}]\n"
            out_edges = self.out_edges([i], data=True)
            if len(out_edges) != 0:
                _str += " " * 6 + "- connected to: \n"
                for e in out_edges:
                    _, to, config = e
                    outlet, inlet, gain = config.values()
                    _str += " " * 8
                    if outlet != "main":
                        _str += f"<{outlet}> "
                    _str += f'-> {to} [{nodes[to]["afx_type"]}] '
                    if inlet != "main":
                        _str += f"<{inlet}> "
                    if abs(gain) > 0.1:
                        # print(gain)
                        _str += f"({gain:.1f}dB)"
                    _str += "\n"
        return _str

    def add(
        self,
        afx_type: Union[str, int] = None,
        name: str = None,
        chain=None,
        structure_node=None,
        **parameters,
    ) -> int:
        node_class = self.afx_configs[afx_type]["class"]
        parameter_config = self.afx_configs[afx_type]["parameters"]
        empty_parameters = {k: None for k in parameter_config.keys()}
        parameters = {**empty_parameters, **parameters}

        # Chain config
        if chain != None:
            chain = {"chain": chain}
        else:
            if node_class == "input":
                chain = {"chain": afx_type}
            else:
                chain = {}

        # Generate Afx name
        if name == None:
            if afx_type in self.nodes():
                dup_idx = 1
                while True:
                    name = f"{afx_type}_{dup_idx}"
                    if name not in self.nodes():
                        break
                    dup_idx += 1
            else:
                name = afx_type

        assert name not in self.nodes()
        if afx_type == 'rir' : self.num_rir += 1
        elif afx_type == 'micir' : self.num_micir += 1
        elif afx_type == 'noise' : self.num_noise += 1

        self.add_node(name, afx_type=afx_type, parameters=parameters, structure_node=structure_node, **chain)
        return name

    def connect(
        self,
        i_from: Union[str, int],
        i_to: Union[str, int],
        outlet: Union[str, int] = "main",
        inlet: Union[str, int] = "main",
        gain=0.0,
        afx_type_from = None,
    ):
        """ """
        afx_type_from = afx_type_from if afx_type_from is not None else self.nodes[i_from]["afx_type"]
        if afx_type_from == "in":
            pass
        else:
            if afx_type_from is not None:
                outlet_config = self.afx_configs[afx_type_from]["outlets"]
                if isinstance(outlet, int):
                    if not outlet < len(outlet_config):
                        raise_warning(
                            f"Provided outlet index: {outlet}, while {afx_type_from} only accepts range of 0-{len(outlet_config)-1}."
                        )
                        return
                    outlet = outlet_config[outlet]
                else:
                    if not outlet in outlet_config:
                        raise_warning(
                            f"Provided outlet str: '{outlet}', while {afx_type_from} only accepts {outlet_config}."
                        )
                        return

        afx_type_to = self.nodes[i_to]["afx_type"]
        inlet_config = self.afx_configs[afx_type_to]["inlets"]
        if isinstance(inlet, int):
            if not inlet < len(inlet_config):
                raise_warning(
                    f"Provided inlet index: {inlet}, while {afx_type_to} only accepts range of 0-{len(inlet_config)-1}. ({i_from})"
                )
                return
            inlet = inlet_config[inlet]
        else:
            if not inlet in inlet_config:
                raise_warning(
                    f"Provided inlet str: '{inlet}', while {afx_type_to} only accepts {inlet_config}."
                )
                return
        self.add_edge(i_from, i_to, outlet=outlet, inlet=inlet, gain=gain)

    def add_and_connect(
        self,
        afx_type: Union[str, int] = None,
        i_from=None,
        outlet="main",
        inlet="main",
        gain=0.0,
        chain=None,
        structure_node=None,
        **parameters,
    ) -> int:
        r""" """
        name = self.add(afx_type, chain=chain, structure_node=structure_node, **parameters)

        if i_from is None:
            self.connect_last_two(outlet=outlet, inlet=inlet, gain=gain)
        else:
            self.connect(i_from, name, outlet=outlet, inlet=inlet, gain=gain)
        return name

    def connect_multiple(self, connections: List[Union[Tuple[int, int], Tuple[int, int, dict]]]):
        for c in connections:
            if len(c) == 2:
                self.connect(c[0], c[1])
            else:
                self.connect(c[0], c[1], **c[2])

    def connect_last_two(self, outlet="main", inlet="main", gain=0.0):
        self.connect(
            list(self.nodes())[-2],
            list(self.nodes())[-1],
            outlet=outlet,
            inlet=inlet,
            gain=gain,
        )

    def add_serial_chain(
        self,
        afx_configs: List[Union[str, dict]],
        i_from: int = None,
        outlet: str = "main",
        mix: float = 1.0,
        mix_to: int = None,
    ):
        r"""
        Add a multiple processors with serial main-to-main connections & additional dry/wet mixing.

        .. code-block:: python

            G = Grafx()
            # add a crossover processor, which has two outlets: low and high.
            i_crossover = G.add('crossover')
            #
            G.add_serial_chain([dict(afx_type='reverb', frequency_hz=440.),
                                dict(afx_type='distortion')],
                                outlet='low',
                                mix=0.5)
            G.add_serial_chain(['chorus', 'bitcrush'],
                                outlet=i_crossover)

        Args:
            afx_configs (list of dicts or strs, required): configs of the processors. Each dictionary of the list must have `afx_type` item.
            i_from (int, optional): index of a processor that will be connected to the serial chain.
            outlet (str, optional): outlet of the connection.
            mix (float, optional): the mixing coefficient. Provided values will be clipped to minimum 0 or maximum 1.
            mix_to (float, optional): index of a
        """

        if i_from == None:
            i_from = list(self.nodes())[-1]

        for i in range(len(afx_configs)):
            if type(afx_configs[i]) == str:
                self.add(afx_configs[i])
            else:
                self.add(**afx_configs[i])
            if i != 0:
                self.connect_last_two()
            if i == 0:
                i_first = list(self.nodes())[-1]
            if i == len(afx_configs) - 1:
                i_last = list(self.nodes())[-1]
        self.connect(i_from, i_first, outlet=outlet)
        if mix != 1:
            i_mix = self.add("mix") if mix_to == None else mix_to

            mix = np.clip(mix, 1e-5, 1 - 1e-5)
            wet, dry = mix, 1 - mix
            norm = np.sqrt(wet**2 + dry**2)
            wet, dry = wet / norm, dry / norm
            wet_db, dry_db = 20 * np.log10(mix), 20 * np.log10(1 - mix)

            self.connect(i_last, i_mix, gain=wet_db)
            self.connect(i_from, i_mix, outlet=outlet, gain=dry_db)
            self.i = i_mix
            return dict(i_from=i_from, first=i_first, last=i_last, mix=i_mix)
        else:
            self.i = i_last
            return dict(i_from=i_from, first=i_first, last=i_last)

    def add_parallel_chains(
        self,
        afx_configs: List[Tuple[Union[list, dict, str], float]],
        i_from: int = None,
        outlet: str = "main",
        dry_gain=None,
    ):
        r"""
        Add a multiple processors with parallel main-to-main connections & mixings

        .. code-block:: python

            G = Grafx()
            G.add_parallel_chains([([dict(afx_type='reverb', damping=0.5), dict(afx_type='distortion')], 0),
                                   ([dict(afx_type='reverb', damping=
                                   outlet='low',
                                   mix=0.5)

        Args:
            afx_configs (list of tuples, required): configs of the processors. Each tuple of the provided list must be a pair of a processor list (or a single processor) and a float that represents mixing gain of the chain (or processor).
            i_from (int, optional): index of a processor that will be connected to the serial chain.
            outlet (str, optional): outlet of the connection.
        """

        assert len(afx_configs) != 0
        if i_from == None:
            i_from = list(self.nodes())[-1]

        # when processor chains are given:
        i_lasts = []
        for afx_config, gain in afx_configs:
            if type(afx_config) == list:
                i_last = self.add_serial_chain(afx_config, i_from=i_from, outlet=outlet)["last"]
            else:
                if type(afx_config) == str:
                    i_last = self.add(afx_config)
                else:
                    i_last = self.add(**afx_config)
                self.connect(i_from, i_last, outlet=outlet)
            i_lasts.append(i_last)

        i_mix = self.add("mix")
        for i_last, (_, gain) in zip(i_lasts, afx_configs):
            self.connect(i_last, i_mix, gain=gain)
        if dry_gain != None:
            self.connect(i_from, i_mix, outlet=outlet, gain=dry_gain)
        return i_mix

    def finalize(self):
        for node_idx in self.nodes:
            if self.nodes[node_idx]["afx_type"] != "out" and len(self.out_edges([node_idx])) == 0:
                self.connect(node_idx, "out")

        nodes = list(self.nodes())
        half_nodes = ['lfo', 'stereo_lfo', 'rir', 'micir', 'noise']
        def is_lfo_behind(suc):
            pred_types = [self.nodes[pred]['afx_type'] for pred in self.predecessors(suc)]
            overlap = set(half_nodes).intersection(set(pred_types))
            return len(overlap) > 0

        for node_idx in nodes:
            if self.nodes[node_idx]['afx_type'] == 'mix':
                successors = [suc for suc in self.successors(node_idx)]
                predecessors = [pred for pred in self.predecessors(node_idx) if pred not in half_nodes]
                if (len(successors) == 1
                    and len(predecessors) == 1
                    and not is_lfo_behind(successors[0])):

                    for edge in self.in_edges(nbunch=node_idx, data=True):
                        pred, _, config = edge
                        self.connect(pred, successors[0], **config)

                    preds = list(self.predecessors(node_idx))
                    for pre in preds:
                        self.remove_edge(pre, node_idx)
                    self.remove_node(node_idx)
