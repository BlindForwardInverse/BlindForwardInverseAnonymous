import os
import random
import time
from typing import List
from collections import Counter

import copy
import matplotlib.pyplot as plt
import networkx as nx
from degrad_operator.grafx import Grafx
from omegaconf import OmegaConf

opj = os.path.join

class Structure(nx.MultiDiGraph):
    def __init__(self, default_chain_type='full_afx'):
        super().__init__()
        self.add_node(0, chain_type=default_chain_type, level=0.) # Initial node

    def __str__(self):
        return f"Structure with {len(self.nodes)} nodes and {len(self.edges)} edges"

    @property
    def node_count(self):
        return len(self.nodes)

    def add_node(self, idx, chain_type='full_afx', **kwargs):
        super().add_node(idx, chain_type=chain_type, **kwargs)

class StructureSampler:
    def __init__(
        self,
        fork_prob=0.7,
        merge_prob=0.7,
        min_node_num=2,
        max_node_num=5,
        force_merge_num=3,
        default_chain_type='full_afx',
        ):
        """
        This returns a structure tree of the audio processing graph
        which has only fork and merge.
        """
        self.fork_prob = fork_prob
        self.merge_prob = merge_prob
        self.min_node_num = min_node_num
        self.max_node_num = max_node_num
        self.force_merge_num = force_merge_num

        self.default_chain_type = default_chain_type
        print("Default Chain Type : ", default_chain_type) 

    def __call__(self, monolithic=False):
        """
        Set self.max_node_num = 1 for monolithic
        """
        # Set up structure
        S = Structure(self.default_chain_type)
        self.cur_leaves = [0]
        while S.node_count < self.max_node_num:
            if len(self.cur_leaves) >= self.force_merge_num:
                merge_node_num = random.randint(2, len(self.cur_leaves))
                leaves_to_merge = random.sample(self.cur_leaves, merge_node_num)
                self.merge(S, leaves_to_merge, chain_type=self.default_chain_type)
                continue

            rand_prob = random.random()
            if rand_prob < self.fork_prob:
                fork_node_num = random.randint(1, len(self.cur_leaves))
                fork_restrict = True if fork_node_num >= 2 else False
                leaves_to_fork = random.sample(self.cur_leaves, fork_node_num)
                self.fork(S, leaves_to_fork, fork_restrict, chain_type=self.default_chain_type)

            elif rand_prob < self.merge_prob:
                if len(self.cur_leaves) < 2:
                    continue
                merge_node_num = random.randint(2, len(self.cur_leaves))
                leaves_to_merge = random.sample(self.cur_leaves, merge_node_num)
                self.merge(S, leaves_to_merge, chain_type=self.default_chain_type)

            else:
                if self.min_node_num <= S.node_count:
                    break  # Quit if min_node_num achieved

        # Final merge to output node
        if len(self.cur_leaves) > 1:
            self.merge(S, self.cur_leaves, chain_type='full_afx')
        return S

    def fork(self, S: Structure, leaves_to_fork: List[int], fork_restrict=False, chain_type='full_afx'):
        for node in leaves_to_fork:
            if fork_restrict:
                fork_count = 2
            else:
                fork_count = random.randint(2, 3)  # 2 or 3 forks from single node
            if S.node_count + fork_count > self.max_node_num:
                return

            level_offset = float(sum(range(fork_count)) / fork_count)
            previous_level = S.nodes[node]['level']
            weight = 0.9 if level_offset == 3 else 1.
            for i in range(fork_count):
                new_node = max(S.nodes) + 1
                S.add_node(new_node, chain_type=self.default_chain_type,
                           level=previous_level + weight*(float(i) - level_offset))
                S.add_edge(node, new_node)
                self.cur_leaves.append(new_node)
            self.cur_leaves.remove(node)
            
    def merge(self, S: Structure, leaves_to_merge: List[int], chain_type='full_afx'):
        new_node = max(S.nodes) + 1
        level = sum(S.nodes[node]['level'] for node in leaves_to_merge) / len(leaves_to_merge)
        S.add_node(new_node, merge=True, level=level, chain_type=chain_type)
        for node in leaves_to_merge:
            S.add_edge(node, new_node)
            self.cur_leaves.remove(node)
        self.cur_leaves.append(new_node)
