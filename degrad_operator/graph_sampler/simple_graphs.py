from .graph_sampler import GraphSampler

class MonolithicGraph(GraphSampler):
    def __init__(self, default_chain_type='full_afx', *args, **kwargs):
        default_kwargs = dict(
                max_node_num = 1,
                min_chain_len = 1,
                max_chain_len = 4,
                )
        default_kwargs.update(kwargs)
        super().__init__(*args, **default_kwargs)

class SingleEffectGraph(GraphSampler):
    def __init__(self, *args, **kwargs):
        default_kwargs = dict(
                max_node_num = 1,
                chain_len = 1,
                )
        default_kwargs.update(kwargs)
        super().__init__(*args, **default_kwargs)

    def get_supporting_afx(self):
        return self.chain_sampler.get_supporting_afx(chain_type='full_afx')

    def set_single_afx(self, afx):
        return self.chain_sampler.change_single_afx_name(afx)

class CustomGraph(GraphSampler):
    pass
