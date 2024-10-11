import numpy as np
from scipy.stats import loguniform, truncnorm
from numpy.random import uniform
import omegaconf

def choice(l, w=None):
    if isinstance(l, omegaconf.listconfig.ListConfig):
        l = list(l)
    if isinstance(w, omegaconf.listconfig.ListConfig):
        w = list(w)

    assert w is None or len(l) == len(w), f'{l} != {w}'
    p = np.array(w) / sum(w) if w != None else None
    return l[np.random.choice(np.arange(len(l)), p=p)]

def choices(l, w=None, size=2, replace=False, preserve_order=False):
    if isinstance(l, omegaconf.listconfig.ListConfig):
        l = list(l)
    if isinstance(w, omegaconf.listconfig.ListConfig):
        w = list(w)
    assert w is None or len(l) == len(w), f'{l} != {w}'
    p = np.array(w) / sum(w) if w != None else None
    idxs = list(np.random.choice(np.arange(len(l)), p=p, size=size, replace=replace))
    if preserve_order:
        idxs.sort()
    return [l[i] for i in idxs]

def true_or_false(p):
    return np.random.choice([True, False], p=[p, 1 - p])

def get_random_values(rule, *args):
    if rule == "uniform":
        return np.random.uniform(*args)
    elif rule == "loguniform":
        return loguniform.rvs(*args)
    elif rule == "categorical":
        return np.random.choice(args)
    elif rule == "truncnorm":
        return run_truncnorm(*args)
    elif rule == "truncnorm_with_random_sign":
        return run_truncnorm_with_random_sign(*args)
    elif rule == 'randint':
        return np.random.randint(*args)

def run_truncnorm(v1, v2, mu, s):
    a, b = (v1 - mu) / s, (v2 - mu) / s
    return truncnorm.rvs(a, b, mu, s)

def run_truncnorm_with_random_sign(v1, v2, mu, s):
    sign = np.random.choice([1, -1])
    return sign * run_truncnorm(v1, v2, mu, s)

def get_mix_db(n=2, initial_db_range=(-12, 12)):
    mix_db = uniform(-12, 12, size=(n,))
    mix = 10 ** (mix_db / 20)
    mix = mix / np.sqrt(np.sum(mix**2))
    mix_db = 20 * np.log10(mix)
    return list(mix_db)

def update_node_meta(G, chain):
    for idx, node in G.nodes(data=True):
        if "chain" not in node:
            node["chain"] = chain
