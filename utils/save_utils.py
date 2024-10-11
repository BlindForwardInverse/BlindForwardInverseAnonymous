import hashlib
import os
import re
from datetime import datetime
import wave

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from torchvision.utils import save_image

from .tensor_utils import get_numpy

opj = os.path.join

def save_audio(audio_batch, sr, G=None, pair_audio_batch=None, path="sample"):
    """
    Save rendered audio (and processed audio) with graph name
    """
    # assert audio.shape == pair_audio.shape  # [B, L, C]
    os.makedirs(path, exist_ok=True)

    # Iterate over batch
    for i in range(audio_batch.shape[0]):
        print(i)
        audio = audio_batch[i]  # [L, C]
        print(audio.shape)

        hash = generate_audio_hash(audio, simplify=True)
        name = f"{hash}" if G is None else f"{get_graph_info(G)}_{hash}"

        audio_path = opj(path, f"{name}_y.wav")
        try:
            sf.write(audio_path, audio, sr)
        except RuntimeError as e:
            print(f"Runtime error occurred: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        print(f"Saved audio at {audio_path}")

        if pair_audio_batch is not None:
            pair_audio = pair_audio_batch[i]  # [L, C]
            pair_audio_path = opj(path, f"{name}_x.wav")
            try:
                sf.write(pair_audio_path, pair_audio, sr)
            except RuntimeError as e:
                print(f"Runtime error occurred: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
            print(f"Saved audio at {pair_audio_path}")

def get_graph_info(G, with_params=False):
    """
    Create graph name with processor nodes
    """
    # TODO apply modulation sampler
    _str = ""
    nodes = G.nodes(data=True)
    for n, node in enumerate(nodes):
        i, afx_type, parameters = (node[0], node[1]["afx_type"], node[1]["parameters"])

        if afx_type in ["in", "out", "convolution"]:
            if "rir" in node[0] or "micir" in node[0]:
                _str += re.match("^[^_]*", node[0]).group(0)
            continue
        for k, v in parameters.items():
            parameters.update({k: float(f"{v:.2f}")})

        if not with_params:
            _str += f"{afx_type}_"
        else:
            _str += f"{afx_type}_{parameters}"

    return _str
