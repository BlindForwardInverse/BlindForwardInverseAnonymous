from data_module.render_valid import RenderValidDataset
from utils.torch_utils import seed_everything

import os
import shutil
import random
import numpy as np
import argparse
from argparse import ArgumentParser

def render_valid(num_audio_per_afx,
                 target_sr,
                 audio_len,
                 seed,
                 randomize_params,
                 num_monolithic_graphs,
                 num_complex_graphs,
                 save_dir,
                 recording_envs,
                 render_effect_types,
                 single_afx_name,
                 # Nonverbal
                 modality,
                 # SE
                 pickle_graph,
                 single_mode,
                 **kwargs):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"Remove the previous valid_set in {save_dir}, and recreate")

    # Fix Seed
    seed_everything(seed)
    renderer = RenderValidDataset(num_audio_per_afx=num_audio_per_afx,
                                  target_sr=target_sr,
                                  target_audio_len=audio_len,
                                  ref_audio_len=audio_len,
                                  target_transform_type='waveform',
                                  ref_transform_type='waveform',
                                  mono=True,
                                  threshold_db=-8.,
                                  # Single AFX
                                  randomize_params=randomize_params,
                                  single_afx_name = single_afx_name,
                                  # Complex AFX
                                  num_monolithic_graphs = num_monolithic_graphs,
                                  num_complex_graphs = num_complex_graphs,
                                  # SAVE
                                  save_dir = save_dir,
                                  #SE
                                  pickle_graph = pickle_graph,
                                  single_mode = single_mode,
                                  )
    if modality == 'speech':
        renderer.render_valid_set(recording_envs=recording_envs,
                                  render_effect_types=render_effect_types,
                                  )
    elif modality == 'maestro':
        renderer.render_non_speech(render_effect_types=render_effect_types)

    elif modality == 'cross_modal':
        renderer.render_cross_modal(render_effect_types=render_effect_types)
    else:
        print("Not Implemented")

if __name__ == "__main__":
    import argparse
    import jax
    from argparse import ArgumentParser

    parser = ArgumentParser()
    add_arg = parser.add_argument
    Boolean = argparse.BooleanOptionalAction
    jax.config.update('jax_platform_name', 'cpu')

    add_arg("--num_audio_per_afx", type=int, default=5)
    add_arg("--target_sr", type=int, default=44100)
    add_arg("--audio_len", type=int, default=512*191)
    add_arg("--seed", type=int, default=42)
    add_arg("--randomize_params", default=False, action=Boolean)
    add_arg("--num_monolithic_graphs", type=int, default=5)
    add_arg("--num_complex_graphs", type=int, default=5)
    add_arg("--save_dir", default='/ssd4/doyo/valid_set')
    add_arg("--recording_envs", nargs='+', default=['vctk1', 'vctk2', 'daps'])
    add_arg("--render_effect_types", nargs='+', default=['single', 'monolithic', 'complex'])
    add_arg("--single_afx_name", nargs='+', default=None)

    # For SE
    add_arg("--pickle_graph", default=False, action=Boolean)
    add_arg("--single_mode", default=False, action=Boolean)

    # Render for non verbal dataset
    add_arg("--modality", default='speech')

    args = parser.parse_args()
    render_valid(**vars(args))
