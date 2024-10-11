#!/bin/bash
cd "$(dirname "$0")"/..
# This renders (dry-wet) audio pairs for a subjective test.
# 100 audios per each afx
# afx_names =  ['lowpass', 'lowpass_ladder', 'bandpass', 'bandpass_ladder', 'highpass', 'highpass_ladder', 'bandreject', 'distortion', 'hard_clipper', 'soft_clipper', 'mono_reverb', 'rir_conv', 'micir_conv', 'add_noise']

GPU_NUM=0
# -----------------------------------------------------
# 1) Audio Effect Complexity  Experiments
# -----------------------------------------------------
# Single Effect / Seen Env
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 prerender_valid.py\
    --num_audio_per_afx 12\
    --no-randomize_params\
    --save_dir /ssd4/doyo/test_set_tiny/single_effects\
    --recording_envs vctk1\
    --render_effect_types single\

# Single Effect + Random Param / Seen
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 prerender_valid.py\
    --num_audio_per_afx 12\
    --randomize_params\
    --seed 0\
    --save_dir /ssd4/doyo/test_set_tiny/single_effects_rand_param\
    --recording_envs vctk1\
    --render_effect_types single\

# Single Effect + Random Param / Seen
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 prerender_valid.py\
    --num_audio_per_afx 12\
    --randomize_params\
    --save_dir /ssd4/doyo/test_set_tiny/single_effects_rand_param_2\
    --seed 42\
    --recording_envs vctk1\
    --render_effect_types single\

# Multiple Effect / Seen
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 prerender_valid.py\
    --num_audio_per_afx 12\
    --randomize_params\
    --save_dir /ssd4/doyo/test_set_tiny/multi_effects\
    --recording_envs vctk1\
    --render_effect_types monolithic complex\
    --num_monolithic_graphs 5\
    --num_complex_graphs 5\

# -----------------------------------------------------
# 2) Environement Experiments
# -----------------------------------------------------
# Single Effect / Multiple Env
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 prerender_valid.py\
    --num_audio_per_afx 12\
    --randomize_params\
    --save_dir /ssd4/doyo/test_set_tiny/single_effects_multi_env\
    --recording_envs vctk1 vctk2 daps\
    --render_effect_types single

# Multi Effect / Multiple Env
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 prerender_valid.py\
    --num_audio_per_afx 12\
    --randomize_params\
    --save_dir /ssd4/doyo/test_set_tiny/multiple_effects_multi_env\
    --recording_envs vctk1 vctk2 daps\
    --render_effect_types monolithic complex\
    --num_monolithic_graphs 5\
    --num_complex_graphs 5\
