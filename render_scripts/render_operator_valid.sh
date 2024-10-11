#!/bin/bash
cd "$(dirname "$0")"/..
# This renders (dry-wet) audio pairs for a subjective test.
# 100 audios per each afx
# afx_names =  ['lowpass', 'lowpass_ladder', 'bandpass', 'bandpass_ladder', 'highpass', 'highpass_ladder', 'bandreject', 'distortion', 'hard_clipper', 'soft_clipper', 'mono_reverb', 'rir_conv', 'micir_conv', 'add_noise']

GPU_NUM=5
# -----------------------------------------------------
# 1) Audio Effect Complexity  Experiments
# -----------------------------------------------------
# Single Effect / Seen Env
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 prerender_valid.py\
    --num_audio_per_afx 4\
    --no-randomize_params\
    --save_dir /ssd4/doyo/valid_set\
    --seed 0\
    --modality speech \
    --render_effect_types single monolithic complex\

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 prerender_valid.py\
    --num_audio_per_afx 4\
    --no-randomize_params\
    --seed 0\
    --save_dir /ssd4/doyo/valid_maestro\
    --modality maestro\
    --render_effect_types single monolithic complex\

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 prerender_valid.py\
    --num_audio_per_afx 4\
    --no-randomize_params\
    --seed 0\
    --save_dir /ssd4/doyo/valid_cross_modal\
    --modality cross_modal\
    --render_effect_types single monolithic complex\

