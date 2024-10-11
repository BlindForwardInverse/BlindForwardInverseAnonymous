#!/bin/bash
cd "$(dirname "$0")"/..

GPU_NUM=1
# Inference
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_inverse.py\
    --task particle_filter_unknown\
    --batch_size 8\
    --save_dir /ssd4/doyo/inference_se_unknown_op_2\
    --diffusion_config_name hybrid_hb1\
    --sampler_type heun_2nd\
    --test_set_path /ssd4/doyo/one_sample/single_effects_rand_param\
