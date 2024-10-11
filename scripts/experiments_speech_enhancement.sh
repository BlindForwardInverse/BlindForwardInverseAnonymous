#!/bin/bash
cd "$(dirname "$0")"/..

GPU_NUM=3
# Inference
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_inverse.py\
    --task conditional\
    --batch_size 32\
    --save_dir /ssd3/doyo/inference_enhancement_sampler2\
    --diffusion_config_name hybrid_hb1\
    --sampler_type heun_2nd\
    --test_set_path /ssd4/doyo/one_sample/single_effects_rand_param\
