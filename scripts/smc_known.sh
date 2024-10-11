#!/bin/bash
cd "$(dirname "$0")"/..

GPU_NUM=3
# Inference

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_inverse.py\
    --task smc_known\
    --batch_size 12\
    --num_particles 4\
    --save_dir /ssd4/doyo/smc_known\
    --diffusion_config_name hybrid_hb1\
    --sampler_type heun_2nd_sigma1\
    --test_set_path /ssd4/doyo/one_sample/single_effects_rand_param\

