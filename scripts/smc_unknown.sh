#!/bin/bash
cd "$(dirname "$0")"/..

GPU_NUM=5

# Inference
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_inverse.py\
    --task smc_unknown\
    --batch_size 8\
    --num_particles 4\
    --save_dir /ssd4/doyo/smc_unknown/particle_4\
    --diffusion_config_name hybrid_hb1\
    --sampler_type heun_2nd_sigma1\
    --test_set_path /ssd4/doyo/test_set_pickle_graph/single_effects_rand_param\

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_inverse.py\
    --task smc_unknown\
    --batch_size 8\
    --num_particles 2\
    --save_dir /ssd4/doyo/smc_unknown/particle_2\
    --diffusion_config_name hybrid_hb1\
    --sampler_type heun_2nd_sigma1\
    --test_set_path /ssd4/doyo/test_set_pickle_graph/single_effects_rand_param\

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_inverse.py\
    --task smc_unknown\
    --batch_size 4\
    --num_particles 8\
    --save_dir /ssd4/doyo/smc_unknown/particle_8\
    --diffusion_config_name hybrid_hb1\
    --sampler_type heun_2nd_sigma1\
    --test_set_path /ssd4/doyo/test_set_pickle_graph/single_effects_rand_param\

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_inverse.py\
    --task smc_unknown\
    --batch_size 2\
    --num_particles 16\
    --save_dir /ssd4/doyo/smc_unknown/particle_16\
    --diffusion_config_name hybrid_hb1\
    --sampler_type heun_2nd_sigma1\
    --test_set_path /ssd4/doyo/test_set_pickle_graph/single_effects_rand_param\
