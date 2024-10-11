#!/bin/bash
cd "$(dirname "$0")"/..

GPU_NUM=0
# Inference

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_inverse.py\
    --task conditional\
    --batch_size 32\
    --save_dir /ssd3/doyo/speech_enhancement_conditional/sampler1\
    --diffusion_config_name hybrid_hb1\
    --sampler_type heun_2nd_sigma2\
    --test_set_path /ssd4/doyo/test_set_tiny/single_effects_rand_param\

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_inverse.py\
    --task conditional\
    --batch_size 32\
    --save_dir /ssd4/doyo/speech_enhancement_conditional/sampler2\
    --diffusion_config_name hybrid_hb1\
    --sampler_type heun_2nd_sigma2\
    --test_set_path /ssd4/doyo/test_set_tiny/single_effects_rand_param\

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_inverse.py\
    --task conditional\
    --batch_size 32\
    --save_dir /ssd4/doyo/speech_enhancement_conditional/sampler3\
    --diffusion_config_name hybrid_hb1\
    --sampler_type heun_2nd_sigma2\
    --test_set_path /ssd4/doyo/test_set_tiny/single_effects_rand_param\

#CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_inverse.py\
#    --task particle_filter_unknown\
#    --batch_size 6\
#    --save_dir /ssd4/doyo/inference_se_unknown_op\
#    --diffusion_config_name hybrid_hb1\
#    --sampler_type heun_2nd_sigma2\
#    --test_set_path /ssd4/doyo/test_set_tiny_se/single_effects_rand_param\

#CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_inverse.py\
#    --task particle_filter_known\
#    --batch_size 6\
#    --save_dir /ssd4/doyo/inference_se_known_op\
#    --diffusion_config_name hybrid_hb1\
#    --sampler_type heun_2nd_sigma2\
#    --test_set_path /ssd4/doyo/test_set_tiny_se/single_effects_rand_param\

