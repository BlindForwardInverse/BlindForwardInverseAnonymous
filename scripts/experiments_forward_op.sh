#!/bin/bash
cd "$(dirname "$0")"/..

GPU_NUM=1
# Forward Operator Inference
# -------------------------------------------------
# (1) Effect of the shown wet signal types
# -------------------------------------------------
# Trained on Single Effect / Evaluated on Single Effect
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_operator.py\
    --task operator_learning\
    --save_dir /ssd4/doyo/infer_forward_final/train_single-eval_single\
    --config_name hb1-single\
    --test_set_path /ssd4/doyo/test_set_v2/single_effects_rand_param\
    --valid_set_types vctk1\
    --batch_size 48\

# Trained on Single Effect / Evaluated on Multi Effect
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_operator.py\
    --task operator_learning\
    --save_dir /ssd4/doyo/infer_forward_final/train_single-eval_multi\
    --config_name hb1-single\
    --test_set_path /ssd4/doyo/test_set_v2/multi_effects\
    --valid_set_types vctk1\
    --batch_size 48\

# Trained on Multi Effect / Evaluated on Single Effect
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_operator.py\
    --task operator_learning\
    --save_dir /ssd4/doyo/infer_forward_final/train_multi-eval_single\
    --config_name hb1-multiple\
    --test_set_path /ssd4/doyo/test_set_v2/single_effects_rand_param\
    --valid_set_types vctk1\
    --batch_size 48\

# Trained on Multi Effect / Evaluated on Multi Effect
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_operator.py\
    --task operator_learning\
    --config_name hb1-multiple\
    --save_dir /ssd4/doyo/infer_forward_final/train_multi-eval_multi\
    --test_set_path /ssd4/doyo/test_set_v2/multi_effects\
    --valid_set_types vctk1\
    --batch_size 48\

# -------------------------------------------------
# (2) Effect of the Recording Envs. of the training set
# -------------------------------------------------
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_operator.py\
    --task operator_learning\
    --save_dir /ssd4/doyo/infer_forward_exp2-rec_env/single_recording_env\
    --config_name hb1-single\
    --test_set_path /ssd4/doyo/test_set_v2/single_effects_multi_env\
    --valid_set_types vctk1 vctk2 daps\
    --batch_size 48\

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_operator.py\
    --task operator_learning\
    --save_dir /ssd4/doyo/infer_forward_exp2-rec_env/multiple_recording_env\
    --config_name hb1-multiple_env\
    --test_set_path /ssd4/doyo/test_set_v2/single_effects_multi_env\
    --valid_set_types vctk1 vctk2 daps\
    --batch_size 48\

# -------------------------------------------------
# (3) Cross Domain
# -------------------------------------------------
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 inference_operator.py\
    --task operator_learning\
    --save_dir /ssd4/doyo/infer_forward_cross_domain/cross_domain\
    --config_name hb1-cross_domain\
    --test_set_path /ssd4/doyo/test_set_v2/cross_domain_single\
    --valid_set_types maestro_single\
    --batch_size 48\
