import argparse
from argparse import ArgumentParser

def add_args():
    parser = ArgumentParser()
    add_arg = parser.add_argument
    boolean = argparse.BooleanOptionalAction

    # Training
    add_arg("--batch_size", type=int, default=8)
    add_arg("--ckpt_dir", default="/ssd3/doyo/ckpt")
    add_arg("--save_dir", default="/ssd3/doyo/samples")
    add_arg("--logger", default="wandb")
    add_arg("--exp_name", default="test")
    add_arg("--project_name", default="operator_rb")

    add_arg("--time_tag", default='') # Set None for the current time. ex) "0715_01"

    # Learning rate / LR Schedule
    add_arg("--lr", type=float, default=1e-4)
    add_arg("--use_lr_schedule", default=False, action=argparse.BooleanOptionalAction)
    add_arg("--warmup_steps", type=int, default=500)
    add_arg("--first_cycle_steps", type=int, default=5000)
    add_arg("--gamma", type=float, default=0.97)
    add_arg("--max_lr", type=float, default=3e-4)
    add_arg("--min_lr", type=float, default=1e-5)

    # Dataset
    add_arg("--modality", default='speech')
    add_arg("--split_mode", default='single_recording_env')
    add_arg("--single_env_mic", default='mic1')
    add_arg("--target_sr", type=int, default=44100)
    add_arg("--target_audio_len", type=int, default= 512*191) # multiple of hop_size * 64 - 1
    add_arg("--ref_audio_len", type=int, default= 512*191)
    add_arg("--len_epoch", type=int, default=2000) # 1000 steps for 1 epoch

    # Wet audio
    add_arg("--default_chain_type", default='full_afx')
    add_arg("--graph_type", default='single_effect')
    add_arg("--perceptual_intensity", default='default')
    add_arg("--randomize_params", default=True, action=argparse.BooleanOptionalAction)
    add_arg("--single_afx_name", default=None)

    # Data Shape
    add_arg("--data_domain", default='spec') # 'spec' or 'waveform'
    add_arg("--n_fft", type=int, default=2046)
    add_arg("--win_length", type=int, default=2046)
    add_arg("--hop_length", type=int, default=512)

    # Seed
    add_arg("--seed", type=int, default=42)

    # Loss
    add_arg("--mse_each_domain", default=True, action=boolean)
    add_arg("--mse_weight", type=float, default=50)
    add_arg("--factor_sc", type=float, default=1)
    add_arg("--factor_mag", type=float, default=1)
    add_arg("--factor_phase", type=float, default=0.0)

    # Discriminator
    add_arg("--use_discriminator", default=True, action=argparse.BooleanOptionalAction)
    add_arg("--discriminator_type", default='MRD')
    add_arg("--pred_loss_weight", type=float, default=1.0)
    add_arg("--freeze_disc_steps", type=int, default=0)
    add_arg("--use_hinge_loss", default=False, action=argparse.BooleanOptionalAction)
    add_arg("--separate_discriminator", default=True, action=boolean)

    # Saving
    add_arg("--save_audio_per_n_epochs", type=int, default=3)

    # Training Strategies
    add_arg("--cat_ref", default=True, action=argparse.BooleanOptionalAction)
    add_arg("--use_ref_encoder", default=True, action=argparse.BooleanOptionalAction)
    add_arg("--hybrid_bridge", default=False, action=boolean)
    add_arg("--hybrid_v2", default=False, action=boolean)
    add_arg("--detach_global_condition", default=False, action=boolean)

    # FX Encoder
    add_arg("--encoder_channels", type=int, default=128)
    add_arg("--embedding_size", type=int, default=512)
    add_arg("--channel_mult", nargs='+', type=int, default=[1, 2, 4])
    add_arg("--ds_factors", nargs='+', type=int, default=[4, 4, 4])
    add_arg("--causal", default=False, action=argparse.BooleanOptionalAction)
    add_arg("--with_t_attn", default=True, action=argparse.BooleanOptionalAction)

    # Attn pool
    add_arg("--use_attn_pool", default=False, action=argparse.BooleanOptionalAction)
    add_arg("--attn_pool_num_latents", type=int, default=8)
    add_arg("--attn_pool_num_latents_mean_pooled", type=int, default=4)
    add_arg("--use_pretrained_codebook", default=False, action=argparse.BooleanOptionalAction)

    # Encodec
    add_arg("--bandwidth", type=float, default=6.0)

    # Condition
    add_arg("--cond_dim", type=int, default=512) # must match the last channel of the encoder
    add_arg("--cat_acoustic_token", default=False, action=argparse.BooleanOptionalAction)
    add_arg("--num_time_pool", type=int, default=12)
    add_arg("--partial_mean_pooled", default=False, action=argparse.BooleanOptionalAction)

    # Unet
    add_arg("--unet_channels", type=int, default=64)
    add_arg("--unet_dim_mults", nargs='+', type=int,    default=[1, 1, 2, 2, 2]) # 1 1 2 2 2
    add_arg("--num_resnet_blocks", nargs='+', type=int, default=[2, 2, 4, 4, 4]) # 2 2 4 8 8 
    add_arg("--num_layer_attns", type=int, default=2)
    add_arg("--num_layer_cross_attns", type=int, default=2)
    add_arg("--attn_dim_head", type=int, default=32)
    add_arg("--attn_heads", type=int, default=16)
    add_arg("--ff_mult", type=float, default=2.)
    add_arg("--memory_efficient", default=True, action=argparse.BooleanOptionalAction)

    add_arg("--unet_type", default='imagen')
    add_arg("--learnable_sum", default=True, action=boolean)

    # Debugging
    add_arg("--debug", dest="debug", action="store_true")
    parser.set_defaults(debug=False)
    add_arg("--resume_training", default=None)

    # checkpoint
    add_arg("--save_ckpt_per_every_n_epoch", type=int, default=4)
    args = parser.parse_args()
    return vars(args)
