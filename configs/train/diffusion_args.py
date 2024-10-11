import argparse
from argparse import ArgumentParser


def edm_args(parser):
    add_arg = parser.add_argument
    add_arg("--p_mean", type=float, default=-1.2)
    add_arg("--p_std", type=float, default=1.2)
    add_arg("--diffusion_steps", type=int, default=32)
    add_arg("--sigma_min", type=float, default=5e-4)
    add_arg("--sigma_max", type=float, default=3)
    add_arg("--sigma_data", type=float, default=0.0126)
    add_arg("--rho", type=float, default=13)
    add_arg("--train_rho", type=float, default=13)
    add_arg("--S_noise", type=float, default=1.003)
    add_arg("--S_churn", type=float, default=5)
    add_arg("--S_tmin", type=float, default=0.3)
    add_arg("--S_tmax", type=float, default=2.0)
    return parser

def add_args():
    parser = ArgumentParser()
    boolean = argparse.BooleanOptionalAction

    add_arg = parser.add_argument
    add_arg("--ckpt_dir", default="/ssd3/doyo/diffusion_v2")
    add_arg("--save_dir", default="/ssd3/doyo/diffusion_v2")
    add_arg("--config_path", default="configs/pretrained_model/forward_operator.yaml")
    add_arg("--config_name", default="hb1-single")

    add_arg("--logger", default="wandb")
    add_arg("--exp_name", default="test")
    add_arg("--time_tag", default='') # Set None for the current time. ex) "0715_01"
    add_arg("--project_name", default="diffusion_v2")

    # Trainer specific args
    add_arg("--trainer", default="edm")
    trainer = parser.parse_known_args()[0].trainer
    if trainer == "edm" or "edm2":
        parser = edm_args(parser)

    # Datamodule Parameters
    add_arg("--split_mode", default="single_recording_env")
    add_arg("--single_env_mic", default="mic1")
    add_arg("--audio_len", type=int, default=512*191)
    add_arg("--batch_size", type=int, default=12)
    add_arg("--target_sr", type=int, default=44100)
    add_arg("--len_epoch", type=int, default=3000)
    add_arg("--transform_type", default="power_ri")
    add_arg("--default_chain_type", default='speech_enhancement')

    add_arg("--data_domain", default='hybrid') # 'spec' or 'waveform'
    add_arg("--n_fft", type=int, default=2046)
    add_arg("--win_length", type=int, default=2046)
    add_arg("--hop_length", type=int, default=512)
    add_arg("--graph_type", default='single_effect')

    # Model Parameters
    add_arg("--unet_channels", type=int, default=64)
    add_arg("--num_resnet_blocks", nargs='+', type=int, default=[2, 2, 4, 4, 8])
    add_arg("--dim_mults", nargs='+', type=int,         default=[1, 2, 2, 2, 4])
    add_arg("--num_layer_attns", type=int, default=2)
    add_arg("--num_layer_cross_attns", type=int, default=2)
    add_arg("--attn_dim_head", type=int, default=64)
    add_arg("--attn_heads", type=int, default=8)
    add_arg("--cat_global_condition", default=False, action=boolean)
    add_arg("--norm_local_condition", default=True, action=boolean)

    add_arg("--sample_from_log_normal", default=False, action=boolean)

    add_arg("--use_ref_encoder", default=True, action=argparse.BooleanOptionalAction)
    add_arg("--detach_spec", default=False, action=boolean)
    add_arg("--operator_config_name", default='operator_w_disc_single')
    add_arg("--cond_dim", type=int, default=512)

    add_arg("--ousde", default=False, action=boolean)
    add_arg("--use_discriminator", default=False, action=boolean)

    # Train Parameters
    add_arg("--lr", type=float, default=1e-4)
    add_arg("--ema_decay", type=float, default=0.999)
    add_arg("--seed", type=int, default=42)
    add_arg("--save_every_n_epochs", type=int, default=1)

    # Loss weight (for deterministic)
    add_arg("--mse_weight", type=float, default=50.)
    add_arg("--factor_sc", type=float, default=1.)
    add_arg("--factor_mag", type=float, default=1.)
    add_arg("--factor_phase", type=float, default=0)
    add_arg("--loss_type", default='mse')

    # Debugging
    add_arg("--debug", dest="debug", action="store_true")
    parser.set_defaults(debug=False)
    add_arg("--resume_training", default=None)

    # Inference
    args = parser.parse_args()
    return vars(args)
