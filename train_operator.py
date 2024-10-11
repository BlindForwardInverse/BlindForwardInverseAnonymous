import os
import warnings
from datetime import datetime

import pytorch_lightning as pl
import torch
import jax

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from data_module.train_dataset import TrainDataset
from data_module.valid_dataset import PrerenderedValidDataset

from solvers.operator_solver_1d import OperatorSolver1d
from solvers.operator_solver_2d import OperatorSolver2d
from solvers.operator_solver_hybrid import OperatorSolverHybrid
from solvers.operator_solver_hybrid_v2 import OperatorSolverHybridV2

from configs.train.train_operator_configs import add_args
from utils.torch_utils import seed_everything
jax.config.update("jax_platform_name", "cpu")
opj = os.path.join

def get_time_tag(time_tag):
    if time_tag == 'current':
        now = datetime.now()
        time_tag = now.strftime("%m%d_%H_")
    elif len(time_tag) > 0:
        time_tag = time_tag + '_'
    return time_tag

def create_dirs(*dir_names):
    for dir_name in dir_names:
        os.makedirs(dir_name, exist_ok=True)

def train():
    args = add_args()
    jax.config.update('jax_platform_name', 'cpu')

    # System Settings
    seed_everything(seed=args["seed"])
    time_tag = get_time_tag(args["time_tag"])
    exp_name = time_tag + args["exp_name"]
    ckpt_dir = opj(args["ckpt_dir"], args["project_name"], 'debug' if args["debug"] else exp_name)
    save_dir = opj(args["save_dir"], args["project_name"], 'debug' if args["debug"] else exp_name)
    create_dirs(ckpt_dir, save_dir)
    args["ckpt_dir"] = ckpt_dir
    args["save_dir"] = save_dir

    # Loggers
    logger = None
    if not args["debug"]:
        logger = WandbLogger(
            name=exp_name,
            version=exp_name,
            save_dir=save_dir,
            project=args["project_name"],
            log_model=False,
        )

    # Train Dataset
    train_ds = TrainDataset(
        # Training Recording Env
        split_mode =            args["split_mode"],
        single_env_mic =        args["single_env_mic"],
        # Data Return Type
        target_audio_len =      args["target_audio_len"],
        ref_audio_len =         args["ref_audio_len"],
        target_transform_type = "power_ri",
        ref_transform_type =    "waveform",
        return_24k =            False,
        return_ref_spec =       True if args["cat_ref"] else False,
        # Data Config
        modality =              args["modality"],
        target_sr =             args["target_sr"],
        n_fft =                 args["n_fft"],
        hop_length =            args["hop_length"],
        win_length =            args["win_length"],
        len_epoch =             args["batch_size"] * args["len_epoch"],
        # Graph type
        graph_type =            args["graph_type"], # "single_effect"
        default_chain_type =    args["default_chain_type"],
        perceptual_intensity =  args["perceptual_intensity"],
        randomize_params =      args["randomize_params"],
        single_afx_name =       args["single_afx_name"],
    )

    valid_set_types = ['vctk1'] if args["modality"] == 'speech' else ['maestro_single']
    valid_ds = PrerenderedValidDataset(
        modality =              args["modality"],
        target_sr =             args["target_sr"],
        tar_transform_type =    'power_ri',
        ref_transform_type =    'power_ri',
        n_fft =                 args["n_fft"],
        hop_length =            args["hop_length"],
        win_length =            args["win_length"],
        valid_set_types =       valid_set_types
        )
    full_afx_types = valid_ds.full_afx_types
    args["full_afx_types"] = full_afx_types
    print(full_afx_types)

    train_loader = DataLoader(
            train_ds,
            batch_size=args["batch_size"],
            drop_last=False,
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
        )
    valid_loader = DataLoader(
            valid_ds,
            batch_size=args["batch_size"],
            num_workers=16,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            drop_last=False)

    # Model
    if args["data_domain"] == 'spec':
        solver = OperatorSolver2d(**args)
    elif args["data_domain"] == 'waveform':
        solver = OperatorSolver1d(**args)
    elif args["data_domain"] == 'hybrid':
        if args["hybrid_v2"]:
            solver = OperatorSolverHybridV2(**args)
        else:
            solver = OperatorSolverHybrid(**args)

    # Regular Callbacks
    regular_callback = ModelCheckpoint(
        dirpath = ckpt_dir,
        filename="ckpt_operator_{epoch}-{step}",
        every_n_epochs=3,
        save_last=True,
        )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Main trainer module
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        max_epochs=250,
        log_every_n_steps=10,
        logger=logger,
        callbacks=[regular_callback, lr_monitor],
        fast_dev_run=args["debug"],
        num_sanity_val_steps=0,
    )

    torch.set_float32_matmul_precision("high")

    # Fit
    trainer.fit(
        solver,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=args["resume_training"],
    )

if __name__ == "__main__":
    import torch.multiprocessing
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    torch.multiprocessing.set_start_method('spawn')
    train()
