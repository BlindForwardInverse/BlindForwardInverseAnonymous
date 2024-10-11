import os
import warnings
from datetime import datetime

# import pytorch_lightning as pl
import torch
import jax

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.utils.data import DataLoader

from data_module.diffusion_dataset import DiffusionTrainDataset
from data_module.dummy_dataset import DummyDataset
from data_module.valid_dataset import PrerenderedValidDataset

from solvers.diffusion_solver_2d import DiffusionSpecSolver
from solvers.diffusion_solver_hybrid import DiffusionHybridSolver
# from solvers.diffusion_solver_hybrid_v2 import DiffusionHybridSolverV2
from solvers.diffusion_solver_2d_proj import DiffusionSpecProj
from solvers.diffusion_solver_hybrid_proj import DiffusionHybridSolverProj

from solvers.diffusion_ousde import DiffusionOUSDE
from solvers.operator_solver_hybrid import OperatorSolverHybrid
from solvers.operator_solver_hybrid_v2 import OperatorSolverHybridV2
from configs.train.diffusion_args import add_args

from utils.torch_utils import seed_everything, load_pretrained_model
jax.config.update('jax_platform_name', 'cpu')

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
    len_epoch = args["len_epoch"] * args["batch_size"]
    train_dataset = DiffusionTrainDataset(
        split_mode=args["split_mode"],
        single_env_mic=args["single_env_mic"],
        audio_len = args["audio_len"],
        transform_type = args["transform_type"],
        target_sr = args["target_sr"],
        threshold_db = -8,
        len_epoch = args["len_epoch"] * args["batch_size"],
        n_fft= args["n_fft"],
        win_length = args["win_length"],
        hop_length = args["hop_length"],
        #Graph
        default_chain_type = args["default_chain_type"],
        graph_type = args["graph_type"],
    )

    valid_dataset = PrerenderedValidDataset(
        target_sr = args["target_sr"],
        n_fft= args["n_fft"],
        win_length = args["win_length"],
        hop_length = args["hop_length"],
        prerendered_valid_dir = '/ssd4/doyo/valid_set_diffusion'
    )

    full_afx_types = valid_dataset.full_afx_types
    args["full_afx_types"] = full_afx_types
    print(full_afx_types)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args["batch_size"],
        drop_last=False,
    )

    # PytorchLightning Module based model
    if args['ousde']:
        solver = DiffusionOUSDE(**args)
    elif args['data_domain'] == 'spec':
        solver = DiffusionSpecSolver(**args)
    elif args['data_domain'] == 'waveform':
        solver = DiffusionWavSolver(**args)
    elif args['data_domain'] == 'hybrid':
        solver = DiffusionHybridSolver(**args)
    elif args['data_domain'] == 'hybrid_v2':
        solver = DiffusionHybridSolverV2(**args)
    elif args['data_domain'] == 'spec_proj':
        solver = DiffusionSpecProj(**args)
    elif args['data_domain'] == 'hybrid_proj':
        solver = DiffusionHybridSolverProj(**args)

    # Regular Callbacks
    regular_callback = ModelCheckpoint(
        dirpath = args["ckpt_dir"],
        filename="ckpt_diffusion_{epoch}-{step}",
        every_n_epochs=3,
        save_last=True,
        )

    # Main trainer Module
    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        max_epochs=200,
        log_every_n_steps=10,
        logger=logger,
        fast_dev_run=args["debug"],
        num_sanity_val_steps=0,
        callbacks=[
#             ckpt_callback,
            regular_callback],
    )

    torch.set_float32_matmul_precision("high")

    # Fit
    if args["resume_training"]:
        trainer.fit(
            solver,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=args["resume_training"],
        )
    else:
        trainer.fit(
            solver,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn')
    train()
