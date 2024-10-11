import torch
from solvers.diffusion_solver_hybrid import DiffusionHybridSolver

def ckpt_to_pt():
    ckpt = '/ssd3/doyo/diffusion_v2/diffusion_v2/hybrid_wo_disc_hb1/last.ckpt'
    model = DiffusionHybridSolver.load_from_checkpoint(ckpt)
    print('loaded')


ckpt_to_pt()
