import torch
import numpy as np
import shutil
import jax
jax.config.update("jax_platform_name", "cpu")

from data_module.train_dataset import TrainDataset
from data_module.valid_dataset import PrerenderedValidDataset
from data_module.diffusion_dataset import DiffusionTrainDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.torch_utils import seed_everything

from degrad_operator.differentiable_render_grafx import DifferentiableRenderGrafx
from degrad_operator.render_grafx import RenderGrafx
from degrad_operator.graph_sampler import (
        GraphSampler,
        MonolithicGraph,
        SingleEffectGraph
        )
from degrad_operator.plot_graph import PlotGrafx
import soundfile as sf
import os; opj=os.path.join

def test_complex_graph():
    seed_everything(42)
    save_dir = '/ssd3/doyo/operator_learning/complex'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"Remove the previous valid_set in {save_dir}, and recreate")
    os.makedirs(save_dir, exist_ok=True)

    # Load Sample Audio
    audio, sr = sf.read('/ssd4/inverse_problem/speech/speech44k/daps_parsed/clean/f6_script4_clean_63_66.wav')
    graph_sampler = GraphSampler(default_chain_type = 'complex_graph',)
    renderer = RenderGrafx(sr=44100,
                           mono_processing=True,
                           output_format=np.array)
    graph_plotter = PlotGrafx()
    audio = {'speech' : audio}

    num_graphs = 100
    for i in tqdm(range(num_graphs)):
        G = graph_sampler()
        rendered = renderer(G, audio)
        sf.write(save_dir + f'/graph_{str(i).zfill(2)}.wav', rendered, 44100)
        graph_plotter.plot(G, plot_mode='default', name=save_dir + f'/graph_{str(i).zfill(2)}.pdf')


def test_train_dataset():
    seed_everything(42)
    ct = 'complex_graph'
    ds = TrainDataset(
        graph_type='complex',
        target_audio_len = 512*191,
        default_chain_type = ct,
        len_epoch=1000000
        )
    loader = DataLoader(ds)
    num_samples = 100
    save_dir = '/ssd3/doyo/operator_learning/complex'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        print(f"Remove the previous valid_set in {save_dir}, and recreate")
    os.makedirs(save_dir, exist_ok=True)
    for x, i in tqdm(zip(loader, range(num_samples))):
        dry_tar = x['dry_tar'].squeeze(0)
        wet_tar = x['wet_tar'].squeeze(0)
        afx_list = x['afx_name'][0]
        print(afx_list)

        sf.write(opj(save_dir, f'{i}_dry_{afx_list}.wav'), dry_tar.numpy(), 44100)
        sf.write(opj(save_dir, f'{i}_wet_{afx_list}.wav'), wet_tar.numpy(), 44100)
        sf.write(opj(save_dir, f'{i}_diff_{afx_list}.wav'), (dry_tar - wet_tar).numpy(), 44100)

def test_multiple_env():
    ds = TrainDataset(
        split_mode='multiple_recording_env',
        target_audio_len = 512*191,
        ref_audio_len = 512*191,
        )
    loader = DataLoader(ds)
    for x in tqdm(loader):
        pass

def test_valid_dataset():
    ds = PrerenderedValidDataset()
    loader = DataLoader(ds)
    for x in tqdm(loader):
        vt = x['valid_type']
        print(vt)

def diffusion_dataset():
    ds = DiffusionTrainDataset()
    loader = DataLoader(ds)
    for x in tqdm(loader):
        print(x['wet_spec'].shape)

def test_jax_autograd():
    wav, sr = sf.read('/ssd4/inverse_problem/speech/speech44k/daps_parsed/clean/f6_script4_clean_63_66.wav')
    graph_sampler = SingleEffectGraph(randomize_params=True,
                                      unseen_noise=True,
                                      unseen_ir=True,
                                      default_chain_type='valid_afx')
    renderer = DifferentiableRenderGrafx()
    G = graph_sampler()
    print(G)
    wav = torch.Tensor(wav)
    wav.requires_grad = True
    wav.retain_grad()
    wet = renderer(G, wav)
    print(wet)
    zero = torch.zeros_like(wet)
    mse = F.mse_loss(zero, wet, reduction='sum')
    print(mse)
    grad = torch.autograd.grad(outputs=mse, inputs=wet)[0]
    print(grad)


    wet = wav.detach().cpu().numpy()
    sf.write('/workspace/wet_signal.wav', wet, sr)

if __name__ == "__main__":
#     test_train_dataset()
#     test_valid_dataset()
#     test_multiple_env()
#     test_train_dataset()
#     diffusion_dataset()
#     test_jax_autograd()
   test_complex_graph() 
