import torch
import torch.nn as nn
import numpy as np
import shutil
from matplotlib import pyplot as plt

import os
opj = os.path.join
import pandas as pd
from copy import deepcopy

from functools import partial
from pprint import pprint

import soundfile as sf
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from solver_diffusion import EDMSpecSolver
from solver_operator import UnetSolver
from metric import MetricHandlerOperator, SpeechEnhancementMetric, ReferenceFreeMetric

from data_module.valid_dataset import PrerenderedValidDataset

from diffusion_sampler.sde_solver import Heun2ndSampler
from diffusion_sampler.sde_class import ProbODEFlow

from utils.audio_transform import spec_transform
from utils.audio_normalization import peak_normalize
from utils.tensor_utils import seed_everything

from sklearn.manifold import TSNE

class SpeechEnhancement:
    def __init__(
        self,
        seed=42,
        device = 'cuda',
        sr = 44100,
        batch_size = 16,
        # Inference
        diffusion_config_name = 'diffusion',
        operator_config_name = 'operator_w_mrd_single',
        sampler_config_name = 'sampler',
        # Override sampler
        save_dir = '/ssd3/doyo/se_inference',
        config_dir = 'configs/pretrained_model',
        sampler_dir = 'configs/diffusion_sampler',
        metric_types = ['pesq', 'stoi', 'sisdr', 'dnsmos'],
        # Operator
        learned_operator = False,
        deterministic = False,
        **kwargs
    ):
        if seed is not None : seed_everything(seed)
        self.config_dir = config_dir
        self.sampler_dir = sampler_dir
        self.save_dir = save_dir
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            print("Remove the previous inference results")
        os.makedirs(self.save_dir, exist_ok=True)
        self.device = device
        self.sr = sr
        self.batch_size = batch_size

        # Load diffusion model
        diffusion_configs = load_config(opj(config_dir, 'diffusion'), diffusion_config_name)
        ckpt, model_config = diffusion_configs['ckpt'], diffusion_configs['model_params']
        diffusion_solver = EDMSpecSolver(**model_config)
        pretrained_solver = load_pretrained_model(diffusion_solver, ckpt)
        for p in pretrained_solver.parameters():
            p.requires_grad = False
        self.pretrained_diffusion = pretrained_solver.model

        # Diffusion Sampler
        sampler_configs = load_config(sampler_dir, sampler_config_name)
        self.sde = ProbODEFlow(self.pretrained_diffusion, **sampler_configs)
        self.sampler = Heun2ndSampler(sde=self.sde,
                                      save_intermediate=False,
                                      print_progress_bar=True,
                                      **sampler_configs)

        # Load Pretrained Operator
        if learned_operator:
            operator_configs = load_config(opj(config_dir, 'operator'), operator_config_name)
            ckpt, model_config = operator_configs['ckpt'], operator_configs['model_params']
            operator_solver = UnetSolver(**model_config)
            self.pretrained_operator = load_pretrained_model(operator_solver, ckpt)
            for p in self.pretrained_operator.parameters():
                p.requires_grad = False

    def enhancement(self):
        dataset = PrerenderedValidDataset(
                    target_sr=self.sr,
                    tar_transform_type='power_ri',
                    ref_transform_type='waveform',
                    n_fft=2046,
                    win_length=2046,
                    hop_length=512,
                    )
        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            drop_last=False,
                            num_workers=16,
                            pin_memory=True,
                            persistent_workers=True)
        operator = self.pretrained_operator
        metric_fn = SpeechEnhancementMetric()
        dns = ReferenceFreeMetric()

        # Metric initialize
        metric_types = self.metric_types
        metric_history = {metric : dict() for metric in metric_types} 
        mixture_history = {metric : dict() for metric in metric_types}
        with operator.ema.average_parameters():
            for batch in tqdm(iter(loader)):
                for k, v in batch.items():
                    batch[k] = v.to('cuda') if isinstance(v, torch.Tensor) else v
                dry_ref, wet_ref, wet_ref_spec = batch['dry_ref'], batch['wet_ref'], batch['wet_ref_spec']
                valid_type, afx, audio_idx = batch['valid_type'], batch['afx'], batch['audio_idx']

                # Enhancement
                fx_latent = operator.fx_encoder(wet_ref)
                t = operator.to_fx_embedding(fx_latent)
                c = operator.to_afx_tokens(fx_latent)
                forward_operator = partial(operator.unet, t=t, condition=c)
                enhanced = self.sampler(wet_ref_spec.shape, operator=forward_operator, y=wet_ref_spec)
                enhanced_audio= spec_transform(enhanced, transform_type='unpower_ri',
                                               n_fft=2046, win_length=2046, hop_length=512)

                # Metric, Save
                for b, (valid_type, afx, audio_num) in enumerate(zip(valid_type, afx, audio_idx)):
                    audio = enhanced_audio[b, :]
                    audio = peak_normalize(audio).detach().cpu().numpy()
                    save_dir_name = opj(self.save_dir, valid_type, afx)
                    os.makedirs(save_dir_name, exist_ok=True)
                    save_name = opj(save_dir_name, f'{afx}_{audio_num}')

                    degrad_audio = wet_ref[b, :].detach().cpu().numpy()
                    clean_audio = dry_ref[b, :].detach().cpu().numpy()
                    sf.write(save_name + '_enhanced.wav', audio, self.sr)
                    sf.write(save_name + '_degrad.wav', degrad_audio, self.sr)
                    sf.write(save_name + '_clean.wav', clean_audio, self.sr)
        
    def enhancement_with_known_operator(self):
        pass

    def unconditional_generation(self,
                                 num_audio=4,
                                 plot_audio=True,
                                 shape=(1024, 192, 2)):
        full_shape = (num_audio, *shape)
        x_0 = self.sampler(full_shape, operator=None, y=None)
        generated_wav = spec_transform(x_0, transform_type='unpower_ri',
                                       n_fft=2046, win_length=2046, hop_length=512)
        for b in range(num_audio):
            audio = generated_wav[b, :]
            audio = peak_normalize(audio)
            audio = audio.detach().cpu().numpy()
            name = 'uncond_' + str(b)
            os.makedirs(opj(self.save_dir, 'unconditional'), exist_ok=True)
            save_name = opj(self.save_dir, 'unconditional', name + '.wav')
            sf.write(save_name, audio, self.sr)

class OperatorLearning:
    def __init__(
        self,
        # Inference
        seed=42,
        sr=44100,
        batch_size=32,
        config_dir = 'configs/pretrained_model',
        # Args
        save_dir = '/ssd3/doyo/operator_learning',
        operator_config_name = 'operator_w_mrd_single',
        test_set_path = None,
        valid_set_types = None,
        metric_types = None,
        # tsne
        before_training = False,
        override=False,
        **kwargs
        ):
        if seed is not None : seed_everything(seed)
        if os.path.exists(save_dir) and override:
            shutil.rmtree(save_dir)
            print("Remove the previous inference results")

        os.makedirs(save_dir, exist_ok=True)
        self.config_dir = config_dir
        self.sr = sr
        self.save_dir = save_dir
        self.batch_size = batch_size

        self.test_set_path = test_set_path
        self.valid_set_types = valid_set_types
        self.metric_types = metric_types

        # Load Operator
        operator_configs = load_config(config_dir, operator_config_name)
        ckpt, model_config = operator_configs['ckpt'], operator_configs['model_params']
        operator_solver = UnetSolver(**model_config)
        if before_training:
            self.pretrained_operator = operator_solver.to('cuda')
        else:
            self.pretrained_operator = load_pretrained_model(operator_solver, ckpt)
            for p in self.pretrained_operator.parameters():
                p.requires_grad = False


    def forward_operator_learning(self):
        dataset = PrerenderedValidDataset(
                    target_sr=self.sr,
                    tar_transform_type='power_ri',
                    ref_transform_type='waveform',
                    n_fft=2046,
                    win_length=2046,
                    hop_length=512,
                    prerendered_valid_dir=self.test_set_path,
                    valid_set_types=self.valid_set_types
                    )
        loader = DataLoader(dataset,
                            batch_size=self.batch_size,
                            drop_last=False,
                            num_workers=24,
                            pin_memory=True,
                            persistent_workers=True)
        operator = self.pretrained_operator
        metric_fn = MetricHandlerOperator(self.metric_types).to('cuda')

        # Metric initialize
        valid_types = self.valid_set_types
        metric_types = self.metric_types

        metric_history = {valid_type : {metric : dict() for metric in metric_types} for valid_type in valid_types}
        mixture_history = {valid_type : {metric : dict() for metric in metric_types} for valid_type in valid_types}
        metric_std = {valid_type : {metric : dict() for metric in metric_types} for valid_type in valid_types}
        mixture_std = {valid_type : {metric : dict() for metric in metric_types} for valid_type in valid_types}
        
        with operator.ema.average_parameters():
            for batch in tqdm(iter(loader)):
                for k, v in batch.items():
                    batch[k] = v.to('cuda') if isinstance(v, torch.Tensor) else v
                dry_tar, wet_tar, dry_ref, wet_ref = [batch[key] for key in \
                                                      ["dry_tar", "wet_tar", "dry_ref", "wet_ref"]]
                wet_ref_spec = batch["wet_ref_spec"]
                wet_ref_24k = None
                valid_types, afxs, audio_nums = [batch[key] for key in ["valid_type", "afx", "audio_idx"]]

                # Valid Loss
                pred_tar = operator(dry_tar, wet_ref, wet_ref_24k, wet_ref_spec)

                # Measure Metric
                transform = partial(spec_transform, transform_type='unpower_ri',
                                n_fft=2046, hop_length=512, win_length=2046,
                                length=512*191)

                pred_wav = transform(pred_tar)
                dry_wav = transform(dry_tar)
                wet_wav = transform(wet_tar)

                for i, (valid_type, afx, audio_num) in enumerate(zip(valid_types, afxs, audio_nums)):
                    pred, dry, wet = pred_wav[i, ...].detach(), dry_wav[i,...].detach(), wet_wav[i, ...].detach()
                    ref = wet_ref[i, ...].cpu()
                    metric = metric_fn(pred, wet)
                    dry_metric = metric_fn(dry, wet)

                    for metric_type in metric:
                        if afx not in metric_history[valid_type][metric_type]:
                            metric_history[valid_type][metric_type][afx] = [metric[metric_type].detach().cpu().numpy()]
                            mixture_history[valid_type][metric_type][afx] = [dry_metric[metric_type].detach().cpu().numpy()]

                        else:
                            metric_history[valid_type][metric_type][afx].append(metric[metric_type].detach().cpu().numpy())
                            mixture_history[valid_type][metric_type][afx].append(dry_metric[metric_type].detach().cpu().numpy())

                    # Save the pred_wav
                    pred = pred.cpu().numpy()
                    save_name = f"{afx}-{audio_num}-pred.wav"
                    self.save_audio(save_name, valid_type, pred)
                    
                    wet = wet.cpu().numpy()
                    save_name = f"{afx}-{audio_num}-wet.wav"
                    self.save_audio(save_name, valid_type, wet)

                    save_name = f"{afx}-{audio_num}-wet_ref.wav"
                    self.save_audio(save_name, valid_type, ref)

            # Summarize Metric
            metric_history_summarized, metric_std_summarized= self.summarize_history(metric_history, metric_std)
            mixture_history_summarized, mixture_std_summarized= self.summarize_history(mixture_history, mixture_std)

            # Save in csv and plot
            self.export_to_csv(metric_history_summarized, tag='eval')
            self.export_to_csv(metric_std_summarized, tag='eval_std')
            self.export_to_csv(mixture_history_summarized, tag='mixture')
            self.export_to_csv(mixture_std_summarized, tag='mixture_std')

    def summarize_history(self, metric_history, std_history=None):
        for valid_type in self.valid_set_types:
            for metric_type in metric_history[valid_type]:
                grouped_history = group_afx(metric_history[valid_type][metric_type])
                metric_history[valid_type][metric_type] = grouped_history
                afx_list = [afx for afx in grouped_history]

                for afx in afx_list:
                    if len(metric_history[valid_type][metric_type][afx]) > 0 :
                        avg_metric, std_metric = calc_avg_std(metric_history[valid_type][metric_type][afx])
                        metric_history[valid_type][metric_type][afx] = avg_metric
                        std_history[valid_type][metric_type][afx] = std_metric
                    else:
                        del metric_history[valid_type][metric_type][afx]

        print("---------------")
        pprint(metric_history)
        pprint(std_history)
        print("---------------")
        self.export_to_csv(metric_history)
        self.export_to_csv(std_history)
        return metric_history, std_history

    def save_audio(self, name, valid_type, audio):
        dir_name = opj(self.save_dir, valid_type)
        os.makedirs(dir_name, exist_ok=True)
        sf.write(opj(dir_name, name), audio, self.sr)

    """
    ---------------------------------------------------------
    Util Methods
    ---------------------------------------------------------
    """
    def export_to_csv(self, summary, tag='inference'):
        for valid_type in self.valid_set_types:
            df = pd.DataFrame(summary[valid_type])
            csv_name = opj(self.save_dir, f'summary_{valid_type}_{tag}.csv')
            df.to_csv(csv_name, index_label='Metric')

    def tsne_global_embedding(self, filename='tsne.pdf',
                              valid_set_types=['in_dataset_same_mic']):
        tsne = TSNE(n_components=2, random_state=0)
        dataset = PrerenderedValidDataset(
                    target_sr=self.sr,
                    tar_transform_type='power_ri',
                    ref_transform_type='waveform',
                    n_fft=2046,
                    win_length=2046,
                    hop_length=512,
                    prerendered_valid_dir=self.test_set_path,
                    valid_set_types=valid_set_types
                    )
        loader = DataLoader(dataset,
                            batch_size=6,
                            drop_last=False,
                            num_workers=12,
                            pin_memory=True,
                            persistent_workers=True)
        operator = self.pretrained_operator
        fx_encoder = operator.fx_encoder
        global_embedder = operator.to_fx_embedding

        embeddings = []
        afx_names = []
        with operator.ema.average_parameters():
            for batch in tqdm(iter(loader)):
                wet_ref = batch['wet_ref'].to('cuda')
                afx_name = batch['afx']
                fx_latent = fx_encoder(wet_ref)
                global_embedding = global_embedder(fx_latent)
                emb_np = global_embedding.detach().cpu().numpy() # (B, 512)
                afx_names += afx_name 
                for b in emb_np:
                    embeddings.append(b)
        cluster = tsne.fit_transform(np.array(embeddings))

        plt.figure(figsize=(4,4), )
        afx_list = list(set(afx_names))
        label_to_int = {label: idx for idx, label in enumerate(afx_list)}
        label_mapping = [label_to_int[label] for label in afx_names]

        scatter = plt.scatter(cluster[:,0], cluster[:,1], s=0.3, c=label_mapping, cmap='turbo', label=afx_names)
        plt.xlim([-100,100])
        plt.ylim([-100,100])

        plt.savefig(opj(self.save_dir, filename), bbox_inches='tight', pad_inches=0)

def calc_avg_std(history):
    mean = np.mean(np.array(history))
    std = np.std(np.array(history))
#     std = 2.807 * std / np.sqrt(len(history))
#     mean = sum(history) / len(history)
#     std = np.sqrt(sum([(h - mean) ** 2 for h in history]) / len(history))
#     mean, std = round(mean, 3), round(std, 3)
    return mean, std
                
def group_afx(history):
    groups = ['lowpass', 'bandpass', 'highpass', 'bandreject', 'distortion', 'clip', 'mono_reverb', 'rir_conv', 'micir_conv', 'noise_hard', 'noise_moderate', 'noise_soft', 'monolithic', 'complex']
    grouped_dict = {group : [] for group in groups}
    for afx, v in history.items():
        for group in groups:
            if group in afx: grouped_dict[group] += v
            elif 'noise' in afx and 'hard' in afx: grouped_dict['noise_hard'] += v
            elif 'noise' in afx and 'moderate' in afx: grouped_dict['noise_moderate'] += v
            elif 'noise' in afx and 'soft' in afx: grouped_dict['noise_soft'] += v
    return grouped_dict

def load_config(config_dir,  config_file):
    config_name = opj(config_dir,  config_file + '.yaml')
    configs = OmegaConf.to_object(OmegaConf.load(config_name))
    return configs

def load_pretrained_model(solver, ckpt, device='cuda', load_ema=True):
    '''
    Load checkpoint (.ckpt) for pl.LightningModule
    '''
    print("=================================")
    print("Pretrained Model is Loading...")
    pretrained_module = solver.load_from_checkpoint(ckpt)
    pretrained_module.eval()
    pretrained_module.to(device)
    print("Trained Model Imported")
    print("=================================")
    return pretrained_module

if __name__ == "__main__":
    import argparse
    from argparse import ArgumentParser

    parser = ArgumentParser()
    add_arg = parser.add_argument
    Boolean = argparse.BooleanOptionalAction

    add_arg("--task", default='operator_learning')
    add_arg("--save_dir", default='/ssd3/doyo/inference')
    add_arg("--batch_size", type=int, default=32)
    add_arg("--operator_config_name", default='operator_w_disc_multiple_ch64')
    add_arg("--test_set_path", default='/ssd4/inverse_problem/test_set_single_effect_multiple_env')
    add_arg("--valid_set_types", nargs= '+', default=['in_dataset_same_mic', 'in_dataset_diff_mic', 'out_dataset'])
    add_arg("--metric_types", nargs= '+', default=['si_sdr', 'si_snr', 'mss', 'mag'])

    add_arg("--learned_operator", default=False, action=Boolean)
    add_arg("--diffusion_config_name", default='diff_cond')
    add_arg("--sampler_config_name", default='sampler')
    add_arg("--deterministic", default=False, action=Boolean)

    args = vars(parser.parse_args())
    if args['task'] == 'operator_learning':
        inference = OperatorLearning(**args)
        inference.forward_operator_learning()

    elif args['task'] == 'tsne':
        inference = OperatorLearning(**args)
        inference.tsne_global_embedding(filename='maestro.pdf',
                                        valid_set_types=['maestro_single'])

    elif args['task'] == 'se':
        infer = SpeechEnhancement(**args)
        infer.enhancement()
    else:
        print("Not Implemented")

