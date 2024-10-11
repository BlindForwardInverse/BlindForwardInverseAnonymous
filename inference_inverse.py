import torch
import torch.nn as nn
import numpy as np
import shutil
from matplotlib import pyplot as plt

import os
opj = os.path.join
import pandas as pd
from copy import deepcopy
import pickle

from functools import partial
from pprint import pprint

import soundfile as sf
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange
from torch.utils.data import DataLoader

from degrad_operator.differentiable_render_grafx import DifferentiableRenderGrafx
from degrad_operator.render_grafx import RenderGrafx
from degrad_operator.graph_sampler import (
        GraphSampler,
        MonolithicGraph,
        SingleEffectGraph
        )
from solvers.diffusion_solver_hybrid import DiffusionHybridSolver
from solvers.operator_solver_hybrid import OperatorSolverHybrid

from metric import EffectWiseSEMetricExtended

from data_module.valid_dataset import PrerenderedValidDataset

from diffusion_sampler.sde_solver import Heun2ndSampler
from diffusion_sampler.sde_solver_hybrid import Heun2ndSamplerHybrid
from diffusion_sampler.sde_class import ProbODEFlow
from diffusion_sampler.sde_class_hybrid import ProbODEFlowHybrid
from diffusion_sampler.particle_filtering import ParticleFiltering

from utils.audio_transform import audio_transform, spec_transform
from utils.audio_normalization import peak_normalize
from utils.torch_utils import seed_everything, load_pretrained_model

class SpeechEnhancement:
    def __init__(
        self,
        seed=42,
        device = 'cuda',
        sr = 44100,
        batch_size = 32,
        num_particles = 1,
        task = 'conditional',
        # Inference
        diffusion_config_path = 'configs/pretrained_model/diffusion_model.yaml',
        diffusion_config_name = 'hybrid_hb1',
        sampler_config_name = 'sampler',
        # Sampler
        save_dir = '/ssd3/doyo/inference_se',
        sampler_config_path = 'configs/sampler/diffusion_sampler.yaml',
        sampler_type = 'heun_2nd',
        **kwargs
    ):
        if seed is not None : seed_everything(seed)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            print("Remove the previous inference results")
        os.makedirs(save_dir, exist_ok=True)

        self.sr = sr
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.num_particles = num_particles

        # Load Diffusion Model
        self.diffusion_model = load_pretrained_model(solver_class=DiffusionHybridSolver,
                                                     config_path=diffusion_config_path,
                                                     config_name=diffusion_config_name,
                                                     freeze=True,
                                                     device='cuda',
                                                     )

        self.spec_to_wav = partial(spec_transform, transform_type='unpower_ri',
                                   n_fft=2046, hop_length=512, win_length=2046, length=512*191)
        self.wav_to_spec = partial(audio_transform, transform_type='power_ri',
                                   n_fft=2046, hop_length=512, win_length=2046,
                                  )

        sampler_configs = OmegaConf.to_object(OmegaConf.load(sampler_config_path))[sampler_type]
        self.sde = ProbODEFlowHybrid(f_theta = self.diffusion_model.model,
                                     weight = self.diffusion_model.weight,
                                     spec_to_wav = self.spec_to_wav,
                                     wav_to_spec = self.wav_to_spec,
                                     **sampler_configs)
        if task == 'conditional':
            self.sampler = Heun2ndSamplerHybrid(self.sde,
                                                **sampler_configs)

        elif task == 'particle_filter_known':
            self.sampler = Heun2ndSamplerHybrid(self.sde,
                                                **sampler_configs)
        elif task == 'particle_filter_unknown':
            self.sampler = Heun2ndSamplerHybrid(self.sde,
                                                **sampler_configs)
        elif task == 'smc_unknown':
            self.sampler = ParticleFiltering(self.sde, self.num_particles,
                                                **sampler_configs)
        elif task == 'smc_known':
            self.sampler = ParticleFiltering(self.sde, self.num_particles,
                                                **sampler_configs)
        
    def enhancement_conditional(self, test_set_path=None):
        loader, full_afx_types = load_test_set_loader(test_set_path, self.sr, batch_size=self.batch_size)
        model = self.diffusion_model

        full_afx_types = [self.map_effect_type(e) for e in full_afx_types]
        pred_metric = EffectWiseSEMetricExtended(full_afx_types) # Metric for (pred, wet)
        dry_metric = EffectWiseSEMetricExtended(full_afx_types) # Metric for (dry, wet)
        with model.ema.average_parameters():
            for batch in tqdm(iter(loader)):
                batch = self.to_cuda(batch)
                dry_spec, dry_wav, wet_spec, wet_wav = [batch[key] for key in\
                                                        ['dry_tar_spec', 'dry_tar', 'wet_tar_spec', 'wet_tar']]
                valid_type, afx_name, afx_types, audio_nums = [batch[key] for key in\
                                                            ['valid_type', 'afx_name', 'afx_type', 'audio_idx']]
                afx_types = [self.map_effect_type(e) for e in afx_types]


                # Enhancement
                global_condition, local_condition = model.get_condition(wet_wav)
                generated_audio = self.sampler(y=wet_spec, y_wav=wet_wav,
                                                t_embedding=global_condition, c_embedding=local_condition)
                generated_audio = generated_audio.detach().cpu()

                dry_wav = dry_wav.detach().cpu()
                wet_wav = wet_wav.detach().cpu()

                if torch.isnan(generated_audio).any():
                    print('NaN!')
                    print(afx_name)
                else:
                    pred_metric.update(generated_audio, dry_wav, afx_types)
                    dry_metric.update(wet_wav, dry_wav, afx_types)

                # Metric, Save
                for b, (_enhanced, _dry, _wet, _valid_type, _afx, _audio_num) in \
                    enumerate(zip(generated_audio, dry_wav, wet_wav, valid_type, afx_name, audio_nums)):
                    # Save the pred_wav
                    save_name = f"{_afx}-{_audio_num}-1dry.wav"
                    self.save_audio(save_name, _valid_type, _dry)

                    save_name = f"{_afx}-{_audio_num}-2wet.wav"
                    self.save_audio(save_name, _valid_type, _wet)

                    save_name = f"{_afx}-{_audio_num}-3enhanced.wav"
                    self.save_audio(save_name, _valid_type, _enhanced)

        # Summarize Metric
        pred_metric = pred_metric.compute()
        dry_metric = dry_metric.compute()
        
        self.export_to_csv(pred_metric, tag='pred')
        self.export_to_csv(dry_metric, tag='dry')

    @staticmethod
    def get_pickled_graph(graph_path):
        graphs = []
        for path in graph_path:
            with open(path, 'rb') as f:
                graph = pickle.load(f)
            graphs.append(graph)
        return graphs
        
    def particle_filter_known(self, test_set_path, effect_type='single'):
        loader, full_afx_types = load_test_set_loader(test_set_path, self.sr, batch_size=self.batch_size, exclude_codec=True, reduce_afx_type=False)
        model = self.diffusion_model
        graph_sampler = self.get_graph_sampler(effect_type)
        graph_renderer = DifferentiableRenderGrafx(sr=self.sr,
                                                   mono_processing=True,
                                                   output_format=torch.Tensor)
        graph_renderer_nondiff = RenderGrafx(sr=self.sr,
                                                   mono_processing=True,
                                                   output_format=torch.Tensor)

        full_afx_types = [self.map_effect_type(e) for e in full_afx_types]
        pred_metric = EffectWiseSEMetricExtended(full_afx_types) # Metric for (pred, wet)
        dry_metric = EffectWiseSEMetricExtended(full_afx_types) # Metric for (dry, wet)

        with model.ema.average_parameters():
            for batch in tqdm(iter(loader)):
                # On-the-Fly Render
                batch = self.to_cuda(batch)
                dry_spec, dry_wav, wet_spec, wet_wav = [batch[key] for key in\
                                                        ['dry_tar_spec', 'dry_tar', 'wet_tar_spec', 'wet_tar']]
                valid_type, afx_name, afx_types, audio_nums = [batch[key] for key in\
                                                            ['valid_type', 'afx_name', 'afx_type', 'audio_idx']]
                _afx_types = [self.afx_types_conversion(afx) for afx in afx_types]

                # graph
                graph_path = batch['graph_path']
                graph = self.get_pickled_graph(graph_path)
#                 dry_wav, wet_wav, dry_spec, wet_spec = self.to_cuda_list([dry_wav, wet_wav, dry_spec, wet_spec])

                # Enhancement
                afx_types = [self.map_effect_type(e) for e in afx_types]
                global_condition, local_condition = model.get_condition(wet_wav)
                generated_audio = self.sampler.forward_with_operator(
                                _operator=graph_renderer, graph=graph, y=wet_spec, y_wav=wet_wav,
                                t_embedding=global_condition, c_embedding=local_condition)
#                 generated_audio = self.sampler(y=wet_spec, y_wav=wet_wav,
#                                                 t_embedding=global_condition, c_embedding=local_condition)
                generated_audio = generated_audio.detach().cpu()

                dry_wav = dry_wav.detach().cpu()
                wet_wav = wet_wav.detach().cpu()

                if torch.isnan(generated_audio).any():
                    print('NaN!')
                    print(afx_name)
                else:
                    pred_metric.update(generated_audio, dry_wav, afx_types)
                    dry_metric.update(wet_wav, dry_wav, afx_types)

                # Metric, Save
                for b, (_enhanced, _dry, _wet, _valid_type, _afx, _audio_num) in \
                    enumerate(zip(generated_audio, dry_wav, wet_wav, valid_type, afx_name, audio_nums)):
                    # Save the pred_wav
                    save_name = f"{_afx}-{_audio_num}-1dry.wav"
                    self.save_audio(save_name, _valid_type, _dry)

                    save_name = f"{_afx}-{_audio_num}-2wet.wav"
                    self.save_audio(save_name, _valid_type, _wet)

                    save_name = f"{_afx}-{_audio_num}-3enhanced.wav"
                    self.save_audio(save_name, _valid_type, _enhanced)

        # Summarize Metric
        pred_metric = pred_metric.compute()
        dry_metric = dry_metric.compute()
        
        self.export_to_csv(pred_metric, tag='pred')
        self.export_to_csv(dry_metric, tag='dry')

    def particle_filter_unknown(self, test_set_path, effect_type='single'):
        loader, full_afx_types = load_test_set_loader(test_set_path, self.sr, batch_size=self.batch_size,
                                                      exclude_codec=False, reduce_afx_type=False)
        model = self.diffusion_model
        graph_sampler = self.get_graph_sampler(effect_type)
        graph_renderer = DifferentiableRenderGrafx(sr=self.sr,
                                                   mono_processing=True,
                                                   output_format=torch.Tensor)
        graph_renderer_nondiff = RenderGrafx(sr=self.sr,
                                                   mono_processing=True,
                                                   output_format=torch.Tensor)

        full_afx_types = [self.map_effect_type(e) for e in full_afx_types]
        pred_metric = EffectWiseSEMetricExtended(full_afx_types) # Metric for (pred, wet)
        dry_metric = EffectWiseSEMetricExtended(full_afx_types) # Metric for (dry, wet)

        with model.ema.average_parameters():
            def _operator(x, y_spec, y_wav, global_condition, local_condition):
                unet = model.pretrained_encoder.unet
                weight_net = model.pretrained_encoder.weight
                x_spec = self.wav_to_spec(x)

                spec_input = torch.cat([x_spec, y_spec], dim=-1)
                wav_input = torch.stack([x, y_wav], dim=1)
                
                pred_spec, pred_wav = unet(spec_input, wav_input, t=global_condition, condition=local_condition)
                weight = weight_net(global_condition)
                summed_wav = rearrange(weight[:, 0], 'b -> b 1') * pred_wav +\
                             rearrange(weight[:, 1], 'b -> b 1') * self.spec_to_wav(pred_spec)
                return summed_wav

            for batch in tqdm(iter(loader)):
                # On-the-Fly Render
                batch = self.to_cuda(batch)
                dry_spec, dry_wav, wet_spec, wet_wav = [batch[key] for key in\
                                                        ['dry_tar_spec', 'dry_tar', 'wet_tar_spec', 'wet_tar']]
                valid_type, afx_name, afx_types, audio_nums = [batch[key] for key in\
                                                            ['valid_type', 'afx_name', 'afx_type', 'audio_idx']]
                _afx_types = [self.afx_types_conversion(afx) for afx in afx_types]

                # Enhancement
                afx_types = [self.map_effect_type(e) for e in afx_types]
                global_condition, local_condition = model.get_condition(wet_wav)
                generated_audio = self.sampler.forward_with_unknown_operator(
                                _operator=_operator, y=wet_spec, y_wav=wet_wav,
                                t_embedding=global_condition, c_embedding=local_condition)
#                 generated_audio = self.sampler(y=wet_spec, y_wav=wet_wav,
#                                                 t_embedding=global_condition, c_embedding=local_condition)
                generated_audio = generated_audio.detach().cpu()

                dry_wav = dry_wav.detach().cpu()
                wet_wav = wet_wav.detach().cpu()

                if torch.isnan(generated_audio).any():
                    print('NaN!')
                    print(afx_name)
                else:
                    pred_metric.update(generated_audio, dry_wav, afx_types)
                    dry_metric.update(wet_wav, dry_wav, afx_types)

                # Metric, Save
                for b, (_enhanced, _dry, _wet, _valid_type, _afx, _audio_num) in \
                    enumerate(zip(generated_audio, dry_wav, wet_wav, valid_type, afx_name, audio_nums)):
                    # Save the pred_wav
                    save_name = f"{_afx}-{_audio_num}-1dry.wav"
                    self.save_audio(save_name, _valid_type, _dry)

                    save_name = f"{_afx}-{_audio_num}-2wet.wav"
                    self.save_audio(save_name, _valid_type, _wet)

                    save_name = f"{_afx}-{_audio_num}-3enhanced.wav"
                    self.save_audio(save_name, _valid_type, _enhanced)

        # Summarize Metric
        pred_metric = pred_metric.compute()
        dry_metric = dry_metric.compute()
        
        self.export_to_csv(pred_metric, tag='pred')
        self.export_to_csv(dry_metric, tag='dry')

    def smc_known(self, test_set_path, effect_type='single'):
        loader, full_afx_types = load_test_set_loader(test_set_path, self.sr, batch_size=self.batch_size,
                                                      exclude_codec=True, reduce_afx_type=False)
        model = self.diffusion_model
        graph_sampler = self.get_graph_sampler(effect_type)
        graph_renderer = DifferentiableRenderGrafx(sr=self.sr,
                                                   mono_processing=True,
                                                   output_format=torch.Tensor)

        full_afx_types = [self.map_effect_type(e) for e in full_afx_types]
        pred_metric = EffectWiseSEMetricExtended(full_afx_types) # Metric for (pred, wet)
        dry_metric = EffectWiseSEMetricExtended(full_afx_types) # Metric for (dry, wet)

        with model.ema.average_parameters():
            for batch in tqdm(iter(loader)):
                # On-the-Fly Render
                batch = self.to_cuda(batch)
                dry_spec, dry_wav, wet_spec, wet_wav = [batch[key] for key in\
                                                        ['dry_tar_spec', 'dry_tar', 'wet_tar_spec', 'wet_tar']]
                valid_type, afx_name, afx_types, audio_nums = [batch[key] for key in\
                                                            ['valid_type', 'afx_name', 'afx_type', 'audio_idx']]
                _afx_types = [self.afx_types_conversion(afx) for afx in afx_types]

                #graph
                graph_path = batch['graph_path']
                graph = self.get_pickled_graph(graph_path)
                
                # Enhancement
                afx_types = [self.map_effect_type(e) for e in afx_types]
                global_condition, local_condition = model.get_condition(wet_wav)
                generated_audio = self.sampler.forward_with_operator(
                                _operator=graph_renderer, graph=graph, y=wet_spec, y_wav=wet_wav,
                                t_embedding=global_condition, c_embedding=local_condition)
#                 generated_audio = self.sampler(y=wet_spec, y_wav=wet_wav,
#                                                 t_embedding=global_condition, c_embedding=local_condition)
                generated_audio = generated_audio.detach().cpu()

                dry_wav = dry_wav.detach().cpu()
                wet_wav = wet_wav.detach().cpu()

                if torch.isnan(generated_audio).any():
                    print('NaN!')
                    print(afx_name)
                else:
                    pred_metric.update(generated_audio, dry_wav, afx_types)
                    dry_metric.update(wet_wav, dry_wav, afx_types)

                # Metric, Save
                for b, (_enhanced, _dry, _wet, _valid_type, _afx, _audio_num) in \
                    enumerate(zip(generated_audio, dry_wav, wet_wav, valid_type, afx_name, audio_nums)):
                    # Save the pred_wav
                    save_name = f"{_afx}-{_audio_num}-1dry.wav"
                    self.save_audio(save_name, _valid_type, _dry)

                    save_name = f"{_afx}-{_audio_num}-2wet.wav"
                    self.save_audio(save_name, _valid_type, _wet)

                    save_name = f"{_afx}-{_audio_num}-3enhanced.wav"
                    self.save_audio(save_name, _valid_type, _enhanced)

        # Summarize Metric
        pred_metric = pred_metric.compute()
        dry_metric = dry_metric.compute()
        
        self.export_to_csv(pred_metric, tag='pred')
        self.export_to_csv(dry_metric, tag='dry')

    def smc_unknown(self, test_set_path, effect_type='single'):
        loader, full_afx_types = load_test_set_loader(test_set_path, self.sr, batch_size=self.batch_size,
                                                      exclude_codec=True, reduce_afx_type=False)
        model = self.diffusion_model
        graph_sampler = self.get_graph_sampler(effect_type)
        graph_renderer = DifferentiableRenderGrafx(sr=self.sr,
                                                   mono_processing=True,
                                                   output_format=torch.Tensor)
        graph_renderer_nondiff = RenderGrafx(sr=self.sr,
                                                   mono_processing=True,
                                                   output_format=torch.Tensor)

        full_afx_types = [self.map_effect_type(e) for e in full_afx_types]
        pred_metric = EffectWiseSEMetricExtended(full_afx_types) # Metric for (pred, wet)
        dry_metric = EffectWiseSEMetricExtended(full_afx_types) # Metric for (dry, wet)

        with model.ema.average_parameters():
            def _operator(x, y_spec, y_wav, global_condition, local_condition):
                unet = model.pretrained_encoder.unet
                weight_net = model.pretrained_encoder.weight
                x_spec = self.wav_to_spec(x)

                spec_input = torch.cat([x_spec, y_spec], dim=-1)
                wav_input = torch.stack([x, y_wav], dim=1)
                
                pred_spec, pred_wav = unet(spec_input, wav_input, t=global_condition, condition=local_condition)
                weight = weight_net(global_condition)
                summed_wav = rearrange(weight[:, 0], 'b -> b 1') * pred_wav +\
                             rearrange(weight[:, 1], 'b -> b 1') * self.spec_to_wav(pred_spec)
                return summed_wav

            for batch in tqdm(iter(loader)):
                # On-the-Fly Render
                batch = self.to_cuda(batch)
                dry_spec, dry_wav, wet_spec, wet_wav = [batch[key] for key in\
                                                        ['dry_tar_spec', 'dry_tar', 'wet_tar_spec', 'wet_tar']]
                valid_type, afx_name, afx_types, audio_nums = [batch[key] for key in\
                                                            ['valid_type', 'afx_name', 'afx_type', 'audio_idx']]
                _afx_types = [self.afx_types_conversion(afx) for afx in afx_types]

                # Enhancement
                afx_types = [self.map_effect_type(e) for e in afx_types]
                global_condition, local_condition = model.get_condition(wet_wav)
                generated_audio = self.sampler.forward_with_unknown_operator(
                                _operator=_operator, y=wet_spec, y_wav=wet_wav,
                                t_embedding=global_condition, c_embedding=local_condition)
#                 generated_audio = self.sampler(y=wet_spec, y_wav=wet_wav,
#                                                 t_embedding=global_condition, c_embedding=local_condition)
                generated_audio = generated_audio.detach().cpu()

                dry_wav = dry_wav.detach().cpu()
                wet_wav = wet_wav.detach().cpu()

                if torch.isnan(generated_audio).any():
                    print('NaN!')
                    print(afx_name)
                else:
                    pred_metric.update(generated_audio, dry_wav, afx_types)
                    dry_metric.update(wet_wav, dry_wav, afx_types)

                # Metric, Save
                for b, (_enhanced, _dry, _wet, _valid_type, _afx, _audio_num) in \
                    enumerate(zip(generated_audio, dry_wav, wet_wav, valid_type, afx_name, audio_nums)):
                    # Save the pred_wav
                    save_name = f"{_afx}-{_audio_num}-1dry.wav"
                    self.save_audio(save_name, _valid_type, _dry)

                    save_name = f"{_afx}-{_audio_num}-2wet.wav"
                    self.save_audio(save_name, _valid_type, _wet)

                    save_name = f"{_afx}-{_audio_num}-3enhanced.wav"
                    self.save_audio(save_name, _valid_type, _enhanced)

        # Summarize Metric
        pred_metric = pred_metric.compute()
        dry_metric = dry_metric.compute()
        
        self.export_to_csv(pred_metric, tag='pred')
        self.export_to_csv(dry_metric, tag='dry')

    def save_audio(self, name, valid_type, audio):
        audio = audio.detach().cpu().numpy()
        dir_name = opj(self.save_dir, valid_type)
        os.makedirs(dir_name, exist_ok=True)
        sf.write(opj(dir_name, name), audio, self.sr)

    def export_to_csv(self, metric, tag='inference'):
        df = pd.DataFrame(metric)
        print(df)
        csv_name = os.path.join(self.save_dir, f'metric_summary_{tag}.csv')
        df.to_csv(csv_name, index_label='Effect Type')

    @staticmethod
    def sample_graph(graph_sampler, afx_types):
        graph = []
        for afx in afx_types:
            G = graph_sampler(single_afx_name=afx)
            graph.append(G)
        return graph

    @staticmethod
    def get_graph_sampler(effect_type):
        if effect_type == 'single' :
            graph_sampler = SingleEffectGraph(randomize_params=True,
                                              unseen_noise=True,
                                              default_chain_type='valid_afx')
        elif effect_type == 'monolithic':
            graph_sampler = MonolithicGraph(randomize_params=True,
                                            min_chain_len=3,
                                            unseen_noise=True,
                                            default_chain_type='monolithic_graph')
        elif effect_type == 'complex':
            graph_sampler = GraphSampler(randomize_params=True,
                                         unseen_noise=True,
                                         default_chain_type='complex_graph')
        return graph_sampler

    @staticmethod
    def afx_types_conversion(afx):
        if 'noise' in afx:
            afx = 'add_noise'
        elif 'micir_conv' in afx:
            afx = 'micir_conv'
        elif 'rir_conv' in afx:
            afx = 'rir_conv'
        return afx

    @staticmethod
    def map_effect_type(effect):
        if 'rir_conv' in effect:
            return 'rir_conv'
        elif 'micir_conv' in effect:
            return 'micir_conv'
        elif 'pass_ladder' in effect:
            return effect.replace('_ladder', '')
        elif 'clip' in effect:
            return 'clipping'
        elif effect in ['svf', 'parametric_eq']:
            return 'EQ'
        elif effect in ['deesser', 'plosive', 'compressor']:
            return 'compressor'
        elif effect in ['aac', 'libmp3lame', 'libopus', 'libvorbis']:
            return 'codec'
        elif effect in ['chorus', 'tremolo', 'flanger']:
            return 'modulation'
        return effect

    @staticmethod
    def to_cuda(batch):
        for k, v in batch.items():
            batch[k] = v.to('cuda') if isinstance(v, torch.Tensor) else v
        return batch

    @staticmethod
    def to_cuda_list(batch):
        to_cuda = []
        for data in batch:
            if isinstance(data, torch.Tensor):
                to_cuda.append(data.detach().to('cuda'))
            else:
                to_cuda.append(data)
        return to_cuda

    @staticmethod
    def squeeze_batch(batch):
        for k, v in batch.items():
            batch[k] = v.squeeze(0) if isinstance(v, torch.Tensor) else v
        return batch

def load_test_set_loader(test_set_path, sr=44100, valid_set_types=['vctk1'], batch_size=12, exclude_codec=False, reduce_afx_type=True):
    dataset = PrerenderedValidDataset(
                modality='speech',
                target_sr=sr,
                tar_transform_type='power_ri',
                ref_transform_type='power_ri',
                n_fft=2046,
                win_length=2046,
                hop_length=512,
                prerendered_valid_dir=test_set_path,
                valid_set_types=valid_set_types,
                exclude_codec = exclude_codec,
                reduce_afx_type = reduce_afx_type,
                load_pickled_graph=True,
                )
    full_afx_types = dataset.full_afx_types
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        drop_last=False,
                        num_workers=32,
                        pin_memory=True,
                        persistent_workers=True)
    return loader, full_afx_types

if __name__ == "__main__":
    import argparse
    from argparse import ArgumentParser
    import jax
    jax.config.update("jax_platform_name", "cpu")

    parser = ArgumentParser()
    add_arg = parser.add_argument
    Boolean = argparse.BooleanOptionalAction

    add_arg("--task", default='conditional')
    add_arg("--seed", type=int, default=0)
    add_arg("--batch_size", type=int, default=8)

    add_arg("--diffusion_config_path", default='configs/pretrained_model/diffusion_model.yaml')
    add_arg("--diffusion_config_name", default='hybrid_hb1')
    add_arg("--sampler_type", default='heun_2nd_sigma1')
    add_arg("--num_particles", type=int, default=4)

    add_arg("--save_dir", default='/ssd3/doyo/inference_enhancement')

    add_arg("--test_set_path", default='/ssd4/doyo/test_set_tiny/single_effects')
    add_arg("--valid_set_types", nargs= '+', default= 'vctk1')

    args = vars(parser.parse_args())
    if args['task'] == 'conditional':
        infer = SpeechEnhancement(**args)
        infer.enhancement_conditional(test_set_path = args['test_set_path'])

    elif args['task'] == 'particle_filter_known':
        infer = SpeechEnhancement(**args)
        infer.particle_filter_known(test_set_path = args['test_set_path'])

    elif args['task'] == 'particle_filter_unknown':
        infer = SpeechEnhancement(**args)
        infer.particle_filter_unknown(test_set_path = args['test_set_path'])

    elif args['task'] == 'smc_known':
        infer = SpeechEnhancement(**args)
        infer.smc_known(test_set_path = args['test_set_path'])

    elif args['task'] == 'smc_unknown':
        infer = SpeechEnhancement(**args)
        infer.smc_unknown(test_set_path = args['test_set_path'])
