import os
import random
from glob import glob
from tqdm import tqdm
from pprint import pprint
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset

from omegaconf import OmegaConf

from degrad_operator.render_grafx import RenderGrafx
from degrad_operator.graph_sampler import (
        GraphSampler,
        MonolithicGraph,
        SingleEffectGraph
        )
from degrad_operator.plot_graph import PlotGrafx
from .data_split import (
        split_single_recording_env,
        split_daps,
        split_maestro
    )
from utils.audio_processing import audio_processing
from utils.audio_transform import audio_transform

opj = os.path.join

class RenderValidDataset:
    def __init__(self,
        num_audio_per_afx=5,
        target_sr=44100,
        target_audio_len=1023*128,
        ref_audio_len=1023*128,
        target_transform_type="waveform",
        ref_transform_type="waveform",
        mono=True,
        threshold_db = -8.,
        # Single afx
        randomize_params=False,
        single_afx_name=None, # Set None for generating whole afx
        single_mode=False,
        add_noise=False,
        add_noise_level = 0.1,
        # Complex afx
        num_monolithic_graphs = 10,
        num_complex_graphs = 10,
        # Save
        save_dir = '/ssd4/inverse_problem/valid_set',
        pickle_graph = False,
    ):
        self.num_audio_per_afx = num_audio_per_afx
        self.target_sr = target_sr
        self.target_audio_len = target_audio_len
        self.ref_audio_len = ref_audio_len
        self.target_transform_type = target_transform_type
        self.ref_transform_type = ref_transform_type
        self.mono = mono
        self.threshold_amp = np.power(10., threshold_db/20.)
        
        self.randomize_params = randomize_params
        self.single_afx_name = single_afx_name
        self.add_noise = add_noise

        self.num_monolithic_graphs = num_monolithic_graphs
        self.num_complex_graphs = num_complex_graphs
        self.save_dir = save_dir
        self.pickle_graph = pickle_graph

        # Load Data Path
        split_paths = dict()
        data_config = OmegaConf.load("configs/data/data_configs.yaml")[self.target_sr]['valid']

        # vctk1, vctk2
        for recording_env, mic in zip(["vctk1", "vctk2"], ["mic1", "mic2"]):
            data_dirs = data_config[recording_env]
            target, ref = split_single_recording_env(data_dirs, mic, shuffle=False)
            split_paths[recording_env] = dict()
            split_paths[recording_env]['target'] = target
            split_paths[recording_env]['ref'] = ref

        # daps
        target, ref = split_daps(data_config['daps'], shuffle=False)
        split_paths['daps'] = dict()
        split_paths['daps']['target'] = target
        split_paths['daps']['ref'] = ref

        self.split_paths = split_paths

        # Set graph sampler
        self.renderer = RenderGrafx(sr=target_sr,
                                    mono_processing=self.mono,
                                    output_format=np.array)
        self.plot_graph = PlotGrafx()
        self.ir_modes = {'rir_conv' : ['concert_hall', 'church', 'studio_1', 'studio_2', 'bathroom'],
                         'micir_conv' : ['vintage_1', 'vintage_2', 'modern_1', 'modern_2'],}
        self.noise_modes = ['plosive', 'static_wind', 'static_noise', 'guitar', 'clap', 'piano', 'car']

        if single_mode:
            self.ir_modes = {'rir_conv' : ['random', 'random2'],
                             'micir_conv' : ['random', 'random2'],}
            self.noise_modes = ['random', 'random2', 'random3']

    def render_cross_modal(self, render_effect_types=['single', 'monolithic', 'complex']):
        # Sample Maestro
        recording_envs = 'maestro_single'
        data_dirs = OmegaConf.load("configs/data/data_configs_maestro.yaml")[self.target_sr]['valid'][recording_envs]
        dry_audio_dict = {recording_envs : dict()}

        target, ref = split_maestro(data_dirs, shuffle=True)
        dry_refs = self.sample_from_dataset(ref, audio_len=self.ref_audio_len, num_samples=self.num_audio_per_afx)

        # Sample Dry Audio
        target_dataset = self.split_paths['vctk1']['target']
        dry_tars = self.sample_from_dataset(target_dataset,
                                            audio_len=self.target_audio_len,
                                            num_samples=self.num_audio_per_afx)

        # Tar and Ref
        dry_audio_dict[recording_envs]['dry_tars'] = dry_tars
        dry_audio_dict[recording_envs]['dry_refs'] = dry_refs

        recording_envs = ['maestro_single']
        # Main Render
        for effect_type in render_effect_types:
            if effect_type == 'single':
                graph_sampler = SingleEffectGraph(randomize_params=self.randomize_params,
                                                  unseen_noise=True,
                                                  unseen_ir=True,
                                                  default_chain_type='valid_afx')
                self.render_single_effect(dry_audio_dict, graph_sampler, recording_envs)
            elif effect_type == 'monolithic':
                graph_sampler = MonolithicGraph(randomize_params=True,
                                                min_chain_len=2,
                                                max_chain_len=5,
                                                unseen_noise=True,
                                                unseen_ir=True,
                                                default_chain_type='monolithic_graph')
                self.render_monolithic_effect(dry_audio_dict, graph_sampler, recording_envs)
            elif effect_type == 'complex':
                graph_sampler = GraphSampler(randomize_params=True,
                                             unseen_noise=True,
                                             unseen_ir=True,
                                             default_chain_type='complex_graph')
                self.render_complex_effect(dry_audio_dict, graph_sampler, recording_envs)
            else:
                raise Exception("Not supported effect_type")

    def render_non_speech(self, render_effect_types=['single', 'monolithic', 'complex']):
        recording_envs = 'maestro_single'
        data_dirs = OmegaConf.load("configs/data/data_configs_maestro.yaml")[self.target_sr]['valid'][recording_envs]
        dry_audio_dict = {recording_envs : dict()}

        target, ref = split_maestro(data_dirs, shuffle=True)
        dry_tars = self.sample_from_dataset(target, audio_len=self.target_audio_len, num_samples=self.num_audio_per_afx)
        dry_refs = self.sample_from_dataset(ref, audio_len=self.ref_audio_len, num_samples=self.num_audio_per_afx)

        dry_audio_dict[recording_envs]['dry_tars'] = dry_tars
        dry_audio_dict[recording_envs]['dry_refs'] = dry_refs

        recording_envs = ['maestro_single']
        # Main Render
        for effect_type in render_effect_types:
            if effect_type == 'single':
                graph_sampler = SingleEffectGraph(randomize_params=self.randomize_params,
                                                  unseen_noise=True,
                                                  unseen_ir=True,
                                                  default_chain_type='valid_afx')
                self.render_single_effect(dry_audio_dict, graph_sampler, recording_envs)
            elif effect_type == 'monolithic':
                graph_sampler = MonolithicGraph(randomize_params=True,
                                                min_chain_len=3,
                                                unseen_noise=True,
                                                unseen_ir=True,
                                                default_chain_type='monolithic_graph')
                self.render_monolithic_effect(dry_audio_dict, graph_sampler, recording_envs)
            elif effect_type == 'complex':
                graph_sampler = GraphSampler(randomize_params=True,
                                             unseen_noise=True,
                                             unseen_ir=True,
                                             default_chain_type='complex_graph')
                self.render_complex_effect(dry_audio_dict, graph_sampler, recording_envs)
            else:
                raise Exception("Not supported effect_type")

    def render_valid_set(self,
                         recording_envs=['vctk1, vctk2, daps'],
                         render_effect_types=['single', 'monolithic', 'complex'],
                         ):
        # Sample Dry Audio
        dry_audio_dict = dict()
        for recording_env in recording_envs:
            target_dataset = self.split_paths[recording_env]['target']
            ref_dataset = self.split_paths[recording_env]['ref']
            dry_tars = self.sample_from_dataset(target_dataset,
                                                audio_len=self.target_audio_len,
                                                num_samples=self.num_audio_per_afx)
            dry_refs = self.sample_from_dataset(ref_dataset,
                                                audio_len=self.ref_audio_len,
                                                num_samples=self.num_audio_per_afx)
            dry_audio_dict[recording_env] = dict()
            dry_audio_dict[recording_env]['dry_tars'] = dry_tars
            dry_audio_dict[recording_env]['dry_refs'] = dry_refs

        # Main Render
        for effect_type in render_effect_types:
            if effect_type == 'single':
                graph_sampler = SingleEffectGraph(randomize_params=self.randomize_params,
                                                  unseen_noise=True,
                                                  unseen_ir=True,
                                                  default_chain_type='valid_afx')
                self.render_single_effect(dry_audio_dict, graph_sampler, recording_envs)
            elif effect_type == 'monolithic':
                graph_sampler = MonolithicGraph(randomize_params=True,
                                                min_chain_len=3,
                                                unseen_noise=True,
                                                unseen_ir=True,
                                                default_chain_type='monolithic_graph')
                self.render_monolithic_effect(dry_audio_dict, graph_sampler, recording_envs)
            elif effect_type == 'complex':
                graph_sampler = GraphSampler(randomize_params=True,
                                             unseen_noise=True,
                                             unseen_ir=True,
                                             default_chain_type='complex_graph')
                self.render_complex_effect(dry_audio_dict, graph_sampler, recording_envs)
            else:
                raise Exception("Not supported effect_type")

    def render_single_effect(self, dry_audio_dict, graph_sampler, recording_envs):
        """Render Single AFX
        Args:
            - dry_audio_dict (dict) : recording_env - [dry_tars / dry_refs] - (list) audios
            - graph_sampler (Class)
            - recording_envs (list) : [vctk1 / vctk2 / daps]
        """
        print("------------------------------------")
        print("Rendering Single Effects")
        print("------------------------------------")
        afx_to_render = self.single_afx_name if self.single_afx_name is not None\
                        else graph_sampler.get_supporting_afx()
        print("AFX TO RENDER :")
        pprint(afx_to_render)

        for recording_env in recording_envs:
            dry_tars = dry_audio_dict[recording_env]['dry_tars']
            dry_refs = dry_audio_dict[recording_env]['dry_refs']
            save_path = opj(self.save_dir, recording_env)
            # Sample AFX Graph 
            for afx_name in tqdm(afx_to_render):
                if afx_name in ['rir_conv', 'micir_conv']:
                    for mode in self.ir_modes[afx_name]:
                        dir_name = opj(save_path, afx_name + '_' + mode)
                        os.makedirs(dir_name, exist_ok=True)

                        # Sampler Graph
                        if not self.randomize_params:
                            if afx_name == 'rir_conv': G = graph_sampler(single_afx_name=afx_name, rir_mode=mode)
                            elif afx_name == 'micir_conv': G = graph_sampler(single_afx_name=afx_name, micir_mode=mode)
                        # Render
                        for i, (dry_tar, dry_ref) in enumerate(zip(dry_tars, dry_refs)):
                            if self.randomize_params:
                                if afx_name == 'rir_conv': G = graph_sampler(single_afx_name=afx_name)
                                elif afx_name == 'micir_conv': G = graph_sampler(single_afx_name=afx_name)
                                if self.pickle_graph: self.save_graph(G, dir_name, afx_name+'_'+mode, '_', i)
                            self.render_and_save(dry_tar, dry_ref, G, dir_name, afx_name+'_'+mode, '_', i)

                elif afx_name == 'add_noise':
                    for mode in self.noise_modes:
                        dir_name = opj(save_path, afx_name + '_' + mode)
                        os.makedirs(dir_name, exist_ok=True)
                        for noise_intensity in ['soft', 'moderate', 'hard']:
                            if not self.randomize_params:
                                # Sampler Graph
                                G = graph_sampler(single_afx_name = afx_name,
                                                  noise_mode = mode,
                                                  noise_intensity = noise_intensity)
                            # Render
                            for i, (dry_tar, dry_ref) in enumerate(zip(dry_tars, dry_refs)):
                                if self.randomize_params:
                                    G = graph_sampler(single_afx_name = afx_name,)
                                    if self.pickle_graph: self.save_graph(G, dir_name, afx_name+'_'+mode, noise_intensity, i)
                                self.render_and_save(dry_tar, dry_ref, G, dir_name, afx_name+'_'+ mode, noise_intensity, i)
                else:
                    dir_name = opj(save_path, afx_name)
                    os.makedirs(dir_name, exist_ok=True)
                    modes = ['soft', 'moderate', 'hard']
                    for mode in modes:
                        if not self.randomize_params:
                            G = graph_sampler(single_afx_name=afx_name, perceptual_intensity=mode)
                        for i, (dry_tar, dry_ref) in enumerate(zip(dry_tars, dry_refs)):
                            if self.randomize_params:
                                G = graph_sampler(single_afx_name=afx_name, perceptual_intensity=mode)
                                if self.pickle_graph: self.save_graph(G, dir_name, afx_name, mode, i)
                            self.render_and_save(dry_tar, dry_ref, G, dir_name, afx_name, mode, i)

    def render_monolithic_effect(self, dry_audio_dict, graph_sampler, recording_envs):
        print("------------------------------------")
        print("Rendering Monolithic Effects")
        print("------------------------------------")
        for recording_env in recording_envs:
            dry_tars = dry_audio_dict[recording_env]['dry_tars']
            dry_refs = dry_audio_dict[recording_env]['dry_refs']
            save_path = opj(self.save_dir, recording_env)
            # Sample AFX Graph
            for i in tqdm(range(self.num_monolithic_graphs)):
                dir_name = opj(save_path, f'monolithic_{i}')
                os.makedirs(dir_name, exist_ok=True)
                G = graph_sampler()
                self.plot_graph.plot(
                    G, plot_mode="big", name=opj(dir_name, f"monolithic_{i}.pdf")
                )
                self.plot_graph.plot(
                    G, plot_mode="default", name=opj(dir_name, f"monolithic_{i}.pdf")
                )
                for audio_idx, (dry_tar, dry_ref) in enumerate(zip(dry_tars, dry_refs)):
                    self.render_and_save(dry_tar, dry_ref, G, dir_name, f"monolithic_{i}", '_', audio_idx)

    def render_complex_effect(self, dry_audio_dict, graph_sampler, recording_envs):
        print("------------------------------------")
        print("Rendering Complex Effects")
        print("------------------------------------")
        for recording_env in recording_envs:
            dry_tars = dry_audio_dict[recording_env]['dry_tars']
            dry_refs = dry_audio_dict[recording_env]['dry_refs']
            save_path = opj(self.save_dir, recording_env)
            # Sample AFX Graph
            for i in tqdm(range(self.num_complex_graphs)):
                dir_name = opj(save_path, f'complex_{i}')
                os.makedirs(dir_name, exist_ok=True)
                G = graph_sampler()
                self.plot_graph.plot(
                    G, plot_mode="big", name=opj(dir_name, f"complex_{i}.pdf")
                )
                self.plot_graph.plot(
                    G, plot_mode="default", name=opj(dir_name, f"complex_{i}_default.pdf")
                )
                for audio_idx, (dry_tar, dry_ref) in enumerate(zip(dry_tars, dry_refs)):
                    self.render_and_save(dry_tar, dry_ref, G, dir_name, f"complex_{i}", '_', audio_idx)
                    
    def render_audio(self, G, dry):
        dry_jnp = jnp.array(dry)
        input_signal = {"speech": dry_jnp}
        wet = self.renderer(G, input_signal)
        dry = self.post_processing(dry)
        wet = self.post_processing(wet)
        return dry, wet

    def render_and_save(self, dry_tar, dry_ref, G, dir_name, afx_name, mode, i):
        dry_tar, wet_tar = self.render_audio(G, dry_tar)
        dry_ref, wet_ref = self.render_audio(G, dry_ref)
        sf.write(opj(dir_name, f"{afx_name}-{mode}-{i}-dry_tar.wav"), dry_tar, self.target_sr)
        sf.write(opj(dir_name, f"{afx_name}-{mode}-{i}-wet_tar.wav"), wet_tar, self.target_sr)
        sf.write(opj(dir_name, f"{afx_name}-{mode}-{i}-dry_ref.wav"), dry_ref, self.target_sr)
        sf.write(opj(dir_name, f"{afx_name}-{mode}-{i}-wet_ref.wav"), wet_ref, self.target_sr)

    def save_graph(self, G, dir_name, afx_name, mode, i):
        save_name = opj(dir_name, f"{afx_name}-{mode}-{i}-graph.pkl")
        with open(save_name, 'wb') as f:
            pickle.dump(G, f)

    def sample_from_dataset(self, dataset, audio_len, num_samples=1):
        sampled_audio = []
        idx = 0
        while len(sampled_audio) < num_samples:
            path = dataset[idx]
            audio, sr = sf.read(path, dtype="float32")
            processed_audio = audio_processing(
                audio, audio_len, sr, self.target_sr, mono=self.mono, crop_mode="adaptive", pad_mode="random"
            )
            if np.max(np.abs(processed_audio)) > self.threshold_amp:
                sampled_audio.append(processed_audio)
            idx += 1
        return sampled_audio

    def post_processing(self, audio):
        audio = audio / np.max(np.abs(audio))
        return audio

    '''
    Audio Utils
    '''
    def save_audio(self, afx, name, audio):
        dir_name = opj(self.save_dir, afx)
        os.makedirs(dir_name, exist_ok=True)
        save_name = opj(dir_name, name)
        save = sf.write(save_name, audio, self.target_sr)

    @staticmethod
    def to_mono(audio):
        return np.mean(audio, axis=-1)
