import os
import random
from glob import glob

import jax
import jax.numpy as jnp
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from utils.audio_processing import audio_processing, resample, rms_normalize_numpy
from utils.audio_transform import audio_transform

from .data_split import (
        split_single_recording_env,
        split_multiple_recording_env,
        split_maestro,
        )

from degrad_operator.render_grafx import RenderGrafx
from degrad_operator.graph_sampler import (
        GraphSampler,
        MonolithicGraph,
        SingleEffectGraph
        )

opj = os.path.join

class TrainDataset(Dataset):
    def __init__(
        self,
        split_mode="single_recording_env",
        single_env_mic="mic1",
        return_ref_spec=False,
        # Signal Info
        modality = 'speech',
        target_sr=44100,
        target_audio_len=1023*128,
        ref_audio_len=1023*128,
        target_transform_type="power_ri",
        ref_transform_type="waveform",
        #fft
        n_fft=2046,
        win_length=2046,
        hop_length=512,
        # preprocessing
        threshold_db = -8.,
        mono=True,
        len_epoch=1000,
        num_data=None,
        # Graph_type
        default_chain_type='full_afx',
        graph_type="single_effect",
        perceptual_intensity="default",
        randomize_params=True,
        single_afx_name=None,
        **kwargs
    ):
        self.split_mode = split_mode
        self.single_env_mic = single_env_mic
        self.return_ref_spec = return_ref_spec

        self.modality = modality
        self.target_sr = target_sr
        self.target_audio_len = target_audio_len
        self.ref_audio_len= ref_audio_len
        self.target_transform_type = target_transform_type
        self.ref_transform_type = ref_transform_type

        self.threshold_amp = np.power(10., threshold_db/20.)
        self.mono = mono
        self.len_epoch = len_epoch
        self.num_data = num_data

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        # Load Data Path
        if modality == 'speech':
            data_config = OmegaConf.load("configs/data/data_configs.yaml")
            data_dirs = data_config[self.target_sr]['train'][self.split_mode]
            if split_mode == 'single_recording_env':
                self.target_audio_paths, self.ref_audio_paths = split_single_recording_env(data_dirs, single_env_mic)
                num_target = len(self.target_audio_paths)
                num_ref = len(self.ref_audio_paths)

            elif split_mode == 'multiple_recording_env':
                single_data_dir = data_config[self.target_sr]['train']['single_recording_env']
                self.environment_list, self.split_environment = split_multiple_recording_env(data_dirs, single_env_dir=single_data_dir)
                num_target = 0
                num_ref = 0
                print(f"The number of Recording env : {len(self.environment_list)}")
                for env in self.split_environment:
                    num_target += len(self.split_environment[env]['target'])
                    num_ref += len(self.split_environment[env]['ref'])
            else:
                raise Exception(f"{split_mode} is undefined")

            print(f"The number of target audio : {num_target}")
            print(f"The number of reference audio : {num_ref}")

        elif modality == 'maestro':
            data_dir = OmegaConf.load("configs/data/data_configs_maestro.yaml")[self.target_sr]['train']['maestro_single']
            self.target_audio_paths, self.ref_audio_paths = split_maestro(data_dir)

        elif modality == 'cross_modal':
            data_config = OmegaConf.load("configs/data/data_configs.yaml")
            data_dirs = data_config[self.target_sr]['train'][self.split_mode]
            self.target_audio_paths, _ = split_single_recording_env(data_dirs, single_env_mic)

            data_dir = OmegaConf.load("configs/data/data_configs_maestro.yaml")[self.target_sr]['train']['maestro_single']
            _, self.ref_audio_paths = split_maestro(data_dir)

        # Set graph sampler
        self.graph_type = graph_type
        self.single_afx_name = single_afx_name
        if self.graph_type == 'single_effect':
            self.graph_sampler = SingleEffectGraph(default_chain_type = default_chain_type, **kwargs)
        elif self.graph_type == 'monolithic':
            self.graph_sampler = MonolithicGraph(default_chain_type = 'monolithic_graph', **kwargs)
        elif self.graph_type == 'complex':
            self.graph_sampler = GraphSampler(default_chain_type = 'complex_graph', **kwargs)
        else:
            raise Exception(f"{graph_type} is undefined")

        self.renderer = RenderGrafx(sr=target_sr,
                                    mono_processing=self.mono,
                                    output_format=np.array)

    def __len__(self):
        return self.len_epoch

    def __getitem__(self, _):
        # Render Audio with G
        # ---------------------------------------------------------------------------------
        G = self.graph_sampler(single_afx_name=self.single_afx_name) 
        if self.modality == 'speech':
            if self.split_mode == 'single_recording_env':
                target_dataset = self.target_audio_paths
                ref_dataset = self.ref_audio_paths
            elif self.split_mode == 'multiple_recording_env':
                env = random.choice(self.environment_list)
                target_dataset = self.split_environment[env]['target']
                ref_dataset = self.split_environment[env]['ref']

        elif self.modality == 'maestro':
            target_dataset = self.target_audio_paths
            ref_dataset = self.ref_audio_paths

        elif self.modality == 'cross_modal':
            target_dataset = self.target_audio_paths
            ref_dataset = self.ref_audio_paths

        dry_tar = self.sample_from_dataset(target_dataset, audio_len=self.target_audio_len)
        dry_ref = self.sample_from_dataset(ref_dataset, audio_len=self.ref_audio_len)

        dry_tar, wet_tar = self.render_audio(G, dry_tar)
        dry_ref, wet_ref = self.render_audio(G, dry_ref)

        # Transform
        # ---------------------------------------------------------------------------------
        dry_tar_spec, wet_tar_spec = [audio_transform(audio, transform_type=self.target_transform_type,
                                        n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)\
                            for audio in [dry_tar, wet_tar]]
        dry_ref, wet_ref = [audio_transform(audio, transform_type=self.ref_transform_type,
                                        n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)\
                            for audio in [dry_ref, wet_ref]]

        afx_name = list(G.nodes())[1]
        item_dict = {
                "dry_tar": dry_tar,
                "wet_tar": wet_tar,
                "dry_tar_spec" : dry_tar_spec,
                "wet_tar_spec" : wet_tar_spec,
                "dry_ref": dry_ref,
                "wet_ref": wet_ref,
                "afx_name": afx_name
            }
        if self.return_ref_spec:
            wet_ref_spec = audio_transform(wet_ref, transform_type=self.target_transform_type,
                                            n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
            item_dict["wet_ref_spec"] = wet_ref_spec
        return item_dict

    def render_audio(self, G, dry):
        dry_jnp = jnp.array(dry)
        input_signal = {"speech": dry_jnp}
        wet = self.renderer(G, input_signal)
        dry = self.post_processing(dry)
        wet = self.post_processing(wet)
        return dry, wet

    def sample_from_dataset(self, dataset, audio_len):
        while True:
            path = random.choice(dataset)
            audio, sr = sf.read(path, dtype="float32")
            if sr < self.target_sr : continue
            processed_audio = audio_processing(
                audio, audio_len, sr, self.target_sr, mono=self.mono, rms_norm=False
            )
            # Reject this cropped audio if silent
            if np.max(np.abs(processed_audio)) > self.threshold_amp: break
        processed_audio = rms_normalize_numpy(processed_audio)
        return processed_audio

    def post_processing(self, audio):
        audio = audio / np.max(np.abs(audio)) # scale audio in [-1, 1]
        return audio

    @staticmethod
    def to_mono(audio):
        return np.mean(audio, axis=-1)
