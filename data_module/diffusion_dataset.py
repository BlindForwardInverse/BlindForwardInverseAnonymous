import torch
from torch.utils.data import Dataset
import numpy as np
import soundfile as sf
import os;opj=os.path.join
import random

import jax
import jax.numpy as jnp

from omegaconf import OmegaConf
from glob import glob

from utils.audio_processing import audio_processing, resample, rms_normalize_numpy
from utils.audio_transform import audio_transform
from .data_split import (
        split_single_recording_env,
        split_multiple_recording_env,
        )
from degrad_operator.render_grafx import RenderGrafx
from degrad_operator.graph_sampler import (
        GraphSampler,
        MonolithicGraph,
        SingleEffectGraph
        )

class DiffusionTrainDataset(Dataset):
    def __init__(
        self,
        split_mode="single_recording_env",
        single_env_mic="mic1",
        # Signal Info
        audio_len = 512*191,
        transform_type = "power_ri",
        target_sr = 44100,
        threshold_db = -8.,
        mono = True,
        n_fft=2046,
        win_length=2046,
        hop_length=512,
        len_epoch = 1000,
        # Conditional Training
        return_wet=True,
        default_chain_type='full_afx',
        graph_type="single_effect",
        perceptual_intensity="default",
        randomize_params=True,
        single_afx_name=None,
        **kwargs
        ):

        self.split_mode = split_mode
        self.single_env_mic = single_env_mic
        self.audio_len = audio_len
        self.transform_type = transform_type
        self.target_sr = target_sr
        self.threshold_db = threshold_db
        self.threshold_amp = np.power(10., threshold_db/20.)
        self.mono = mono
        self.len_epoch = len_epoch

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        self.return_wet = return_wet

        # Load Data Path
        data_config = OmegaConf.load("configs/data/data_configs.yaml")
        data_dirs = data_config[self.target_sr]['train'][self.split_mode]
        if split_mode == 'single_recording_env':
            self.data_paths, _ = split_single_recording_env(data_dirs, single_env_mic)
        elif split_mode == 'multiple_recording_env':
            single_data_dir = data_config[self.target_sr]['train']['single_recording_env']
            self.environment_list, self.split_environment = split_multiple_recording_env(data_dirs, single_env_dir=single_data_dir)
            num_target = 0
            num_ref = 0
            print(f"The number of Recording env : {len(self.environment_list)}")
            for env in self.split_environment:
                num_target += len(self.split_environment[env]['target'])
                num_ref += len(self.split_environment[env]['ref'])
            print(f"The number of trainset audio : {num_ref}")
        else:
            raise Exception(f"{split_mode} is undefined")

        # Set graph sampler
        self.graph_type = graph_type
        self.single_afx_name = single_afx_name
        if self.graph_type == 'single_effect':
            self.graph_sampler = SingleEffectGraph(default_chain_type = default_chain_type,
                                                   **kwargs)
        elif self.graph_type == 'monolithic':
            self.graph_sampler = MonolithicGraph(default_chain_type = 'monolithic_graph',
                                                 **kwargs)
        elif self.graph_type == 'complex':
            self.graph_sampler = GraphSampler(default_chain_type = 'complex_graph',
                                              **kwargs)
        else:
            raise Exception(f"{graph_type} is undefined")

        self.renderer = RenderGrafx(sr=target_sr,
                                    mono_processing=self.mono,
                                    output_format=np.array)
    def __len__(self):
        return self.len_epoch

    def __getitem__(self, _):
        if self.split_mode == 'single_recording_env':
            dataset = self.data_paths
        elif self.split_mode == 'multiple_recording_env':
            env = random.choice(self.environment_list)
            dataset = self.split_environment[env]['ref']
        
        # Sample Dry signal
        while True:
            path = random.choice(dataset)
            audio, sr = sf.read(path, dtype='float32')
            if sr < self.target_sr : continue
            processed_audio = audio_processing(
                audio, self.audio_len, sr, self.target_sr, mono=self.mono, rms_norm=False, crop_mode='adaptive'
            )
            # Reject this cropped audio if silent
            if np.max(np.abs(processed_audio)) > self.threshold_amp: break
        processed_audio = rms_normalize_numpy(processed_audio)
        
        # Render Wet Signal
        G = self.graph_sampler(single_afx_name=self.single_afx_name) 
        dry, wet = self.render_audio(G, processed_audio)

        # Transformed
        transformed_audio = audio_transform(processed_audio, self.transform_type,
                                            n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)
        transformed_wet = audio_transform(wet, self.transform_type,
                                        n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length)

        item_dict = {'dry_spec' : transformed_audio,
                     'dry_wav'  : processed_audio,
                     'wet_spec' : transformed_wet,
                     'wet_wav' : wet}
        return item_dict

    def render_audio(self, G, dry):
        dry_jnp = jnp.array(dry)
        input_signal = {"speech": dry_jnp}
        wet = self.renderer(G, input_signal)
        dry = self.post_processing(dry)
        wet = self.post_processing(wet)
        return dry, wet

    def post_processing(self, audio):
        audio = audio / np.max(np.abs(audio)) # scale audio in [-1, 1]
        return audio
