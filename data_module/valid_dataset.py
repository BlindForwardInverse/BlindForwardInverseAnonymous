import torch
import torch.nn as nn

from glob import glob
from pprint import pprint
import os; opj=os.path.join
import re

import numpy as np
import soundfile as sf
import torchaudio
import torchaudio.transforms as T

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from utils.audio_processing import audio_processing, resample
from utils.audio_transform import audio_transform

class PrerenderedValidDataset(Dataset):
    def __init__(self,
                 modality='speech',
                 target_sr=44100,
                 tar_transform_type='power_ri',
                 ref_transform_type='power_ri',
                 single_afx_name=None,
                 # fft
                 n_fft=2046,
                 win_length=2046,
                 hop_length=512,
                 # subsample
                 num_subsample = None,
                 valid_set_types = ['vctk1'],
                 prerendered_valid_dir=None, # None : From yaml config
                 reduce_afx_type=True,
                 exclude_codec=False,
                 # load_graph
                 load_pickled_graph=False,
            ):
        self.target_sr = target_sr
        self.tar_transform_type = tar_transform_type
        self.ref_transform_type = ref_transform_type
        self.single_afx_name = single_afx_name
        self.n_fft=n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.valid_set_types = valid_set_types
        self.reduce_afx_type = reduce_afx_type
        self.exclude_codec = exclude_codec

        self.load_pickled_graph = load_pickled_graph
        print('valid set modality', modality)

        if modality == 'speech':
            if prerendered_valid_dir is not None : 
                self.valid_set_path = {v : opj(prerendered_valid_dir, v) for v in valid_set_types}
                pprint(self.valid_set_path)
            else:
                self.valid_set_path = OmegaConf.load("configs/data/data_configs.yaml").get(target_sr)['prerendered_valid']
                pprint(self.valid_set_path)

        elif modality == 'maestro':
            self.valid_set_path = OmegaConf.load('configs/data/data_configs_maestro.yaml').get(target_sr)['prerendered_valid']
            pprint(self.valid_set_path)

        elif modality == 'cross_modal':
            self.valid_set_path = OmegaConf.load('configs/data/data_configs_cross_modal.yaml').get(target_sr)['prerendered_valid']
            pprint(self.valid_set_path)

        self.data_path = self.load_data_path()
        self.key_list = []
        self.full_afx_types = []
        self.parsed_dict = self.parse_data_path()
        self.num_subsample = num_subsample

    def load_data_path(self):
        data_path = dict()
        for valid_type in self.valid_set_path:
            valid_path = self.valid_set_path[valid_type]
            if valid_type in self.valid_set_types:
                paths = sorted(glob(opj(valid_path, '**/*.wav'), recursive=True))
                data_path[valid_type] = paths
        return data_path

    def parse_data_path(self):
        parsed_dict = dict() # valid_type - key - audio_type : path
        for valid_type, paths in self.data_path.items():
            parsed_dict[valid_type] = dict()
            for path in paths:
                filename = path.split('/')[-1]
                key = '-'.join(filename.split('-')[:3]) # bandpass-default-0
                afx_type = key.split('-')[0]
                if self.exclude_codec and afx_type in ['aac', 'libopus', 'libvorbis', 'libmp3lame', 'deesser', 'plosive', 'compressor', 'limiter']: continue
                if self.reduce_afx_type: afx_type = self.reduce(afx_type)
                if afx_type not in self.full_afx_types:
                    self.full_afx_types.append(afx_type)
                audio_type = filename.split('-')[3][:-4] # dry_tar

                if key not in parsed_dict[valid_type]:
                    parsed_dict[valid_type][key] = dict()
                    self.key_list.append((valid_type, key))
                parsed_dict[valid_type][key][audio_type] = path
        return parsed_dict

    def __len__(self):
        if self.num_subsample is not None:
            total_len = self.num_subsample
        else:
            total_len = len(self.key_list)
        return total_len

    @staticmethod
    def read_info_from_key(key):
        # ex) bandpass-default-0
        afx_type, perceptual_intensity, audio_idx = key.split('-')
        return afx_type, perceptual_intensity, int(audio_idx)

    def __getitem__(self, idx):
        valid_type, key = self.key_list[idx]
        afx_type, mode, audio_idx = self.read_info_from_key(key)
        afx_name = afx_type + "_" + mode if mode != '_' else afx_type
        if self.reduce_afx_type: afx_type = self.reduce(afx_type)

        audio_paths = self.parsed_dict[valid_type][key]
        audio_dict = {}
        for audio_type in ['dry_tar', 'wet_tar', 'dry_ref', 'wet_ref']:
            audio, sr = sf.read(audio_paths[audio_type], dtype = "float32")
            if 'tar' in audio_type:
                transform_type = self.tar_transform_type
            elif 'ref' in audio_type:
                transform_type = self.ref_transform_type
            transformed = audio_transform(audio, transform_type=transform_type,
                                          n_fft=self.n_fft, win_length = self.win_length, hop_length = self.hop_length)
            audio_dict[audio_type] = audio
            audio_dict[audio_type + '_spec'] = transformed
            
        item_dict = {
            'dry_tar' : audio_dict['dry_tar'],
            'wet_tar' : audio_dict['wet_tar'],
            'dry_tar_spec' : audio_dict['dry_tar_spec'],
            'wet_tar_spec' : audio_dict['wet_tar_spec'],
            'dry_ref' : audio_dict['dry_ref'],
            'wet_ref' : audio_dict['wet_ref'],
            'wet_ref_spec' : audio_dict['wet_ref_spec'],
            'valid_type' : valid_type,
            'afx_name' : afx_name,
            'afx_type' : afx_type,
            'audio_idx' : audio_idx,
        }

        if self.load_pickled_graph:
            item_dict['graph_path'] = '/'.join(audio_paths['dry_tar'].split('/')[:-1]) + f'/{key}-graph.pkl'
        return item_dict

    @staticmethod
    def reduce(afx_type):
        if 'monolithic' in afx_type:
            afx_type = 'monolithic'
        elif 'complex' in afx_type:
            afx_type = 'complex'
        elif 'noise_' in afx_type:
            afx_type = 'noise'
        return afx_type
