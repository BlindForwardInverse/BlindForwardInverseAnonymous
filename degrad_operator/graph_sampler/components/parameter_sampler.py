import os; opj=os.path.join
import random
from glob import glob

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from degrad_operator.grafx import Grafx
from utils.audio_processing import audio_processing, ir_preprocessing, rms_normalize_numpy
from .sample_functions import get_random_values

class ParameterSampler:
    def __init__(
        self,
        randomize_params=False,
        perceptual_intensity="default",
        target_sr = 44100,
        max_ir_seconds = 2.,
        max_noise_seconds = 3.,
        mono_processing = True,
        unseen_noise = False,
    ):
        self.randomize_params = randomize_params
        assert perceptual_intensity in [
            "default",
            "soft",
            "moderate",
            "hard",
        ], f"perceptual_intensity : {perceptual_intensity} not supported"
        self.perceptual_intensity = (
            perceptual_intensity  # 'default', 'soft', 'moderate', 'hard'
        )
        self.parameter_config = OmegaConf.load('configs/degrad_operator/afx_module_configs.yaml')

        # RIR, MicIR, Noise
        self.dataset_config = OmegaConf.load('configs/degrad_operator/noise_and_ir_data.yaml')
        self.rir_dataset = self.load_data_path('rir_dataset')
        self.micir_dataset = self.load_data_path('micir_dataset')
        self.noise_dataset = self.load_data_path('noise_dataset')

        self.target_sr = target_sr
        self.max_ir_seconds = max_ir_seconds
        self.max_noise_seconds = max_noise_seconds
        self.max_noise_len = int(target_sr * max_noise_seconds)
        self.mono_processing = mono_processing
        self.unseen_noise = unseen_noise

        threshold_db = -8.
        self.threshold_amp = np.power(10., threshold_db/20.)

    def __call__(self, 
                 G: Grafx,
                 micir_mode=None,
                 rir_mode=None,
                 noise_mode=None,
                 *args, **kwargs):
        nodes = G.nodes(data=True)
        for node in nodes:
            idx, data = node
            afx_type = data.get("afx_type")
            chain_type = data.get("chain")
            if self.parameter_config[afx_type]['class'] == 'controller':
                continue # Params for the controllers are performed at the modulation sampler
            else:
                if afx_type in ['rir', 'micir']:
                    data["parameters"] = self.sample_ir(afx_type, rir_mode, micir_mode)
                elif afx_type == 'noise':
                    data["parameters"] = self.sample_noise(noise_mode, **kwargs)
                else:
                    data["parameters"] = self.sample_parameters(afx_type, chain_type, **kwargs)

    def load_data_path(self, data_type):
        data_dir = self.dataset_config[data_type]
        data_path = []
        extentions = ["wav", "flac"]
        for ext in extentions:
            data_path += glob(opj(data_dir, f"**/*.{ext}"), recursive=True)
        print(f"Number of {data_type} : {len(data_path)}")
        data_path = sorted(data_path)
        return data_path

    def sample_parameters(self, afx_type, chain_type=None, perceptual_intensity=None, **kwargs):
        if perceptual_intensity is None:
            perceptual_intensity = self.perceptual_intensity
        param_config = self.parameter_config[afx_type]["parameters"]
        param_dict = dict()
        for param_type in param_config:
            config = param_config[param_type]
            if self.randomize_params:
                distribution = config["distribution"]
                sampling_range = config["sampling_range"]
                sampling_range = list(
                    sampling_range.get(
                        perceptual_intensity, sampling_range['default']
                    )
                )
                value = get_random_values(distribution, *sampling_range)
            else:
                fixed_values = config["fixed_value"]
                value = fixed_values.get(perceptual_intensity, fixed_values['default'])
            param_dict[param_type] = value
        return param_dict

    def sample_ir(self, afx_type, rir_mode=None, micir_mode=None):
        param_dict = dict()
        def get_ir_from_dataset(dataset):
            while True:
                ir_path = random.choice(dataset)
                ir, sr = sf.read(ir_path, dtype="float32")
                # Reject lower sr or too long IRs.
                if sr >= self.target_sr : break
            processed_ir = ir_preprocessing(
                                  ir,
                                  sr,
                                  target_sr=self.target_sr,
                                  max_ir_sec=self.max_ir_seconds,
                                  mono=self.mono_processing)
            return processed_ir

        def get_single_ir(ir_path):
            ir, sr = sf.read(ir_path, dtype="float32")
            processed_ir = ir_preprocessing(
                                  ir,
                                  sr,
                                  target_sr=self.target_sr,
                                  max_ir_sec=self.max_ir_seconds,
                                  mono=self.mono_processing)
            return processed_ir

        if self.randomize_params:
            if afx_type == "rir":
                ir = get_ir_from_dataset(self.rir_dataset)
                param_dict["rir_mode"] = 'random'
            elif afx_type == "micir":
                ir = get_ir_from_dataset(self.micir_dataset)
                param_dict["micir_mode"] = 'random'

        else:
            if afx_type == "rir":
                ir_path = self.parameter_config[afx_type]['parameters']['ir_path'].get(rir_mode)
                ir_path = opj(self.dataset_config['unseen_rir_dataset'], ir_path)
                param_dict["rir_mode"] = rir_mode

            elif afx_type == "micir":
                ir_path = self.parameter_config[afx_type]['parameters']['ir_path'].get(micir_mode)
                ir_path = opj(self.dataset_config['unseen_micir_dataset'], ir_path)
                param_dict["micir_mode"] = micir_mode
            ir = get_single_ir(ir_path)

        param_dict["ir"] = ir
        return param_dict
        
    def sample_noise(self, noise_mode='static_noise', noise_intensity=None, **kwargs):
        '''
        return : {'noise': xxxx, 'snr': xxxx, 'noise_mode' : noise_mode}
        '''
        param_dict = dict()
        def get_noise_from_dataset():
            if self.unseen_noise:
                noise_dir = self.dataset_config['unseen_noise_dataset']
            else:
                noise_dir = self.noise_dataset
            while True:
                noise_path = random.choice(self.noise_dataset)
                noise, sr = sf.read(noise_path, dtype="float32")
                processed_noise = audio_processing(noise,
                                                   self.max_noise_len,
                                                   sr,
                                                   self.target_sr,
                                                   mono=self.mono_processing,
                                                   crop_mode="front",
                                                   pad_mode="back",
                                                   pad_type="repeat",
                                                   rms_norm=False
                                                )
                if np.max(np.abs(processed_noise)) > self.threshold_amp : break
            processed_noise = rms_normalize_numpy(processed_noise)
            return processed_noise

        def get_single_noise(noise_path):
            noise, sr = sf.read(noise_path, dtype="float32")
            processed_noise = audio_processing(noise,
                                               self.max_noise_len,
                                               sr,
                                               self.target_sr,
                                               mono=self.mono_processing,
                                               crop_mode="front",
                                               pad_mode="back",
                                               pad_type="repeat",
                    )
            return processed_noise

        def sample_snr(noise_intensity):
            snr_config = self.parameter_config['noise']['parameters']['snr']
            distribution = snr_config['distribution']
            sampling_range = snr_config['sampling_range']
            sampling_range = (sampling_range[noise_intensity])
            snr = get_random_values(distribution, *sampling_range)
            return snr

        if noise_intensity is None:
            noise_intensity = self.perceptual_intensity

        if self.randomize_params:
            noise = get_noise_from_dataset()
            snr = sample_snr(noise_intensity)
            param_dict['noise_mode'] = 'random'

        else:
            noise_path = self.parameter_config['noise']['parameters']['noise_path'][noise_mode]
            noise_path = opj(self.dataset_config['unseen_noise_dataset'], noise_path)
            noise = get_single_noise(noise_path)

            snr = self.parameter_config['noise']['parameters']['snr']['fixed_value'][noise_intensity] # 10
            param_dict['noise_mode'] = noise_mode

        param_dict['noise'] = noise
        param_dict['snr'] = snr
        return param_dict
