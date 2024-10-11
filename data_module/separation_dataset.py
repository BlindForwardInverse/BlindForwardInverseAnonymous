import torch
from torch.utils.data import Dataset
import numpy as nn
import soundfile as sf
import os

opj = os.path.join
import random

from omegaconf import OmegaConf
from glob import glob
from utils.audio_processing import audio_preprocessing
from utils.audio_transform import audio_transform


class SourceSepDataset(Dataset):
    def __init__(
        self,
        audio_len=45000,
        target_sr=16000,
        valid_len=48,
        transform="waveform",
        N=2,
    ):
        self.audio_len = audio_len
        self.target_sr = target_sr
        self.transform = transform
        self.valid_len = valid_len
        self.N = N

        self.data_config = OmegaConf.load("configs/valid_dataset.yaml")["speech"]
        self.data_path = list()
        self.load_data_path()

    def load_data_path(self):
        for dataset in self.data_config:
            for ext in ["wav", "flac"]:
                self.data_path += glob(opj(dataset, f"**/*.{ext}"), recursive=True)
        print(f"Number of Valid Data: {len(self.data_path)}")
        self.data_path = sorted(self.data_path)[: self.valid_len]

    def __len__(self):
        return self.valid_len

    def __getitem__(self, _):
        paths = random.sample(self.data_path, self.N)
        audio_list = []
        for path in paths:
            audio, sr = sf.read(path, dtype="float32")
            processed_audio = audio_preprocessing(
                audio, self.audio_len, sr, self.target_sr, crop_mode="front"
            )
            transformed_audio = audio_transform(processed_audio, self.transform)
            audio_list.append(transformed_audio)

        mixed = sum(audio_list) / self.N
        return mixed
