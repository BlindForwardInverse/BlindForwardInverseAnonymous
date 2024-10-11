import torch
import torch.nn as nn
from torch.utils.data import Dataset

import numpy as np
import random

from utils.audio_transform import audio_transform

class DummyDataset(Dataset):
    def __init__(self,
                 transform_type='power_ri',
                 n_fft=2046,
                 win_length=2046,
                 hop_length=512,
                 audio_len=45000,
                 num_return=1):
        self.transform_type = transform_type
        self.audio_len = audio_len
        self.num_return = num_return

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def __len__(self):
        return self.num_return
    
    def __getitem__(self, _):
        dummy_audio = np.random.rand(self.audio_len)
        transformed = audio_transform(dummy_audio,
                                      n_fft=self.n_fft,
                                      hop_length=self.hop_length,
                                      win_length=self.win_length)
        return transformed
