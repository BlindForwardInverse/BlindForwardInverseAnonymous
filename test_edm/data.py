import torch
from torch.utils.data import Dataset
import numpy as np
import random
import os;opj=os.path.join

from omegaconf import OmegaConf
from glob import glob
from PIL import Image
import torchvision.transforms as transforms

class AFHQDataset(Dataset):
    def __init__(self):
        self.data_path = []
        self.transform = transforms.Compose([
                transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.load_data_path()
        print(len(self.data_path))

    def load_data_path(self):
        data_dir = '/ssd3/doyo/afhq/val'
        self.data_path = glob(opj(data_dir, f'**/*.jpg'), recursive=True)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, i):
        path = self.data_path[i]
        image = Image.open(path)
        image = image.resize((64, 64))
        image = self.transform(image)
        image = 2 * image - 1
        return image
