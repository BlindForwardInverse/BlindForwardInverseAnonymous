import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.speech_dataset import DiffusionTrainDataset

def test_dataset():
    ds = DiffusionTrainDataset(audio_len=22*128*32-1)
    loader = DataLoader(ds)
    for b in loader:
        print(torch.max(b), torch.min(torch.abs(b)))

if __name__ == "__main__":
    test_dataset()
    # test_dataset()
    #     test_model()
    #     test_dag()
#     test_dataset()
#     test_specds()
