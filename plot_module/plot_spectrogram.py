import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
from librosa.display import specshow

def plot_audio(audio, save_name, sr=44100, hop_length=512, win_length=2048):
    fig, ax = plt.subplots()
    spec = np.abs(librosa.stft(audio))
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    img = specshow(spec, sr=sr, hop_length=hop_length, win_length=win_length, x_axis='time',
                   y_axis='log', ax=ax, cmap='jet')
    title = save_name.split('/')[-1].split('.')[0]
    fig.suptitle(title)
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    fig.savefig(save_name)
    plt.close()
