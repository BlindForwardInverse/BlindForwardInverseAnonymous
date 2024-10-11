import os

import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

opj = os.path.join


def plot_dry_wet_pred(dry, wet, pred, name, input_type="wav", sr=None, save_dir="../samples"):
    assert input_type in ['mel', 'wav', 'path']
    if sr is None and input_type != 'mel':
        if not all(isinstance(var, str) for var in [dry, wet, pred]):
            raise ValueError(
                "sr is not given; all arguments must be mel spec or path of a wav file."
            )
        sr = librosa.get_samplerate(dry)

    # Load the audio data if the variables are file paths
    if input_type == "path":
        if isinstance(dry, str):
            dry, _ = librosa.load(dry)
        if isinstance(wet, str):
            wet, _ = librosa.load(wet)
        if isinstance(pred, str):
            pred, _ = librosa.load(pred)
    
    if input_type in ['wav', 'path']:
        # Compute Mel spectrograms
        dry_mel, wet_mel, pred_mel = [
            librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128, fmax=8000) 
            for wav in [dry, wet, pred]
        ]

        
        # Convert to decibels
        dry_mel, wet_mel, pred_mel = [
            librosa.power_to_db(mel, ref=np.max) for mel in [dry_mel, wet_mel, pred_mel]
        ]
    elif input_type == "mel":
        dry_mel, wet_mel, pred_mel = dry, wet, pred        

    # print(dry_mel.shape)
    # print(wet_mel.shape)
    # print(pred_mel.shape)
    
    #TODO fix
    dry_mel = dry_mel.cpu().numpy() if isinstance(dry_mel, torch.Tensor) else dry_mel
    wet_mel = wet_mel.cpu().numpy() if isinstance(wet_mel, torch.Tensor) else wet_mel
    pred_mel = pred_mel.cpu().numpy() if isinstance(pred_mel, torch.Tensor) else pred_mel

    # Create a figure with GridSpec layout
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])

    # Dry Mel Spectrogram
    ax0 = plt.subplot(gs[0])
    librosa.display.specshow(dry_mel, sr=sr, x_axis="time", y_axis="mel", fmax=8000, cmap='viridis') #
    plt.title("Dry")
    # plt.axis("off")

    # Wet Mel Spectrogram
    ax1 = plt.subplot(gs[1])
    librosa.display.specshow(wet_mel, sr=sr, x_axis="time", y_axis="mel", fmax=8000, cmap='viridis')
    plt.title("Wet")
    # plt.axis("off")

    # Pred Mel Spectrogram
    ax2 = plt.subplot(gs[2])
    img = librosa.display.specshow(pred_mel, sr=sr, x_axis="time", y_axis="mel", fmax=8000, cmap='viridis')
    plt.title("Pred")
    # plt.axis("off")

    # Colorbar
    ax3 = plt.subplot(gs[3])
    plt.colorbar(img, cax=ax3, format="%+2.0fdB")

    plt.tight_layout()
    plt.savefig(opj(save_dir, name))


if __name__ == "__main__":
    plot_dry_wet_pred(
        "../samples/bandpass_dry.wav",
        "../samples/bandpass_wet.wav",
        "../samples/bandpass_pred.wav",
        name="bandpass",
    )
