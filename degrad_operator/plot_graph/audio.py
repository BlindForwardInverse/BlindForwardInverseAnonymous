"""
audio plot tool
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from afx.primitives import rms_normalize

spec_config = {"diff": ("bwr", -20, 20), "spec": ("jet", -60, 20)}


def plot_spectrogram(x, y, y_hat=None):
    def get_mono(x):
        if x.ndim == 2:
            if x.shape[-1] == 2:
                return np.array(rms_normalize(np.mean(x, -1)))
            else:
                return x[:, 0]
        else:
            return x

    def get_spec(x):
        X = librosa.stft(x, n_fft=2048, hop_length=512)
        return 20 * np.log10(np.abs(X) + 1e-7)

    def single_spec(X, ax, mode="spec", title=""):
        cmap, vmin, vmax = spec_config[mode]
        img = librosa.display.specshow(
            X,
            hop_length=512,
            x_axis="time",
            y_axis="log",
            sr=FLAGS.sr,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.text(
            0.08,
            16000,
            title,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", linewidth=0.7, facecolor="wheat", alpha=1),
        )
        ax.set_yticks([250, 1000, 4000, 16000], minor=False)
        ax.set_yticklabels(["250", "1k", "4k", "16k"], minor=False)
        ax.set_yticks([], minor=True)
        ax.label_outer()
        ax.set_ylim(125, FLAGS.sr / 2)
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        return img

    if isinstance(y_hat, np.ndarray):
        X, Y, Y_HAT = map(lambda x: get_spec(get_mono(x)), (x, y, y_hat))

        fig, ax = plt.subplots(3, 2, sharex=True, sharey=True)
        img = single_spec(X, ax[0, 0], "spec", "dry")
        single_spec(Y - Y_HAT, ax[0, 1], "diff", "wet/wet_pred")
        single_spec(Y, ax[1, 0], "spec", "wet")
        img2 = single_spec(Y - X, ax[1, 1], "diff", "wet/dry")
        single_spec(Y_HAT, ax[2, 0], "spec", "wet_pred")
        single_spec(Y_HAT - X, ax[2, 1], "diff", "wet_pred/dry")

        fig.subplots_adjust(hspace=0, wspace=0.1)

        fig.colorbar(img, ax=ax[:, 0], format="%+2.f", aspect=50)
        fig.colorbar(img2, ax=ax[:, 1], format="%+2.f", aspect=50)
        fig.set_size_inches(8, 5)
        return fig
    else:
        X, Y = map(lambda x: get_spec(get_mono(x)), (x, y))

        fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
        img = single_spec(X, ax[0, 0], "spec", "dry")
        single_spec(Y, ax[1, 0], "spec", "wet")
        img2 = single_spec(Y - X, ax[1, 1], "diff", "wet/dry")
        fig.subplots_adjust(hspace=0, wspace=0.1)
        ax[0, 1].axis("off")

        fig.colorbar(img, ax=ax[:, 0], format="%+2.f", aspect=50)
        fig.colorbar(img2, ax=ax[:, 1], format="%+2.f", aspect=50)
        fig.set_size_inches(8, 3)
        return fig
