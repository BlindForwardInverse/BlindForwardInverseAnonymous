import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_sig():
    mean = -1.2
    std = 1.2
    num_sig = 5000
    log_sig = mean + std * torch.randn(num_sig)
    sig = torch.exp(log_sig)

    plt.hist(sig, bins=1000, range=(0, 50), density=True)
    plt.xlabel("sigma")
    plt.ylabel("freq")
    plt.tight_layout()
    plt.savefig("sigma.png")

    plt.hist(sig, bins=1000, log=True, range=(0, 50), density=True)
    plt.xlabel("sigma")
    plt.ylabel("freq")
    plt.tight_layout()
    plt.savefig("logsc_sigma.png")


plot_sig()
