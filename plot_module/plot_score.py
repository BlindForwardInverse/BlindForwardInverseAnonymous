import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_score(uncond_score, post_score, save_name):
    fig, ax = plt.subplots()
    steps = len(uncond_score)
    ax.plot(steps, uncond_score, label='uncond_score')
    ax.plot(steps, post_score, label='post_score')
    fig.suptitle(save_name)
    fig.savefig(save_name + '.png')
    plt.close()
    
    
    
