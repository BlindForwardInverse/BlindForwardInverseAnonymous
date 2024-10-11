import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pprint import pprint

import os;opj = os.path.join

def plot_metric_effectwise(mixture_single_csv,
                           mixture_multi_csv,
                           single_single_csv,
                           single_multi_csv,
                           multi_single_csv,
                           multi_multi_csv,
                           save_dir='/ssd4/doyo/plots'
                           ):
    mixture = load_results_from_csv(mixture_single_csv, mixture_multi_csv)
    single_effect = load_results_from_csv(single_single_csv, single_multi_csv)
    multi_effect = load_results_from_csv(multi_single_csv, multi_multi_csv)

    # Define the metrics and types for grouping
    types = ['Mixture', 'Single Effect', 'Multiple Effect']
    metrics = ['si_sdr', 'mss', 'mag']
    effects = list(mixture[metrics[0]].keys())
    metric_wise = dict()
    metric_wise_std = dict()
    for metric in metrics:
        metric_wise[metric] = pd.concat([mixture[metric], single_effect[metric], multi_effect[metric]], axis=1)
        metric_wise[metric].columns = types

        metric_wise_std[metric] = pd.concat([mixture_error[metric], single_effect_error[metric], multi_effect_error[metric]], axis=1)
        metric_wise_std[metric].columns = types
        print(metric_wise[metric])
        print(metric_wise_std[metric])

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 6), sharex=True)  # Adjust size as needed
    axes = axes.flatten()
    label_x_position = -0.06

#     # Iterate over metrics and create a subplot for each
    metric_name = {'si_sdr' : 'SI-SDR (↑)', 'mss' : 'SC (↓)', 'mag': 'LSM (↓)'} 
    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_wise[metric].plot.bar(rot=25,
                                     ax=axes[i],
                                     yerr=metric_wise_std[metric],
                                     width=0.65,
                                     capsize=2,
                                     error_kw=dict(linewidth=0.5))
        ax.set_ylabel(metric_name[metric], labelpad=20)
        ax.set_xlabel('')
        ax.yaxis.set_label_coords(label_x_position, 0.5)

#         # Set horizontal alignment for x-axis labels
#         for tick in ax.get_xticklabels():
#             tick.set_ha('right')  # or 'center', 'left' as per your requirement

        # Set horizontal alignment for x-axis labels with rotation_mode='anchor'
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', rotation_mode='anchor')

        ax.legend(loc='upper center',
                  bbox_to_anchor=(0.5, 1.25),
                  frameon=False,
                  ncols=3,
                  fontsize='small')
        if i != 0: ax.get_legend().remove()

    # Adjust layout
    save_name = 'metric_effectwise.pdf'
    plt.tight_layout()
    fig.savefig(opj(save_dir, save_name))

def calc_confidence_interval(std, n=100, z_value=1.96):
    # z=1.96 corresponds to 95% confidence interval
    error = z_value * std / np.sqrt(n)
    return error

def load_results_from_csv(single_csv, multi_csv):
    single_data = pd.read_csv(single_csv, index_col=0)
    multi_data = pd.read_csv(multi_csv, index_col=0)
    data = pd.concat([single_data, multi_data])
    return data

if __name__ == '__main__':
    mixture_single_csv = "/ssd4/doyo/infer_forward/train_single-eval_single/summary_in_dataset_same_mic_mixture.csv"
    mixture_multi_csv = "/ssd4/doyo/infer_forward/train_single-eval_multi/summary_in_dataset_same_mic_mixture.csv"
    single_single_csv = "/ssd4/doyo/infer_forward/train_single-eval_single/summary_in_dataset_same_mic_eval.csv"
    single_multi_csv = "/ssd4/doyo/infer_forward/train_single-eval_multi/summary_in_dataset_same_mic_eval.csv"
    multi_single_csv = "/ssd4/doyo/infer_forward/train_multi-eval_single/summary_in_dataset_same_mic_eval.csv"
    multi_multi_csv = "/ssd4/doyo/infer_forward/train_multi-eval_multi/summary_in_dataset_same_mic_eval.csv"

    plot_metric_effectwise(
                          mixture_single_csv, mixture_multi_csv,
                          single_single_csv, single_multi_csv,
                          multi_single_csv, multi_multi_csv,
                          mixture_single_std, mixture_multi_std,
                          single_single_std, single_multi_std,
                          multi_single_std, multi_multi_std,
                          )

