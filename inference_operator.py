import os
opj = os.path.join
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt

import pandas as pd
from copy import deepcopy
import soundfile as sf
from functools import partial
from pprint import pprint
from tqdm import tqdm
from omegaconf import OmegaConf

from metric import EffectWiseWetMetric, EffectWiseWetMetricExtended
from data_module.valid_dataset import PrerenderedValidDataset
from solvers.operator_solver_hybrid import OperatorSolverHybrid

from utils.audio_transform import spec_transform
from utils.audio_normalization import peak_normalize
from utils.torch_utils import seed_everything, load_pretrained_model

from sklearn.manifold import TSNE

class InferenceForwardOperator:
    def __init__(
        self,
        seed=42,
        sr=44100,
        batch_size=32,
        valid_set_types=['vctk1'],
        # Args
        save_dir = '/ssd3/doyo/inference_forward_operator',
        config_path = 'configs/pretrained_model/forward_operator.yaml',
        config_name = 'hb2-single-wo-codec',
        **kwargs
        ):
        if seed is not None : seed_everything(seed)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
            print("Remove the previous inference results")
        os.makedirs(save_dir, exist_ok=True)

        self.sr = sr
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.valid_set_types = valid_set_types

        # Load Operator
        self.pretrained_operator = load_pretrained_model(solver_class=OperatorSolverHybrid,
                                                        config_path=config_path,
                                                        config_name=config_name,
                                                        freeze=True,
                                                        device='cuda')
        self.spec_to_wav = partial(spec_transform, transform_type='unpower_ri',
                                   n_fft=2046, hop_length=512, win_length=2046, length=512*191)

    @staticmethod
    def to_cuda(batch):
        for k, v in batch.items():
            batch[k] = v.to('cuda') if isinstance(v, torch.Tensor) else v
        return batch

    def forward_operator(self, test_set_path):
        loader, full_afx_types = load_test_set_loader(test_set_path, self.sr, self.valid_set_types)
        operator = self.pretrained_operator

        full_afx_types = [self.map_effect_type(e) for e in full_afx_types]
        pred_metric = {valid_set : EffectWiseWetMetricExtended(full_afx_types) for valid_set in self.valid_set_types} # Metric for (pred, wet)
        dry_metric = {valid_set : EffectWiseWetMetricExtended(full_afx_types) for valid_set in self.valid_set_types}# Metric for (dry, wet)

        with operator.ema.average_parameters():
            for batch in tqdm(iter(loader)):
                batch = self.to_cuda(batch)
                dry_tar_wav, wet_tar_wav, dry_tar_spec, wet_tar_spec, dry_ref_wav, wet_ref_wav, wet_ref_spec =\
                    [batch.get(key) for key in\
                     ["dry_tar", "wet_tar", "dry_tar_spec", "wet_tar_spec", "dry_ref", "wet_ref", "wet_ref_spec"]]
                valid_types, afx_names, afx_types, audio_nums = \
                    [batch[key] for key in ["valid_type", "afx_name", "afx_type", "audio_idx"]]

                # Model Evaluation
                _, _, pred_tar_wav = operator(dry_tar_spec, dry_tar_wav, wet_ref_wav, wet_ref_spec)

                afx_types = [self.map_effect_type(e) for e in afx_types]
                for _pred_tar_wav, _wet_tar_wav, _dry_tar_wav, _afx_types, _valid_types in zip(pred_tar_wav, wet_tar_wav, dry_tar_wav, afx_types, valid_types):
                    pred_metric[_valid_types].update(_pred_tar_wav.unsqueeze(0), _wet_tar_wav.unsqueeze(0), _afx_types)
                    dry_metric[_valid_types].update(_dry_tar_wav.unsqueeze(0), _wet_tar_wav.unsqueeze(0), _afx_types)

                for i, (_pred, _wet, _dry, _dry_ref, _wet_ref, valid_type, afx, audio_num) in \
                        enumerate(zip(pred_tar_wav, wet_tar_wav, dry_tar_wav, dry_ref_wav, wet_ref_wav, valid_types, afx_names, audio_nums)):
                    # Save the pred_wav
                    save_name = f"{afx}-{audio_num}-0dry_ref.wav"
                    self.save_audio(save_name, valid_type, _dry_ref)

                    save_name = f"{afx}-{audio_num}-0wet_ref.wav"
                    self.save_audio(save_name, valid_type, _wet_ref)

                    save_name = f"{afx}-{audio_num}-1dry_tar.wav"
                    self.save_audio(save_name, valid_type, _dry)

                    save_name = f"{afx}-{audio_num}-2wet_tar.wav"
                    self.save_audio(save_name, valid_type, _wet)

                    save_name = f"{afx}-{audio_num}-3pred.wav"
                    self.save_audio(save_name, valid_type, _pred)

        # Summarize Metric
        pred_metric = {valid_type : pred_metric[valid_type].compute() for valid_type in self.valid_set_types}
        dry_metric = {valid_type : dry_metric[valid_type].compute() for valid_type in self.valid_set_types}
        
        self.export_to_csv(pred_metric, tag='pred')
        self.export_to_csv(dry_metric, tag='dry')

    def save_audio(self, name, valid_type, audio):
        audio = audio.detach().cpu().numpy()
        dir_name = opj(self.save_dir, valid_type)
        os.makedirs(dir_name, exist_ok=True)
        sf.write(opj(dir_name, name), audio, self.sr)

    def export_to_csv(self, metric_dict, tag='inference'):
        for valid_type, metric in metric_dict.items():
            df = pd.DataFrame(metric)
            print(df)
            exp_name = self.save_dir.split('/')[-1]
            name = exp_name + f'_{valid_type}_' + f'_metric_summary_{tag}.csv'
            csv_name = os.path.join(self.save_dir, name)
            csv_save_dir = '/ssd4/doyo/csv_summary'
            os.makedirs(csv_save_dir, exist_ok=True)
            df.to_csv(csv_name, index_label='Effect Type')
            df.to_csv(os.path.join(csv_save_dir, name), index_label='Effect Type')

    @staticmethod
    def map_effect_type(effect):
        if 'rir_conv' in effect:
            return 'rir_conv'
        elif 'micir_conv' in effect:
            return 'micir_conv'
        elif 'pass_ladder' in effect:
            return effect.replace('_ladder', '')
        elif 'clip' in effect:
            return 'clipping'
        elif effect in ['svf', 'parametric_eq']:
            return 'EQ'
        elif effect in ['deesser', 'plosive', 'compressor']:
            return 'compressor'
        elif effect in ['aac', 'libmp3lame', 'libopus', 'libvorbis']:
            return 'codec'
        elif effect in ['chorus', 'tremolo', 'flanger']:
            return 'modulation'
        return effect

class PlotTSNE:
    def __init__(self, sr=44100,
                 config_path=None, config_name=None,
                 tsne_save_dir='/ssd4/doyo/inference_operator/tsne'):
        self.pretrained_operator = None
        self.sr = 44100
        self.tsne_save_dir = tsne_save_dir

        pretained_operator = load_pretrained_model(solver_class, config_path, config_name, freeze=True, device='cuda')

    def get_embedding(self, pretrained=True):
        if pretrained:
            operator = self.pretrained_operator
            fx_encoder = operator.fx_encoder
            global_embedder = operator.to_fx_embedding

        embeddings = []
        afx_names = []
        with operator.ema.average_parameters():
            for batch in tqdm(iter(loader)):
                wet_ref = batch['wet_ref'].to('cuda')
                afx_name = batch['afx']
                fx_latent = fx_encoder(wet_ref)
                global_embedding = global_embedder(fx_latent)
                emb_np = global_embedding.detach().cpu().numpy() # (B, 512)
                afx_names += afx_name 
                for b in emb_np:
                    embeddings.append(b)

        embeddings = np.array(embeddings)
        return embeddings

    def plot_tsne(self, embeddings):
        cluster = tsne.fit_transform(np.array(embeddings))

        plt.figure(figsize=(4,4), )
        afx_list = list(set(afx_names))
        label_to_int = {label: idx for idx, label in enumerate(afx_list)}
        label_mapping = [label_to_int[label] for label in afx_names]

        scatter = plt.scatter(cluster[:,0], cluster[:,1], s=0.3, c=label_mapping, cmap='turbo', label=afx_names)
        plt.xlim([-100,100])
        plt.ylim([-100,100])

        plt.savefig(opj(self.save_dir, filename), bbox_inches='tight', pad_inches=0)

def load_test_set_loader(test_set_path, sr=44100, valid_set_types=['vctk1', 'vctk2', 'daps']):
    dataset = PrerenderedValidDataset(
                modality='speech',
                target_sr=sr,
                tar_transform_type='power_ri',
                ref_transform_type='power_ri',
                n_fft=2046,
                win_length=2046,
                hop_length=512,
                prerendered_valid_dir=test_set_path,
                valid_set_types=valid_set_types
                )
    full_afx_types = dataset.full_afx_types

    loader = DataLoader(dataset,
                        batch_size=48,
                        drop_last=False,
                        num_workers=16,
                        pin_memory=True,
                        persistent_workers=True)

    return loader, full_afx_types

if __name__ == "__main__":
    import argparse
    from argparse import ArgumentParser

    parser = ArgumentParser()
    add_arg = parser.add_argument
    Boolean = argparse.BooleanOptionalAction

    add_arg("--task", default='operator_learning')
    add_arg("--batch_size", type=int, default=32)

    add_arg("--save_dir", default='/ssd3/doyo/inference_forward_final/single_effects')
    add_arg("--config_name", default='hb1-single')
    add_arg("--test_set_path", default='/ssd4/doyo/test_set_v2/single_effects')
    add_arg("--valid_set_types", nargs= '+', default=['vctk1', 'vctk2', 'daps'])
    args = vars(parser.parse_args())

    if args['task'] == 'operator_learning':
        inference = InferenceForwardOperator(**args)
        inference.forward_operator(args['test_set_path'])

    elif args['task'] == 'tsne':
        inference = OperatorLearning(**args)
        inference.tsne_global_embedding(filename='maestro.pdf',
                                        valid_set_types=['maestro_single'])
    else:
        print("Not Implemented")
