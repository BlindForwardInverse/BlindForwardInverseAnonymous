import torch

from metric import SpeechEnhancementMetric, ReferenceFreeMetric
import soundfile as sf
import os; opj=os.path.join
from glob import glob
from tqdm import tqdm
from pprint import pprint
from librosa import resample

def measure_metric(sample_dir):
    paths = glob(opj(sample_dir, f'in_dataset_same_mic*.wav'))
    print(len(paths), sample_dir)
    metric_fn = SpeechEnhancementMetric()
    dns = ReferenceFreeMetric()

    key_list = []
    for path in tqdm(paths):
        key = '_'.join(path[:-4].split('_')[:-1])
        if key not in key_list : key_list.append(key)

    metric_types = ['sisdr', 'pesq', 'estoi']
    groups = ['lowpass', 'bandpass', 'highpass', 'bandreject', 'distortion', 'clip', 'mono_reverb', 'rir_conv', 'micir_conv', 'noise', 'monolithic', 'complex']
    metric_history = {metric_type:{group : [] for group in groups} for metric_type in metric_types}
    for key in tqdm(key_list):
        clean, _ = sf.read(key + '_clean' + '.wav')
        noisy, _ = sf.read(key + '_noisy' + '.wav')
        enhanced, _ = sf.read(key + '_enhanced' + '.wav')
        afx, _, num, _ = info(key + '_clean'+'.wav')
        enhanced_16k = torch.Tensor(resample(y=enhanced,orig_sr=44100, target_sr=16000))
        clean, noisy, enhanced = [torch.Tensor(audio) for audio in [clean, noisy, enhanced]]
        metric = metric_fn(enhanced, clean)
        for group in groups:
            if group in afx:
                for met in metric:
                    metric_history[met][group].append(metric[met])

    for met in metric_history:
        for g in groups:
            hist = metric_history[met][g]
            if len(hist) > 0 :
                avg = sum(hist) / len(hist)
                metric_history[met][g] = avg.item()
            else:
                del metric_history[met][g]
    pprint(metric_history)


def info(path):
    percep, num, audio_type = path[:-4].split('_')[-3:]
    afx = '_'.join(path.split('/')[-1].split('_')[4:-3])
    return afx, percep, int(num), audio_type



if __name__ == '__main__':
    sample_dir = '/ssd3/doyo/uncond_diffusion/0522_20_deterministic_single_env_single_effect_ch128/epoch_99'
    sample_dir = '/ssd3/doyo/uncond_diffusion/0516_17_t_sum_c_cat/epoch_199'
    sample_dir = '/ssd3/doyo/uncond_diffusion/0522_21_single_env_monolithic_ch128/epoch_54'
    measure_metric(sample_dir)

