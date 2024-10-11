from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
import torchaudio
from torchaudio.pipelines import SQUIM_SUBJECTIVE as bundle

from torchmetrics import Metric

from torchaudio.transforms import Resample
import torchaudio.functional as F
import torch.nn as nn
import torch
import numpy as np

from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
from utils.audio_normalization import rms_normalize
from utils.audio_processing import make_it_stereo

from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2Config
from collections import OrderedDict
from einops import rearrange

from tqdm import tqdm

import numpy.polynomial.polynomial as poly
import onnxruntime as ort
import soundfile as sf
import pandas as pd
import numpy as np

import concurrent.futures
import urllib.request
import hashlib
import librosa
import glob
import os

from loss import MultiResolutionSTFTLoss

'''
Metric for Operator Learning
----------------------------------------------
'''
class EffectWiseWetMetric(nn.Module):
    # Inherit nn.Module to enable .to() method
    def __init__(self, full_afx_types, device='cuda'):
        super().__init__()
        if full_afx_types is not None:
            self.si_sdr = {afx_type : ScaleInvariantSignalDistortionRatio().to(device) for afx_type in full_afx_types}

    def update(self, pred, target, afx_types):
        # Pred, target, afx_types : batched
        for _pred, _target, _afx_type in zip(pred, target, afx_types):
            self.si_sdr[_afx_type].update(_pred, _target)

    def compute(self):
        metric_dict = {}
        for afx_type in self.si_sdr:
            metric_dict[afx_type] = self.si_sdr[afx_type].compute().detach().cpu().item()
        return metric_dict

    def reset(self):
        for afx_type in self.si_sdr:
            self.si_sdr[afx_type].reset()

class EffectWiseWetMetricExtended(nn.Module):
    def __init__(self, full_afx_types, device='cuda'):
        super().__init__()
        self.full_afx_types = full_afx_types

        self.si_sdr = {afx_type : ScaleInvariantSignalDistortionRatio().to(device) for afx_type in full_afx_types}
        self.sc_loss = {afx_type : [] for afx_type in full_afx_types}
        self.lsm_loss = {afx_type : [] for afx_type in full_afx_types}

        self.MSS = MultiResolutionSTFTLoss(fft_sizes=[4096, 2048, 1024, 512],
                                           hop_sizes=[1024, 512, 256, 128],
                                           win_lengths=[4096, 2048, 1024, 512],
                                           factor_sc=1.,
                                           factor_mag=1.,
                                           factor_phase=0)

    def update(self, pred, target, afx_types):
        # Pred, target, afx_types : batched
        _afx_type = afx_types
        for _pred, _target in zip(pred, target):
            sc_loss, lsm_loss, _ = self.MSS(_pred.unsqueeze(0), _target.unsqueeze(0))
#             sc_loss, lsm_loss, _ = self.MSS(_pred, _target)
            self.si_sdr[_afx_type].update(_pred, _target)
            self.sc_loss[_afx_type].append(sc_loss)
            self.lsm_loss[_afx_type].append(lsm_loss)

    def compute(self):
        si_sdr = {}
        sc_loss = {}
        lsm_loss = {}
        for afx_type in self.full_afx_types:
            si_sdr[afx_type] = self.si_sdr[afx_type].compute().detach().cpu().item()
            sc_loss[afx_type] = (sum(self.sc_loss[afx_type]) / len(self.sc_loss[afx_type])).detach().cpu().item()
            lsm_loss[afx_type] = (sum(self.lsm_loss[afx_type]) / len(self.lsm_loss[afx_type])).detach().cpu().item()

        metric = {'si_sdr': si_sdr, 'sc_loss' : sc_loss, 'lsm_loss' : lsm_loss}
        return metric

    def reset(self):
        for afx_type in self.full_afx_types:
            self.si_sdr[afx_type].reset()
            self.sc_loss[afx_type] = []
            self.lsm_loss[afx_type] = []

class MetricHandlerOperator(nn.Module):
    def __init__(self, metric_type=['si_sdr', 'si_snr']):
        super().__init__()
        self.metric_type = metric_type
        if 'si_sdr' in metric_type: self.si_sdr = ScaleInvariantSignalDistortionRatio()
        if 'si_snr' in metric_type: self.si_snr = ScaleInvariantSignalNoiseRatio()
        if 'mss' in metric_type: self.mss = MultiResolutionSTFTLoss(
                                 fft_sizes=  [4096, 2048, 1024, 512],
                                 hop_sizes=  [1024, 512,  256,  128],
                                 win_lengths=[4096, 2048, 1024, 512],
                                 window="hann_window", 
                                 factor_sc=1,
                                 factor_mag=1,
                                 factor_phase=0)

    def forward(self, pred, target):
        metric = dict()
        if 'si_sdr' in self.metric_type: metric['si_sdr'] = self.si_sdr(pred, target)
        if 'si_snr' in self.metric_type: metric['si_snr'] = self.si_snr(pred, target)
        if 'mss' in self.metric_type :
            pred = torch.unsqueeze(pred,0)
            target = torch.unsqueeze(target,0)
            mss, mag, _ = self.mss(pred, target)
            metric['mss'] = mss
            metric['mag'] = mag
        return metric

'''
Metric for Diffusion Training
----------------------------------------------
'''
class EffectWiseSEMetric(nn.Module):
    # Inherit nn.Module to enable .to() method
    def __init__(self, full_afx_types, sr=44100, device='cuda'):
        super().__init__()
        self.full_afx_types = full_afx_types

        self.si_sdr = {afx_type : ScaleInvariantSignalDistortionRatio().to(device) for afx_type in full_afx_types}
        self.pesq = {afx_type : PerceptualEvaluationSpeechQuality(fs=16000, mode='wb').to(device) for afx_type in full_afx_types}
        self.estoi = {afx_type : ShortTimeObjectiveIntelligibility(fs=16000, extended=True) for afx_type in full_afx_types}
        self.resampler_16k = Resample(sr, 16000)

    def update(self, pred, target, afx_types):
        # Pred, target, afx_types : batched
        for _pred, _target, _afx_type in zip(pred, target, afx_types):
            self.si_sdr[_afx_type].update(_pred, _target)

            _pred_16k = self.downsample_to_16k(_pred)
            _target_16k = self.downsample_to_16k(_target)
            self.pesq[_afx_type].update(_pred_16k, _target_16k)
            self.estoi[_afx_type].update(_pred_16k, _target_16k)

    def compute(self):
        si_sdr = {}
        pesq = {}
        estoi = {}
        for afx_type in self.full_afx_types:
            si_sdr['si_sdr-' + afx_type] = self.si_sdr[afx_type].compute().detach().cpu().item()
            pesq['pesq-' + afx_type] = self.pesq[afx_type].compute().detach().cpu().item()
            estoi['estoi-' + afx_type] = self.estoi[afx_type].compute().detach().cpu().item()
        return si_sdr, pesq, estoi

    def reset(self):
        for afx_type in self.full_afx_types:
            self.si_sdr[afx_type].reset()
            self.pesq[afx_type].reset()
            self.estoi[afx_type].reset()

    def downsample_to_16k(self, audio):
        return self.resampler_16k(audio)

class EffectWiseSEMetricExtended(nn.Module):
    # Inherit nn.Module to enable .to() method
    def __init__(self, full_afx_types, sr=44100, device='cuda'):
        super().__init__()
        self.full_afx_types = full_afx_types

        self.si_sdr = {afx_type : ScaleInvariantSignalDistortionRatio().to(device) for afx_type in full_afx_types}
        self.pesq = {afx_type : PerceptualEvaluationSpeechQuality(fs=16000, mode='wb').to(device) for afx_type in full_afx_types}
        self.estoi = {afx_type : ShortTimeObjectiveIntelligibility(fs=16000, extended=True) for afx_type in full_afx_types}
        self.squim_sub = {afx_type : [] for afx_type in full_afx_types}

        self.squim_model = bundle.get_model()
        self.resampler_16k = Resample(sr, 16000)

    def update(self, pred, target, afx_types):
        # Pred, target, afx_types : batched
        for _pred, _target, _afx_type in zip(pred, target, afx_types):
            self.si_sdr[_afx_type].update(_pred, _target)

            _pred_16k = self.downsample_to_16k(_pred)
            _target_16k = self.downsample_to_16k(_target)
            self.pesq[_afx_type].update(_pred_16k, _target_16k)
            self.estoi[_afx_type].update(_pred_16k, _target_16k)
            self.squim_sub[_afx_type].append(self.squim_model(_pred_16k.unsqueeze(0), _target_16k.unsqueeze(0)).detach())

    def compute(self):
        si_sdr = {}
        pesq = {}
        estoi = {}
        squim_sub = {}
        for afx_type in self.full_afx_types:
            si_sdr[afx_type] = self.si_sdr[afx_type].compute().detach().cpu().item()
            pesq[afx_type] = self.pesq[afx_type].compute().detach().cpu().item()
            estoi[afx_type] = self.estoi[afx_type].compute().detach().cpu().item()
            squim_sub[afx_type] = sum(self.squim_sub[afx_type])/len(self.squim_sub[afx_type])
            squim_sub[afx_type] = squim_sub[afx_type].detach().cpu().item()

        metric = {'si_sdr' : si_sdr, 'pesq' : pesq, 'estoi' : estoi, 'squim_sub' : squim_sub}
        return metric

    def reset(self):
        for afx_type in self.full_afx_types:
            self.si_sdr[afx_type].reset()
            self.pesq[afx_type].reset()
            self.estoi[afx_type].reset()

    def downsample_to_16k(self, audio):
        return self.resampler_16k(audio)

class SpeechEnhancementMetric(nn.Module):
    def __init__(self, sr=44100):
        super().__init__()
        self.sr = sr
        self.sisdr = ScaleInvariantSignalDistortionRatio()
        self.pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb')
        self.estoi = ShortTimeObjectiveIntelligibility(fs=16000, extended=True)

        self.resampler_16k = Resample(sr, 16000)

    def forward(self, estimated, reference):
        metric = {}
        metric["sisdr"] = self.sisdr(estimated, reference)

        if self.sr != 16000:
            estimated = self.downsample_to_16k(estimated)
            reference = self.downsample_to_16k(reference)

        metric["pesq"] = self.pesq(estimated, reference)
        metric["estoi"] = self.estoi(estimated, reference)
        return metric

    def downsample_to_16k(self, audio):
        return self.resampler_16k(audio)

class ReferenceFreeMetric(nn.Module):
    def __init__(self, sr=44100):
        super().__init__()
        self.sr = sr
#         self.wvmos = Wav2Vec2MOS()
        self.dnsmos = DNSMOS(sr=sr)

        # self.squim_obj = SQUIM_OBJECTIVE.get_model()
        # self.squim_subj = SQUIM_SUBJECTIVE.get_model()
        # for param in self.squim_obj.parameters():
        #     param.requires_grad = False
        # for param in self.squim_subj.parameters():
        #     param.requires_grad = False

    def forward(self, estimated):
        metric = {}
#         metric["wvmos"] = self.wvmos.calculate(estimated)
        metric["dnsmos"] = self.dnsmos.calculate(estimated)

        # estimated = F.resample(estimated, 44100, 16000)
        # estimated = rearrange(estimated, 't -> 1 t')

        # reference = F.resample(reference, 44100, 16000)
        # reference = rearrange(reference, 't -> 1 t')

        # for i, key in zip(range(3), ['squim_stoi', 'squim_pesq', 'squim_sisdr']):
        #     metric[key] = self.squim_obj(estimated)[i].item()
        # metric['squim_mos'] = self.squim_subj(estimated, reference).item()

        return metric

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


class Wav2Vec2MOS(nn.Module):
    def __init__(self, freeze=True, cuda=True):
        super().__init__()
        self.path = os.path.join(os.path.expanduser("~"), ".cache/wv_mos/wv_mos.ckpt")
        self.encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.freeze = freeze
        self.dense = nn.Sequential(
            nn.Linear(768, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1)
        )

        self.wvmos_setup()

        if self.freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.load_state_dict(
            self.__extract_prefix("model.", torch.load(self.path)["state_dict"])
        )  ##
        self.eval()
        self.cuda_flag = cuda
        if cuda:
            self.cuda()
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )

    def __extract_prefix(self, prefix, weights):
        result = OrderedDict()
        for key in weights:
            if key.find(prefix) == 0:
                result[key[len(prefix) :]] = weights[key]
        return result

    def __compute_sha256(self, file_path):
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha256.update(block)

        return sha256.hexdigest()

    def wvmos_setup(self):
        EXPECTED_CHECKSUM = (
            "f222559733f9d660fc0592b39abba9726adec6b97e9030c072c3ed6fa14ef12f"
        )
        WVMOS_PATH = "https://zenodo.org/record/6201162/files/wav2vec2.ckpt?download=1"

        local_checksum = None

        if os.path.exists(self.path):
            local_checksum = self.__compute_sha256(self.path)
            print("exists")
            print(local_checksum)

        # Download if the file doesn't exist or the checksum doesn't match
        if local_checksum is None or local_checksum != EXPECTED_CHECKSUM:
            try:
                print("Downloading the checkpoint for WV-MOS")
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                download_url(WVMOS_PATH, self.path)
                # urllib.request.urlretrieve(WVMOS_PATH, self.path)
                print(
                    f"Weights downloaded in: {self.path} Size: {os.path.getsize(self.path)}"
                )
                local_checksum = self.__compute_sha256(self.path)
                print(local_checksum)
            except Exception as e:
                print(f"An error occurred: {e}")

    def forward(self, x):
        x = self.encoder(x)["last_hidden_state"]  # [B, T, F]
        x = self.dense(x)  # [B, T, 1]
        x = x.mean(dim=[1, 2], keepdims=True)  # [B, 1, 1]
        return x

    def train(self, mode):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()

    def calculate(self, signal):
        x = self.processor(
            signal, return_tensors="pt", padding=True, sampling_rate=16000
        ).input_values
        with torch.no_grad():
            if self.cuda_flag:
                x = x.cuda()
            res = self.forward(x).mean()
        return res.cpu().item()


class DNSMOS:
    def __init__(self, sr=44100, personalized=False) -> None:
        self.sr = sr
        self.is_personalized = personalized  # ?
        self.INPUT_LENGTH = 9.01  # why!!!!!

        # Model paths
        self.p808_model_path = os.path.join("metric/dnsmos", "model_v8.onnx")
        self.primary_model_path = os.path.join(
            "metric/dnsmos", "sig_bak_ovr.onnx" if not personalized else "p_sig_bak_ovr.onnx"
        )

        # Initialize onnx sessions
        self.onnx_sess = ort.InferenceSession(
            self.primary_model_path, providers=["CPUExecutionProvider"]
        )
        self.p808_onnx_sess = ort.InferenceSession(
            self.p808_model_path, providers=["CPUExecutionProvider"]
        )

    def __audio_melspec(
        self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True
    ):
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=frame_size + 1, hop_length=hop_length, n_mels=n_mels
        )
        return (
            (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
            if to_db
            else mel_spec.T
        )

    def __get_polyfit_val(self, sig, bak, ovr):
        # Get polynomial fit values based on whether the model is personalized or not
        if self.is_personalized:
            coefficients = {
                "ovr": [-0.00533021, 0.005101, 1.18058466, -0.11236046],
                "sig": [-0.01019296, 0.02751166, 1.19576786, -0.24348726],
                "bak": [-0.04976499, 0.44276479, -0.1644611, 0.96883132],
            }
        else:
            coefficients = {
                "ovr": [-0.06766283, 1.11546468, 0.04602535],
                "sig": [-0.08397278, 1.22083953, 0.0052439],
                "bak": [-0.13166888, 1.60915514, -0.39604546],
            }

        return (
            np.poly1d(coefficients["sig"])(sig),
            np.poly1d(coefficients["bak"])(bak),
            np.poly1d(coefficients["ovr"])(ovr),
        )

    def __process_audio_segments(self, audio):
        # Process audio segments and compute metrics
        actual_audio_len = len(audio)
        len_samples = int(self.INPUT_LENGTH * self.sr)

        # Ensure audio has a minimum length
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / self.sr) - self.INPUT_LENGTH) + 1
        hop_len_samples = self.sr

        # Metrics storage
        (
            predicted_mos_sig_seg_raw,
            predicted_mos_bak_seg_raw,
            predicted_mos_ovr_seg_raw,
        ) = ([], [], [])
        predicted_mos_sig_seg, predicted_mos_bak_seg, predicted_mos_ovr_seg = [], [], []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[
                int(idx * hop_len_samples) : int(
                    (idx + self.INPUT_LENGTH) * hop_len_samples
                )
            ]

            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype("float32")[np.newaxis, :]
            p808_input_features = np.array(
                self.__audio_melspec(audio=audio_seg[:-160])
            ).astype("float32")[np.newaxis, :, :]
            # p808_input_features = p808_input_features.transpose(1, 2) # <- ?...
            p808_input_features = np.swapaxes(p808_input_features, 1, 2)

            oi = {"input_1": input_features}
            p808_oi = {"input_1": p808_input_features}

            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw, mos_bak_raw, mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]

            mos_sig, mos_bak, mos_ovr = self.__get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovr_raw
            )

            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        # Consolidate the metrics into dict
        clip_dict = {
            "len_in_sec": actual_audio_len / self.sr,
            "sr": self.sr,
            "num_hops": num_hops,
            "OVRL_raw": np.mean(predicted_mos_ovr_seg_raw),
            "SIG_raw": np.mean(predicted_mos_sig_seg_raw),
            "BAK_raw": np.mean(predicted_mos_bak_seg_raw),
            "OVRL": np.mean(predicted_mos_ovr_seg),
            "SIG": np.mean(predicted_mos_sig_seg),
            "BAK": np.mean(predicted_mos_bak_seg),
            "P808_MOS": np.mean(predicted_p808_mos),
        }

        return clip_dict

    def __fetch_all_audio_files(self, ref_path):
        # Recursively fetch all audio files under a path
        audio_extensions = ["*.wav", "*.flac"]
        audio_files = []

        for ext in audio_extensions:
            audio_files.extend(
                glob.glob(os.path.join(ref_path, "**", ext), recursive=True)
            )

        return audio_files

    def evaluate_audio(self, fpath, sampling_rate):
        # Evaluate audio quality metrics for the provided audio file
        aud, input_fs = sf.read(fpath)
        audio = (
            librosa.resample(aud, input_fs, sampling_rate)
            if input_fs != sampling_rate
            else aud
        )
        return self.__process_audio_segments(audio)

    def calculate(self, audio):
        # Calculate the quality metric for an individual audio array
        metrics = self.__process_audio_segments(audio)
        return metrics["P808_MOS"]

    def calculate_dir(self, ref_path, csv_path=".", to_csv=False):
        # Calculate metrics for all audio files under a directory
        audio_files = self.__fetch_all_audio_files(ref_path)

        # Compute metrics for each audio
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_audio = {
                executor.submit(self.__process_audio_segments, clip, self.sr): clip
                for clip in audio_files
            }
            rows = []
            for future in tqdm(concurrent.futures.as_completed(future_to_audio)):
                clip = future_to_audio[future]
                try:
                    data = future.result()
                    data["filename"] = clip
                    rows.append(data)
                except Exception as exc:
                    print(f"{clip} generated an exception: {exc}")

        # Save to CSV or print the results
        df = pd.DataFrame(rows)
        df.to_csv(csv_path) if to_csv else print(df.describe())


if __name__ == "__main__":
    from pprint import pprint

    metric = MetricHandler(sr=16000)
    ref = torch.randn(16000)
    est = torch.randn(16000)
    metric_dict = metric(ref, est)
    pprint(metric_dict)

    rfmetric = ReferenceFreeMetric(sr=16000)
    rfmetric_dict = rfmetric(est)
    pprint(rfmetric_dict)
