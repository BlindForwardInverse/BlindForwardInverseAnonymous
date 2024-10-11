import argparse
import concurrent.futures
import glob
import os
import torch

import librosa
import numpy as np
import numpy.polynomial.polynomial as poly
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from requests import session
from tqdm import tqdm


class DNSMOS:
    def __init__(self, sr=16000, personalized=False) -> None:
        self.sr = sr
        self.p808_model_path = os.path.join('metric', 'model_v8.onnx')
        self.is_personalized = personalized
        if personalized:
            self.primary_model_path = os.path.join('metric', 'sig_bak_ovr.onnx')
        else:
            self.primary_model_path = os.path.join('metric', 'p_sig_bak_ovr.onnx')
        self.onnx_sess = ort.InferenceSession(self.primary_model_path, providers=['CPUExecutionProvider'])
        self.p808_onnx_sess = ort.InferenceSession(self.p808_model_path, providers=['CPUExecutionProvider'])
        self.INPUT_LENGTH = 9.01 # why!!!!!
        
    def audio_melspec(self, audio, n_mels=120, frame_size=320, hop_length=160, sr=16000, to_db=True):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=frame_size+1, hop_length=hop_length, n_mels=n_mels)
        if to_db:
            mel_spec = (librosa.power_to_db(mel_spec, ref=np.max)+40)/40
        return mel_spec.T

    def get_polyfit_val(self, sig, bak, ovr):
        if self.is_personalized:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, fpath, sampling_rate):
        aud, input_fs = sf.read(fpath)
        fs = sampling_rate
        if input_fs != fs:
            audio = librosa.resample(aud, input_fs, fs)
        else:
            audio = aud
        actual_audio_len = len(audio)
        len_samples = int(self.INPUT_LENGTH*fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/fs) - self.INPUT_LENGTH) + 1
        hop_len_samples = fs
        predicted_mos_sig_seg_raw = []
        predicted_mos_bak_seg_raw = []
        predicted_mos_ovr_seg_raw = []
        predicted_mos_sig_seg = []
        predicted_mos_bak_seg = []
        predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+self.INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw)
            predicted_mos_sig_seg_raw.append(mos_sig_raw)
            predicted_mos_bak_seg_raw.append(mos_bak_raw)
            predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            predicted_mos_sig_seg.append(mos_sig)
            predicted_mos_bak_seg.append(mos_bak)
            predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        clip_dict = {'filename': fpath, 'len_in_sec': actual_audio_len/fs, 'sr':fs}
        clip_dict['num_hops'] = num_hops
        clip_dict['OVRL_raw'] = np.mean(predicted_mos_ovr_seg_raw)
        clip_dict['SIG_raw'] = np.mean(predicted_mos_sig_seg_raw)
        clip_dict['BAK_raw'] = np.mean(predicted_mos_bak_seg_raw)
        clip_dict['OVRL'] = np.mean(predicted_mos_ovr_seg)
        clip_dict['SIG'] = np.mean(predicted_mos_sig_seg)
        clip_dict['BAK'] = np.mean(predicted_mos_bak_seg)
        clip_dict['P808_MOS'] = np.mean(predicted_p808_mos)
        return clip_dict
    
    def calculate_dir(self, ref_path="/ssd1/sample16k", csv_path=".", to_csv=False):
        models = glob.glob(os.path.join(ref_path, "*"))

        audio_clips_list = []

        compute_score = DNSMOS()

        rows = []
        clips2 = [glob.glob(os.path.join(ref_path, e)) for e in ['*.wav', '*.flac']]
        clips = [item for sublist in clips2 for item in sublist]
        print(clips)

        for m in tqdm(models):
            max_recursion_depth = 10 
            audio_path = os.path.join(ref_path, m)
            audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
            while len(audio_clips_list) == 0 and max_recursion_depth > 0: # ......
                audio_path = os.path.join(audio_path, "**")
                audio_clips_list = glob.glob(os.path.join(audio_path,"*.wav"))
                max_recursion_depth -= 1
            clips.extend(audio_clips_list)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {executor.submit(compute_score, clip, self.sr): clip for clip in clips}
            for future in tqdm(concurrent.futures.as_completed(future_to_url)):
                clip = future_to_url[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (clip, exc))
                else:
                    rows.append(data)

        df = pd.DataFrame(rows)
        if to_csv:
            df.to_csv(csv_path)
        else:
            print(df.describe())

    def calculate(self, ref):
        audio = ref
        len_samples = int(self.INPUT_LENGTH * self.sr)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        
        num_hops = int(np.floor(len(audio)/self.sr) - self.INPUT_LENGTH) + 1
        hop_len_samples = self.sr
        # predicted_mos_sig_seg_raw = []
        # predicted_mos_bak_seg_raw = []
        # predicted_mos_ovr_seg_raw = []
        # predicted_mos_sig_seg = []
        # predicted_mos_bak_seg = []
        # predicted_mos_ovr_seg = []
        predicted_p808_mos = []

        for idx in range(num_hops):
            audio_seg = audio[int(idx*hop_len_samples) : int((idx+self.INPUT_LENGTH)*hop_len_samples)]
            if len(audio_seg) < len_samples:
                continue

            input_features = np.array(audio_seg).astype('float32')[np.newaxis,:]
            p808_input_features = np.array(self.audio_melspec(audio=audio_seg[:-160])).astype('float32')[np.newaxis, :, :]
            oi = {'input_1': input_features}
            p808_oi = {'input_1': p808_input_features}
            p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
            # mos_sig_raw,mos_bak_raw,mos_ovr_raw = self.onnx_sess.run(None, oi)[0][0]
            # mos_sig,mos_bak,mos_ovr = self.get_polyfit_val(mos_sig_raw,mos_bak_raw,mos_ovr_raw)
            # predicted_mos_sig_seg_raw.append(mos_sig_raw)
            # predicted_mos_bak_seg_raw.append(mos_bak_raw)
            # predicted_mos_ovr_seg_raw.append(mos_ovr_raw)
            # predicted_mos_sig_seg.append(mos_sig)
            # predicted_mos_bak_seg.append(mos_bak)
            # predicted_mos_ovr_seg.append(mos_ovr)
            predicted_p808_mos.append(p808_mos)

        return np.mean(predicted_p808_mos)

if __name__=="__main__":
    # metric = MetricHandler(sr=16000)
    # metric_dict = metric(ref, est)
    dnsmos = DNSMOS(sr=16000)
    # dnsmos.calculate_dir()

    ref = torch.randn(16000)
    print(dnsmos.calculate(ref))
