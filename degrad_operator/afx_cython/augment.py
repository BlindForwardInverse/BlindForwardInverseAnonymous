import torch
from torch.nn import *
#import torchaudio
from scipy import signal
from scipy.stats import loguniform
from pedalboard import Pedalboard, Bitcrush, Compressor, Chorus, Reverb, Distortion, Delay, LadderFilter, Limiter, NoiseGate, Phaser, PitchShift
import os; opj = os.path.join
import soundfile as sf
import numpy as np
from einops import repeat
from torch.utils.data import Dataset
from pprint import pprint
import ffmpeg
from afx.primitives import get_signal, gain_stage

LogU   = loguniform.rvs
U      = np.random.uniform
choice = np.random.choice

encoders = {'mp3' : {'weight': 20, 'encoders': {'libmp3lame': {'bitrate': (8, 256)}}},
            'mp2' : {'weight':  5, 'encoders': {'mp2'       : {'bitrate': [32, 48, 56, 64, 80, 96, 112, 128, 160]},
                                                'mp2fixed'  : {'bitrate': [32, 48, 56, 64, 80, 96, 112, 128, 160]}}},
            'opus': {'weight': 15, 'encoders': {'opus'      : {'bitrate': (8, 256),  'sample_rate': 48000},
                                                'libopus'   : {'bitrate': (8, 256),  'sample_rate': 48000}}},
            'ogg' : {'weight':  3, 'encoders': {'vorbis'    : {'bitrate': (8, 256),  'sample_rate': (4000, 16000)}, 
                                                'libvorbis' : {'bitrate': (48, 200)}}},
            'aac' : {'weight':  3, 'encoders': {'aac'       : {'bitrate': (8, 256)}}},
            'ac3' : {'weight':  2, 'encoders': {'ac3'       : {'bitrate': (32, 256), 'sample_rate': 48000}, 
                                                'ac3_fixed' : {'bitrate': (8, 256),  'sample_rate': 48000}}},
            'eac3': {'weight':  3, 'encoders': {'eac3'      : {'bitrate': (32, 256), 'sample_rate': 32000}}},
            'wma' : {'weight':  2, 'encoders': {'wmav1'     : {'bitrate': (24, 256)}, 
                                                'wmav2'     : {'bitrate': (24, 256)}}}}

exts = {"libmp3lame": "mp3",
        "mp2": "mp2",
        "mp2fixed": "mp2",
        "opus": "opus",
        "libopus": "opus",
        "vorbis": "ogg",
        "libvorbis": "ogg",
        "aac": "aac",
        "ac3": "ac3",
        "ac3_fixed": "ac3",
        "eac3": "eac3",
        "wmav1": "wma",
        "wmav2": "wma"}

def add_verbose(command_str, verbose=False):
    return command_str if verbose else command_str+' -loglevel panic' 

def _apply_codec(x, 
                 encoder, 
                 sr=44100, 
                 sr_codec=48000, 
                 bitrate=192, 
                 save_dir='codec_temp', 
                 verbose=False, 
                 remove=True):

    os.makedirs(save_dir, exist_ok=True)
    while True:
        filename = str(np.random.randint(100000000))
        if not os.path.exists(opj(save_dir, filename+".wav")): break

    codec = exts[encoder]
    wav_dir, converted_dir = opj(save_dir, filename+'.wav'), opj(save_dir, filename+'.'+codec)
    if x.ndim == 2: x = x[:, 0]
    if codec == 'ogg' : x = repeat(x, 't -> t c', c=2)
    channels = 2 if codec == 'ogg' else 1
    sf.write(wav_dir, x, sr)
    os.system(add_verbose('ffmpeg -y -i %s -vn -ar %d -ac %d -codec:a %s -strict experimental -b:a %dk %s', verbose) 
                          % (wav_dir, sr_codec, channels, encoder, bitrate, converted_dir))
    try:
        out, _ = (ffmpeg.input(converted_dir).output('-', format='f32le', acodec='pcm_f32le', ac=1, ar=str(sr))
                    .run(capture_stdout=True, capture_stderr=True))
        out = np.frombuffer(out, np.float32)
    except:
        os.remove(wav_dir)
        os.remove(converted_dir)
        assert False

    if remove: os.remove(wav_dir); os.remove(converted_dir)
    if len(out) >= len(x):
        out = out[:len(x)]
    else:
        out = np.pad(out, (0, len(x)-len(out)))
    return out[:, None]

def apply_codec(input_signal, afx_type, gain_staging, **param_dict):
    x = get_signal(input_signal, "main", c=1)
    y = _apply_codec(x, afx_type, **param_dict)
    if gain_staging: y = gain_stage(x, y)
    return {"main": y}

def apply_random_codec(x, 
                       sr=44100, 
                       save_dir="codec_temp",
                       verbose=False, 
                       remove=True):

    os.makedirs(save_dir, exist_ok=True)
    while True:
        filename = str(np.random.randint(100000))
        if not os.path.exists(opj(save_dir, filename+".wav")): break

    exts = list(encoders.keys())
    ext_weights = np.array([encoders[ext]['weight'] for ext in exts])
    ext_p = ext_weights/sum(ext_weights)
    ext = choice(exts, p=ext_p)
    encoder_name = choice(list(encoders[ext]['encoders'].keys()))
    encoder_config = encoders[ext]['encoders'][encoder_name]

    bitrate_config = encoder_config['bitrate']
    if type(bitrate_config) == tuple:
        bitrate = int(LogU(*bitrate_config))
    else:
        bitrate = choice(bitrate_config)

    print(encoder_name, bitrate)
    if 'sample_rate' in encoder_config.keys():
        if type(encoder_config['sample_rate']) == tuple:
            codec_rate = int(LogU(*encoder_config['sample_rate'])) 
        else:
            codec_rate = encoder_config['sample_rate']
    else:
        codec_rate = sr

    return apply_codec(x, ext, encoder_name, sr, codec_rate, bitrate, save_dir, filename, verbose, remove)
    
def mulaw(x, mu=255):
    x = librosa.mu_compress(x, mu=mu, quantize=True)
    x = librosa.mu_expand(x, mu=mu, quantize=True)
    return x

def random_mulaw(x, mu_range=(5, 256)):
    mu = int(loguniform(*mu_range))
    return mulaw(x, mu)

def test_codec():
    from tqdm import tqdm
    #wav, _ = sf.read('/data4/voicebank_demand/clean_trainset_wav/p226_002.wav')
    wav, sr = sf.read("/nas/public/singing/japanese/sr44100/wav-only/kiritan/kiritan_01/01_0.wav")
    print(sr)
    for i in tqdm(range(100)):
        wav_o = apply_random_codec(wav)#, verbose=True)
        sf.write(f'/workspace/test_out/{i}.wav', wav_o, 44100)

effects = {'compressor': Compressor, 'chorus': Chorus, 'delay': Delay, 'distortion': Distortion, 
           'limiter': Limiter, 'noisegate': NoiseGate, 'phaser': Phaser, 'reverb': Reverb, 'pitchshift': PitchShift}

def apply_pedalboard_effect(x, effect_name='chorus', sr=44100, **kwargs):
    return Pedalboard([effects[effect_name](**kwargs)])(x, sr)

if __name__ == "__main__":
    test_codec()
