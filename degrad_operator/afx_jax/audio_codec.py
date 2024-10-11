import jax
import numpy as np
import jax.numpy as jnp
import os
import soundfile as sf
import ffmpeg

from functools import partial

from .jafx_utils import gain_stage, get_signal

def apply_codec(input_signal, afx_type, gain_staging, sr, mono, bitrate, **param_dict):
    assert afx_type in ['libmp3lame', 'aac', 'libvorbis', 'libopus']
    x = get_signal(input_signal, "main")
    y = _render_codec(x, afx_type, sr, bitrate=bitrate)
    out = gain_stage(x, y) if gain_staging else y
    return {"main" : y}

def _render_codec(x,
                  afx_type='libmp3lame',
                  sr=44100,
                  mono=True,
                  save_dir='/workspace/codec_temp',
                  bitrate=256):
    # Save Temp Audio -> Save Encoded Audio -> Recon Audio
    bitrate = str(bitrate) + 'k'
    os.makedirs(save_dir, exist_ok=True)
    while True:
        file_num = str(np.random.randint(100000))
        file_name = os.path.join(save_dir, file_num + '.wav')
        if not os.path.exists(file_name): break

    ext = get_extention(afx_type)
    sf.write(file_name, np.array(x), sr)
    converted_path = os.path.join(save_dir, file_num + '_converted.' + ext)
    reverted_path = os.path.join(save_dir, file_num + '_reverted.wav')

    (
        ffmpeg.input(file_name)
        .output(converted_path,
                ac=1 if mono else 2,
                ar=48000 if afx_type == 'libopus' else sr,
                audio_bitrate=bitrate,
                acodec=afx_type)
        .run(quiet=True)
    )
    (
        ffmpeg.input(converted_path)
        .output(reverted_path,
                ac=1 if mono else 2,
                ar=sr,
                acodec='pcm_s32le')
        .run(quiet=True)
    )
    reverted_audio, sr =  sf.read(reverted_path)
    reverted_audio = reverted_audio[:x.shape[-1]]
    os.remove(converted_path)
    os.remove(reverted_path)
    os.remove(file_name)
    y = jnp.array(reverted_audio)
    return y

def get_extention(codec):
    extention_pair = {'libmp3lame' : 'mp3',
                      'aac' : 'aac',
                      'libvorbis': 'ogg',
                      'libopus': 'opus',
                      }
    return extention_pair[codec]
