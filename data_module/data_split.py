import os; opj=os.path.join
from glob import glob
import random

def split_single_recording_env(data_dirs, mic, shuffle=False):
    data_path = []
    speakers = []
    for data_dir in data_dirs:
        speakers += [path.split('/')[-1] for path in glob(opj(data_dir, '*'))]
        for ext in ["wav", "flac"]:
            data_path += glob(opj(data_dir, f"**/*.{ext}"), recursive=True)

    # Split Speakers for target and reference audio
    split_idx = int(0.5 * len(speakers))
    target_speakers = speakers[:split_idx]
    ref_speakers = speakers[split_idx:]

    # Picks up audio path recorded with a single mic.
    target_audio = []
    ref_audio = []
    for path in data_path:
        speaker, filename = path.split('/')[-2:]
        if mic in filename:
            if speaker in target_speakers:
                target_audio.append(path)
            else:
                ref_audio.append(path)

    if shuffle:
        random.shuffle(target_audio)
        random.shuffle(ref_audio)
    return target_audio, ref_audio

def split_daps(data_dirs, shuffle=True):
    data_path = []
    for data_dir in data_dirs:
        for ext in ["wav", "flac"]:
            data_path += glob(opj(data_dir, f"**/*.{ext}"), recursive=True)

    # Split Speakers for target and reference audio
    split_idx = int(0.5 * len(data_path))
    if shuffle: random.shuffle(data_path)
    else: data_path = sorted(data_path)

    target_audio = data_path[:split_idx]
    ref_audio = data_path[split_idx:]
    return target_audio, ref_audio

def split_multiple_recording_env(data_dirs,
                                 shuffle=True,
                                 add_vctk=True,
                                 single_env_dir=None,
                                 ):
    data_path = []
    for data_dir in data_dirs:
        for ext in ["wav", "flac"]:
            data_path += glob(opj(data_dir, f"**/*.{ext}"), recursive=True)

    # Classify paths according to recording environments (speaker)
    environment = dict()
    for path in data_path:
        env = '/'.join(path.split('/')[:-1])
        if env not in environment: environment[env] = [path] # Initialization
        else: environment[env].append(path)

    split_environment = dict()
    for env in environment:
        classified_paths = environment[env]
        if shuffle: random.shuffle(classified_paths)
        if len(classified_paths) > 1 :
            split_idx = int(0.5 * len(classified_paths))
            split_environment[env] = dict()
            split_environment[env]['target'] = classified_paths[:split_idx]
            split_environment[env]['ref'] = classified_paths[split_idx:]

    if add_vctk:
        for mic in ['mic1', 'mic2']:
            target_audio, ref_audio = split_single_recording_env(single_env_dir, mic=mic, shuffle=shuffle)
            split_environment['vctk_' + mic] = dict()
            split_environment['vctk_' + mic]['target'] = target_audio
            split_environment['vctk_' + mic]['ref'] = ref_audio

    environment_list = [env for env in split_environment]
    return environment_list, split_environment
        
def split_unmatched_recording_env():
    pass

def split_maestro(data_path, shuffle=True,):
    data_path = list(glob(opj(data_path, f"**/*.wav"), recursive=True))
    if shuffle : random.shuffle(data_path)
    split_idx = int(0.5 * len(data_path))
    target_dataset = data_path[:split_idx]
    ref_dataset = data_path[split_idx:]
    print(f"Number of target maestro : {len(target_dataset)}")
    print(f"Number of ref maestro : {len(ref_dataset)}")

    return target_dataset, ref_dataset
