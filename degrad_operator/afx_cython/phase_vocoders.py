"""
phase vocoders
currently supporting:
    * pitch shift
    * (TODO) random phase (ghost voice)
"""

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from afx.delay_and_reverb import overlap_add
from afx.primitives import get_signal, gain_stage
from afx.augment import apply_pedalboard_effect
import flags as FLAGS
import librosa
import scipy
from scipy import signal
from scipy.stats import truncnorm

# @partial(jax.jit, backend='cpu')
#def pitchshift(x, semitones=12.0, cepstrum_compensation=True, liftering_cutoff=50):
#    def princarg(x):
#        return (x + np.pi) % (2 * np.pi) - np.pi
#
#    n_fft, hop_length = 2048, 512
#    c = x.shape[-1]
#    _, _, X = jax.scipy.signal.stft(
#        x.T, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
#    )  # c f t
#    ratio = 2 ** (semitones / 12)
#    arange = jnp.arange(1 + n_fft // 2)[:, None]
#    omega = 2 * np.pi * arange * hop_length / n_fft
#    phase_comp = 2 * np.pi * arange / 2
#    arange = jnp.arange(1, hop_length + 1)[None, None, :]
#
#    def pitchshift_step(state, X):
#        phi_0, psi, r_0 = state
#        r, phi = jnp.abs(X), jnp.angle(X) + phase_comp
#        delta_phi = omega + princarg(phi - phi_0 - omega)
#        delta_r = (r - r_0) / hop_length
#        delta_psi = ratio * delta_phi / hop_length
#        r_0 = r_0[:, :, None] + arange * delta_r[:, :, None]
#        psi = psi[:, :, None] + arange * delta_psi[:, :, None]
#        res = jnp.sum(r_0 * jnp.cos(psi), 0)
#        phi_0 = phi
#        psi = princarg(psi[:, :, -1])
#        r_0 = r
#        return (phi_0, psi, r_0), res
#
#    z = jnp.zeros((1 + n_fft // 2, c))
#    y = lax.scan(pitchshift_step, (z, z, z), X.transpose(2, 1, 0))[1]
#    y = y.transpose(1, 0, 2)
#    y = overlap_add(y, hop_length)
#
#    def compensate_cepstrum(y, X):
#        def cepstrum_match(X_s, X_t):
#            def get_cepstrum(X):
#                cep_lpf = jnp.array(
#                    [1.0] + [2] * (liftering_cutoff - 1) + [0] * (n_fft - liftering_cutoff)
#                )[None, :, None]
#                x_cep = jnp.fft.irfft(jnp.log(jnp.abs(X) + 1e-5), axis=-2)
#                X_cep = jnp.exp(jnp.fft.rfft(x_cep * cep_lpf, axis=-2)) - 1e-5
#                return X_cep
#
#            X_s_cep, X_t_cep = get_cepstrum(X_s), get_cepstrum(X_t)
#            comp = X_t_cep / (X_s_cep + 1e-5)
#            return X_s * comp
#
#        _, _, Y = jax.scipy.signal.stft(
#            y, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
#        )
#        Y_comp = cepstrum_match(Y[:, :, : X.shape[-1]], X)
#        _, y = jax.scipy.signal.istft(
#            Y_comp, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
#        )
#        return y
#
#    y = compensate_cepstrum(y, X)
#    # y = lax.cond(cepstrum_compensation,
#    #             lambda _: compensate_cepstrum(y, X),
#    #             lambda _: y, None)
#    return y.T



def pitchshift(x, semitones=12.0):
    return apply_pedalboard_effect(np.array(x).T, "pitchshift", FLAGS.sr, semitones=semitones).T


def ghost(x, key, amount=1, n_fft=2048, hop_length=512):
    c = x.shape[-1]
    _, _, X = jax.scipy.signal.stft(
        x.T, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length
    )  # c f t
    X_mag, X_pha = jnp.abs(X), jnp.angle(X)
    Y_pha = X_pha + jax.random.normal(key, shape=X_pha.shape) * np.pi * amount
    Y = X_mag * jnp.exp(1j * Y_pha)
    _, y = jax.scipy.signal.istft(Y, window="hann", nperseg=n_fft, noverlap=n_fft - hop_length)
    return y.T


def stft(x, n_fft=2048, hop_ratio=0.25):
    assert x.shape[-1] == 1
    x = np.array(x, np.float32)
    noverlap = int(n_fft*(1-hop_ratio))
    _, _, X = signal.stft(x[:, 0], nperseg=n_fft, noverlap=noverlap)
    return X

def istft(X, n_fft=2048, hop_ratio=0.25):
    noverlap = int(n_fft*(1-hop_ratio))
    _, x = signal.istft(X, nperseg=n_fft, noverlap=noverlap)
    return x[:, None]

def match_len(x, y):
    if x.ndim == 2: x = x[:, 0]
    if y.ndim == 2: y = y[:, 0]
    if len(x) <= len(y):
        return y[:len(x), None]
    else:
        return jnp.pad(y, (0, len(x)-len(y)))[:, None]

def spectral_convolve(x, sigma_f=4, sigma_t=4, **spec_kwargs):
    """
    convolve a window in the magnitude domain.
    """
    X = stft(x, **spec_kwargs)
    mag, phase = np.abs(X), np.angle(X)
    mag_convolved = scipy.ndimage.gaussian_filter(mag, sigma=(sigma_f, sigma_t))
    X_conv = mag_convolved*(np.cos(phase)+1j*np.sin(phase))
    x_conv = istft(X_conv, **spec_kwargs)
    return x_conv[:len(x)]

def griffin_lim(x, n_fft=2048, hop_ratio=0.25, n_iter=16):
    """
    apply griffin-lim
    """
    hop_length = int(n_fft*hop_ratio)
    n_fft = int(n_fft)
    X = np.abs(librosa.stft(np.array(x[:, 0]), n_fft=n_fft, hop_length=hop_length))
    x_inv = librosa.griffinlim(X, n_iter=int(n_iter), n_fft=n_fft, hop_length=hop_length)
    return match_len(x, x_inv)

def phase_randomization(x, mix=0.4, **spec_kwargs):
    """
    add randomness to the phase.
    """
    X = stft(x, **spec_kwargs)
    mag, phase = np.abs(X), np.angle(X)
    rand_phase = np.random.uniform(-3.14, 3.14, size=phase.shape)
    phase = mix*rand_phase+(1-mix)*phase
    X_rand = mag*(np.cos(phase)+1j*np.sin(phase))
    x_rand = istft(X_rand, **spec_kwargs)
    return x_rand[:len(x)]

def spectral_noise(x, strength_db=0, **spec_kwargs):
    """
    add randomness to the magnitude.
    """
    X = stft(x, **spec_kwargs)
    mag, phase = np.abs(X), np.angle(X)

    rand_mag_db = strength_db*truncnorm.rvs(-1, 1, 0, 1, size=mag.shape)
    rand_mag = 10**(rand_mag_db/20)
    mag = rand_mag*mag
    X_rand = mag*(np.cos(phase)+1j*np.sin(phase))
    x_rand = istft(X_rand, **spec_kwargs)
    return x_rand[:len(x)]

def phase_shuffle(x, **spec_kwargs):
    """
    shuffle accross the freq axis.
    """
    X = stft(x, **spec_kwargs)
    mag, phase = np.abs(X), np.angle(X)
    f = X.shape[-2]
    permute = np.random.permutation(f)
    phase = phase[permute, :]
    X_rand = mag*(np.cos(phase)+1j*np.sin(phase))
    x_rand = istft(X_rand, **spec_kwargs)
    return x_rand[:len(x)]

def spectral_holes(x, hole_type="time", num_holes=4, hole_f=4, hole_t=4, **spec_kwargs):
    """
    specaug-like spectral holes
    """
    X = stft(x, **spec_kwargs)
    f, t = X.shape[-2], X.shape[-1]
    hole_f, hole_t = int(hole_f), int(hole_t)
    for i in range(num_holes):
        f_from, t_from = np.random.randint(f-hole_f), np.random.randint(t-hole_t)
        if hole_type == "time":
            X[:, t_from:t_from+hole_t] = 0.
        elif hole_type == "freq":
            X[f_from:f_from+hole_f, :] = 0.
        elif hole_type == "timefreq":
            X[f_from:f_from+hole_f, t_from:t_from+hole_t] = 0.
    x_hole = istft(X, **spec_kwargs)
    return x_hole[:len(x)]


def apply_phase_vocoder(input_signal, afx_type, gain_staging, **param_dict):
    x = get_signal(input_signal, "main")
    if afx_type == "pitchshift":
        y = gain_stage(x, pitchshift(x, **param_dict))
        y = y[: len(x)]
        return {"main": y}
    elif afx_type == "ghost":
        key = jax.random.PRNGKey(np.random.randint(1000))
        y = ghost(x, key, **param_dict)
    elif afx_type == "spectral_convolve":
        y = spectral_convolve(x, **param_dict)
    elif afx_type == "griffin_lim":
        y = griffin_lim(x, **param_dict)
    elif afx_type == "phase_randomization":
        y = phase_randomization(x, **param_dict)
    elif afx_type == "spectral_noise":
        y = spectral_noise(x, **param_dict)
    elif afx_type == "phase_shuffle":
        y = phase_shuffle(x, **param_dict)
    elif afx_type == "spectral_holes":
        y = spectral_holes(x, **param_dict)
    else:
        assert False
    if gain_staging: y = gain_stage(x, y)
    return {"main": y}
