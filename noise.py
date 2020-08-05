import numpy as np
from typing import Union


def gen_white_gaussian_noise(n: int) -> np.ndarray:
    return np.random.normal(0, 1, n)


def add_noise(signal: np.ndarray, noise: np.ndarray, snr: Union[int, float]) -> np.ndarray:
    signal_amplitude = np.std(signal)
    noise_amplitude = np.std(noise)
    target_noise_amplitude = signal_amplitude / 10 ** (snr / 10)
    noise_adjusted = noise * (target_noise_amplitude / noise_amplitude)
    return signal + noise_adjusted


def add_white_gaussian_noise(signal: np.ndarray, snr: Union[int, float]) -> np.ndarray:
    noise = gen_white_gaussian_noise(len(signal))
    return add_noise(signal, noise, snr)
