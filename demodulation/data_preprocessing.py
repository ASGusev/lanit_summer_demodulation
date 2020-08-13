import json
from pathlib import Path
from typing import List, Dict, Tuple, Generator, Union, Callable

import numpy as np

import noise
from modulation import Modulator


SignalDataFrame = Tuple[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], np.ndarray]
SnrSampleSet = Dict[int, List[Tuple[np.ndarray, np.ndarray]]]
SnrSampleSetWithSGs = Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]


def _norm_wave(x):
    x -= x.mean()
    x /= x.std()


def _norm_spectrogram(x):
    x -= x.min(axis=0)
    x /= x.max(axis=0)


def get_generator(n_samples: int, sample_len: int, modulator: Modulator,
                  add_noise: Callable[[np.ndarray], np.ndarray] = None,
                  return_wave: bool = True, return_sg: bool = True, normalize: bool = False) \
        -> Callable[[], Generator[SignalDataFrame, None, None]]:
    def gen() -> Generator[SignalDataFrame, None, None]:
        for i in range(n_samples):
            data = np.random.randint(0, modulator.n_channels, sample_len, np.int32)
            target_one_hot = np.zeros((sample_len, modulator.n_channels), dtype=np.float32)
            np.put_along_axis(target_one_hot, np.expand_dims(data, 1), 1, axis=1)

            wave = modulator.modulate(data)
            if add_noise is not None:
                wave = add_noise(wave)
            wave = np.expand_dims(wave, 1)
            x = wave

            if return_sg:
                spectrogram = modulator.wave_to_spectrogram(wave.ravel()).T
                spectrogram = np.expand_dims(spectrogram, 2)
                if normalize:
                    _norm_spectrogram(spectrogram)
                x = (wave, spectrogram) if return_wave else spectrogram

            if normalize:
                _norm_wave(wave)

            yield x, target_one_hot
    return gen


def generate_signal(n_samples: int, sample_len: int, modulator: Modulator,
                    add_noise: Callable[[np.ndarray], np.ndarray] = None,
                    return_wave: bool = True, return_sg: bool = True, normalize: bool = False) \
        -> SignalDataFrame:
    waves, sgs, targets = [], [], []
    for x, tgt in get_generator(n_samples, sample_len, modulator, add_noise, return_wave, return_sg, normalize)():
        if return_wave:
            waves.append(x[0] if return_sg else x)
        if return_sg:
            sgs.append(x[1] if return_wave else x)
        targets.append(tgt)
    waves = np.float32(waves)
    sgs = np.float32(sgs)
    inputs = ((waves, sgs) if return_sg else waves) if return_wave else sgs
    targets = np.array(targets)
    return inputs, targets


def generate_samples_per_snr(n_samples: int, sample_len: int, modulator: Modulator, ratios: List[int]) -> \
            SnrSampleSet:
    snr_samples = {}
    for snr in ratios:
        samples = []
        for i in range(n_samples):
            sample = np.random.randint(0, modulator.n_channels, sample_len, np.int32)
            wave = modulator.modulate(sample)
            wave = noise.add_white_gaussian_noise(wave, snr)
            samples.append((sample, wave))
        snr_samples[snr] = samples
    return snr_samples


def load_samples_per_snr(path: Path) -> SnrSampleSet:
    with path.open() as samples_file:
        snr_samples = json.load(samples_file)
    return {
        int(snr): [
            (np.int32(data_point['sample']), np.float32(data_point['wave']))
            for data_point in samples
        ]
        for snr, samples in snr_samples.items()
    }


def preprocess_samples(samples: List[Tuple[np.ndarray, np.ndarray]], modulator: Modulator, normalize: bool = False) -> \
        Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    waves, sgs, messages = [], [], []
    for message, wave in samples:
        spectrogram = modulator.wave_to_spectrogram(wave).T
        if normalize:
            _norm_wave(wave)
            _norm_spectrogram(spectrogram)
        waves.append(wave)
        sgs.append(spectrogram)
        messages.append(message)
    waves, sgs, messages = np.float32(waves), np.float32(sgs), np.int32(messages)
    waves = np.expand_dims(waves, 2)
    sgs = np.expand_dims(sgs, 3)
    messages_one_hot = np.zeros((*messages.shape, modulator.n_channels), dtype=np.float32)
    np.put_along_axis(messages_one_hot, np.expand_dims(messages, 2), 1, axis=2)
    return (waves, sgs), messages_one_hot
