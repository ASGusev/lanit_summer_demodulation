import json
from pathlib import Path
from typing import List, Dict, Tuple, Generator, Union, Callable

import numpy as np

import noise
from modulation import Modulator


SignalDataFrame = Tuple[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], np.ndarray]
SnrSampleSet = Dict[int, List[Tuple[np.ndarray, np.ndarray]]]
SnrSampleSetWithSGs = Dict[int, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]


def get_generator(n_samples: int, sample_len: int, modulator: Modulator,
                  add_noise: Callable[[np.ndarray], np.ndarray] = None,
                  return_wave: bool = True, return_sg: bool = True) -> \
            Callable[[], Generator[SignalDataFrame, None, None]]:
    def gen() -> Generator[SignalDataFrame, None, None]:
        for i in range(n_samples):
            data = np.random.randint(0, modulator.n_channels, sample_len, np.int32)
            target_one_hot = np.zeros((sample_len, modulator.n_channels), dtype=np.float32)
            np.put_along_axis(target_one_hot, np.expand_dims(data, 1), 1, axis=1)

            wave = modulator.modulate(data)
            if add_noise is not None:
                wave = add_noise(wave)
            x = np.expand_dims(wave, 1) if return_wave else None

            if return_sg:
                spectrogram = modulator.wave_to_spectrogram(wave).T
                spectrogram = np.expand_dims(spectrogram, 2)
                x = (x, spectrogram) if return_wave else spectrogram

            yield x, target_one_hot
    return gen


def generate_signal(n_samples: int, sample_len: int, modulator: Modulator,
                    add_noise: Callable[[np.ndarray], np.ndarray] = None,
                    return_wave: bool = True, return_sg: bool = True) -> SignalDataFrame:
    waves, sgs, targets = [], [], []
    for x, tgt in get_generator(n_samples, sample_len, modulator, add_noise, return_wave, return_sg)():
        if return_wave:
            waves.append(x[0] if return_sg else x)
        if return_sg:
            sgs.append(x[1] if return_wave else x)
        targets.append(tgt)
    waves = np.array(waves)
    sgs = np.array(sgs)
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
            (np.array(data_point['sample'], dtype=np.int32), np.array(data_point['wave'], dtype=np.float32))
            for data_point in samples
        ]
        for snr, samples in snr_samples.items()
    }


def add_sgs_to_samples(snr_samples: SnrSampleSet, modulator: Modulator) -> SnrSampleSetWithSGs:
    return {
        snr: [
            (sample, wave, modulator.wave_to_spectrogram(wave).T)
            for sample, wave in samples
        ]
        for snr, samples in snr_samples.items()
    }


def preprocess_samples_for_tf(snr_samples: SnrSampleSetWithSGs, modulator: Modulator) -> \
        Dict[int, SignalDataFrame]:
    reordered_snr_samples = {}
    for snr, samples in snr_samples.items():
        waves, sgs, messages = [], [], []
        for message, wave, sg in samples:
            waves.append(wave)
            sgs.append(sg)
            messages.append(message)
        waves, sgs, messages = np.array(waves), np.array(sgs), np.array(messages)
        waves = np.expand_dims(waves, 2)
        sgs = np.expand_dims(sgs, 3)
        messages_one_hot = np.zeros((*messages.shape, modulator.n_channels), dtype=np.float32)
        np.put_along_axis(messages_one_hot, np.expand_dims(messages, 2), 1, axis=2)
        reordered_snr_samples[snr] = ((waves, sgs), messages_one_hot)
    return reordered_snr_samples
