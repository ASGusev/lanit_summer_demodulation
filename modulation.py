import numpy as np
from scipy.signal import stft, butter, lfilter


EPS = 1e-9
DEFAULT_FD = 2 ** 14
DEFAULT_MIN_FREQ = 2048
DEFAULT_MAX_FREQ = 3072
DEFAULT_N_CHANNELS = 16
DEFAULT_SYMBOLS_PER_SECOND = 16
DEFAULT_FOURIER_WINDOWS_IN_SYMBOL = 2

DEFAULT_FILTER_CRITICAL_FREQUENCY = 1536
DEFAULT_FILTER_ORDER = 8
DEFAULT_BASE_FREQUENCY = 2048


class Modulator:
    def __init__(self, fd=DEFAULT_FD, min_freq=DEFAULT_MIN_FREQ, max_freq=DEFAULT_MAX_FREQ,
                 n_channels=DEFAULT_N_CHANNELS, symbols_per_second=DEFAULT_SYMBOLS_PER_SECOND,
                 fourier_windows_in_symbol=DEFAULT_FOURIER_WINDOWS_IN_SYMBOL, frequency_shift=None):
        self.fd = fd
        self.sampling_step = fd ** -1
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.n_channels = n_channels
        self.channel_width = (self.max_freq - self.min_freq) // self.n_channels
        self.symbols_per_second = symbols_per_second
        self.points_per_unit = self.fd // self.symbols_per_second
        self.fourier_window_size = self.points_per_unit // fourier_windows_in_symbol
        self.fourier_window = np.ones(self.fourier_window_size)
        self.channel_frequencies = np.arange(
            self.min_freq + self.channel_width, self.max_freq + EPS, self.channel_width)
        if frequency_shift:
            self.channel_frequencies += frequency_shift * self.channel_width
        self.channel_deltas = 2 * np.pi * self.channel_frequencies * self.sampling_step

    def modulate(self, data):
        phase_deltas = np.repeat(self.channel_deltas[data], self.points_per_unit)
        initial_phase = np.random.uniform(0, 2 * np.pi)
        phase_deltas[0] = initial_phase
        phase = np.cumsum(phase_deltas)
        return np.cos(phase).astype(np.float32)

    def wave_to_spectrogram(self, wave, cut_f=False, return_f=False):
        f, _, z = stft(wave, self.fd, nperseg=self.fourier_window_size, noverlap=self.fourier_window_size - 1,
                       window=self.fourier_window, padded=True, boundary='zeros')
        spectrogram = np.abs(z)[:, :len(wave)]
        if cut_f:
            f_mask = (self.min_freq <= f) & (f <= self.max_freq)
            f, spectrogram = f[f_mask], spectrogram[f_mask]
        return (spectrogram, f) if return_f else spectrogram


class FrequencyCorrector:
    def __init__(self, fd=DEFAULT_FD, filter_order=DEFAULT_FILTER_ORDER, base_frequency=DEFAULT_BASE_FREQUENCY,
                 filter_critical_frequency=DEFAULT_FILTER_CRITICAL_FREQUENCY):
        self.sampling_step = fd ** -1
        # noinspection PyTupleAssignmentBalance
        self.filter_b, self.filter_a = butter(filter_order, filter_critical_frequency, fs=fd)
        self.base_frequency = base_frequency

    def move_frequencies(self, wave, apply_filter=True):
        sampling_points = np.arange(0, len(wave) * self.sampling_step - EPS, self.sampling_step)
        sample_phases = 2 * np.pi * sampling_points * self.base_frequency
        ref_wave = np.cos(sample_phases)
        moved_wave = wave * ref_wave
        if apply_filter:
            moved_wave = self.filter(moved_wave)
        return moved_wave

    def filter(self, wave):
        return lfilter(self.filter_b, self.filter_a, wave)
