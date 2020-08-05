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
        self.points_per_bit = self.fd // self.symbols_per_second
        self.fourier_window_size = self.points_per_bit // fourier_windows_in_symbol
        self.fourier_window = np.ones(self.fourier_window_size)
        self.frequency_shift = int(frequency_shift * self.channel_width) if frequency_shift else None

    def modulate(self, data):
        n_points = len(data) * self.points_per_bit
        sampling_points = np.arange(0, n_points * self.sampling_step - EPS, self.sampling_step)

        base_phase = 2 * np.pi * sampling_points * (self.min_freq + self.channel_width)
        if self.frequency_shift:
            base_phase += self.frequency_shift
        modulation_deltas = np.repeat(data, self.points_per_bit) * self.channel_width
        delta_phase = np.cumsum(2 * np.pi * modulation_deltas * self.sampling_step)
        final_phase = base_phase + delta_phase
        return np.cos(final_phase).astype(np.float32)

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