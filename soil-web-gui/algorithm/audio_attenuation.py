import numpy as np
import soundfile as sf
from scipy.signal import correlate, medfilt


def _to_mono(samples):
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 2:
        return samples.mean(axis=1)
    return samples


class Soundfile:
    def __init__(self, soundfilepath):
        samples, samplerate = sf.read(soundfilepath)
        # Audio scans can arrive as mono or stereo files; the attenuation pipeline uses one channel.
        self.soundfile = _to_mono(samples)
        self.samplerate = samplerate
        self.is_correlated = False
        self.kernel_size = None

    @property
    def length(self):
        return len(self.soundfile)

    @property
    def raw_fft(self):
        fft = np.fft.rfft(self.soundfile)
        if self.kernel_size is not None:
            fft = medfilt(np.abs(fft), kernel_size=self.kernel_size)
        return fft

    @property
    def window_fft(self):
        window = np.hanning(self.length)
        fft = np.fft.rfft(self.soundfile * window) / self.length
        if self.kernel_size is not None:
            fft = medfilt(np.abs(fft), kernel_size=self.kernel_size)
        return fft


def _pad(signal, target_len):
    pad_width = target_len - len(signal)
    if pad_width > 0:
        return np.pad(signal, (0, pad_width))
    return signal


def correlate_soundfiles(initial_signal, final_signal):
    # Align each measurement to the baseline before comparing their frequency response.
    correlation = correlate(
        in1=final_signal.soundfile,
        in2=initial_signal.soundfile,
        mode="full",
        method="auto",
    )
    lags = np.arange(-initial_signal.length + 1, final_signal.length)
    lag = lags[np.argmax(np.abs(correlation))]

    if lag > 0:
        final_signal.soundfile = final_signal.soundfile[lag:]
        initial_signal.soundfile = initial_signal.soundfile[: final_signal.length]
    else:
        initial_signal.soundfile = initial_signal.soundfile[-lag:]
        final_signal.soundfile = final_signal.soundfile[: initial_signal.length]

    max_len = max(initial_signal.length, final_signal.length)
    initial_signal.soundfile = _pad(initial_signal.soundfile, max_len)
    final_signal.soundfile = _pad(final_signal.soundfile, max_len)
    initial_signal.is_correlated = True
    final_signal.is_correlated = True


def scalarize_transfer_function(
    initial_signal,
    final_signal,
    desired_freq,
    use_window=True,
    kernel_size=11,
    sample_window=512,
):
    # Convert the strongest transfer response near one target frequency into a projection value.
    if kernel_size is not None:
        initial_signal.kernel_size = kernel_size
        final_signal.kernel_size = kernel_size

    if not initial_signal.is_correlated or not final_signal.is_correlated:
        correlate_soundfiles(initial_signal, final_signal)

    fft_in = initial_signal.window_fft if use_window else initial_signal.raw_fft
    fft_final = final_signal.window_fft if use_window else final_signal.raw_fft
    freqs = np.fft.rfftfreq(final_signal.length, 1 / final_signal.samplerate)
    desired_index = np.searchsorted(freqs, desired_freq, side="right")
    desired_index = min(desired_index, len(freqs) - 1)

    eps = 1e-12
    transfer = fft_final * np.conj(fft_in) / (np.abs(fft_in) ** 2 + eps)
    half_window = max(1, sample_window) // 2
    window_start = max(0, desired_index - half_window)
    window_end = min(len(transfer), desired_index + half_window)
    if window_end <= window_start:
        window_end = min(len(transfer), window_start + 1)

    transmission = np.max(np.abs(transfer[window_start:window_end])) ** 2
    return float(-np.log(transmission + eps))


def scale_attenuations(attenuations, rotations, detectors, scale_by_rows=False):
    # Percentile scaling reduces the effect of unusually loud or quiet measurements.
    values = np.asarray(attenuations, dtype=float)
    if values.size != rotations * detectors:
        raise ValueError(
            f"Expected {rotations * detectors} measurements, got {values.size}."
        )

    if scale_by_rows:
        rows = values.reshape(rotations, detectors)
        scaled_rows = []
        for row in rows:
            scaled_rows.append(_percentile_scale(row))
        return np.asarray(scaled_rows, dtype=np.uint8).reshape(-1)

    return _percentile_scale(values)


def _percentile_scale(values):
    low = np.percentile(values, 25)
    high = np.percentile(values, 75)
    if high <= low:
        high = float(np.max(values))
        low = float(np.min(values))
    if high <= low:
        return np.zeros_like(values, dtype=np.uint8)

    scaled = 255 * (values - low) / (high - low)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def measurements_to_p_vector(
    baseline_path,
    measurement_paths,
    desired_freq=2500,
    use_window=True,
    kernel_size=11,
    sample_window=512,
    rotations=6,
    detectors=13,
    scale_by_rows=False,
):
    # Recreate the baseline for every measurement because correlation mutates both waveforms.
    attenuations = []
    for measurement_path in sorted(measurement_paths):
        baseline = Soundfile(baseline_path)
        measurement = Soundfile(measurement_path)
        attenuations.append(
            scalarize_transfer_function(
                baseline,
                measurement,
                desired_freq=desired_freq,
                use_window=use_window,
                kernel_size=kernel_size,
                sample_window=sample_window,
            )
        )

    return scale_attenuations(attenuations, rotations, detectors, scale_by_rows)
