"""Block-based, mono DSP. Each processor belongs to one audio stream."""

from dataclasses import dataclass
from math import exp, pi

import numpy as np
from scipy import fft, signal

from presets import Preset


NOTES = ("C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B")
SCALES = {
    "Хроматическая": tuple(range(12)),
    "Мажор": (0, 2, 4, 5, 7, 9, 11),
    "Минор": (0, 2, 3, 5, 7, 8, 10),
    "Мажорная пентатоника": (0, 2, 4, 7, 9),
    "Минорная пентатоника": (0, 3, 5, 7, 10),
}


@dataclass(frozen=True, slots=True)
class Controls:
    gain_db: float = 0.0
    tune: bool = False
    root: int = 0
    scale: str = "Хроматическая"
    amount: float = 0.7


class Filter:
    def __init__(self, sos):
        self.sos = np.asarray(sos, dtype=np.float32)
        self.state = np.zeros((len(self.sos), 2), dtype=np.float32)

    def process(self, samples):
        out, self.state = signal.sosfilt(self.sos, samples, zi=self.state)
        return out


class PitchTracker:
    """YIN-style difference function evaluated with an FFT on recent audio."""

    def __init__(self, rate: int):
        self.decimation = max(1, round(rate / 12000))
        self.rate = rate / self.decimation
        self.window = np.zeros(round(self.rate * .043) * self.decimation, np.float32)
        self.fft_size = fft.next_fast_len(2 * len(self.window) // self.decimation)
        self.min_lag = max(2, int(self.rate / 1000))
        self.max_lag = int(self.rate / 65)
        self.hop = round(rate * .012)
        self.pending = 0
        self.filled = 0
        self.hz = None

    def reset(self):
        self.pending = self.filled = 0
        self.hz = None

    def update(self, samples):
        count = min(len(samples), len(self.window))
        self.window[:-count] = self.window[count:]
        self.window[-count:] = samples[-count:]
        self.filled = min(len(self.window), self.filled + count)
        self.pending += count
        if self.filled == len(self.window) and self.pending >= self.hop:
            self.pending %= self.hop
            self.hz = self.detect()
        return self.hz

    def detect(self):
        block = self.window.reshape(-1, self.decimation).mean(axis=1)
        block -= block.mean()
        if np.dot(block, block) < len(block) * .002**2:
            return None

        spectrum = fft.rfft(block, n=self.fft_size)
        correlation = fft.irfft(spectrum.real**2 + spectrum.imag**2, n=self.fft_size)
        lags = np.arange(1, self.max_lag + 2)
        energy = np.r_[0., np.cumsum(block**2, dtype=np.float64)]
        difference = energy[-1] - energy[lags] + energy[len(block) - lags] - 2 * correlation[lags]
        np.maximum(difference, 0, out=difference)
        normalized = np.r_[1., difference * lags / np.maximum(np.cumsum(difference), 1e-12)]
        candidates = np.flatnonzero(
            (normalized[self.min_lag:-1] < .18)
            & (normalized[self.min_lag:-1] <= normalized[self.min_lag - 1:-2])
            & (normalized[self.min_lag:-1] < normalized[self.min_lag + 1:])
        )
        if not candidates.size:
            return None
        lag = int(candidates[0]) + self.min_lag
        left, center, right = normalized[lag - 1:lag + 2]
        offset = .5 * (left - right) / max(left - 2 * center + right, 1e-12)
        return self.rate / (lag + offset)


def nearest_note(hz: float, root: int, scale: str) -> float:
    midi = 69 + 12 * np.log2(hz / 440)
    intervals = SCALES[scale]
    nearest = min(
        (note for note in range(round(midi) - 6, round(midi) + 7) if (note - root) % 12 in intervals),
        key=lambda note: abs(note - midi),
    )
    return float(nearest - midi)


class VariableDelay:
    def __init__(self, maximum):
        self.history = np.zeros(int(maximum) + 2, np.float32)

    def read(self, samples, delay):
        combined = np.concatenate((self.history, samples))
        positions = len(self.history) + np.arange(len(samples), dtype=np.float32) - delay
        left = positions.astype(np.intp)
        fraction = positions - left
        right = np.minimum(left + 1, len(combined) - 1)
        self.history[:] = combined[-len(self.history):]
        return (combined[left] + fraction * (combined[right] - combined[left])).astype(np.float32)


class PitchShifter:
    """Continuous two-head delay with overlapping Hann envelopes."""

    def __init__(self, rate):
        self.span = round(rate * .027)
        self.minimum = round(rate * .002)
        self.delay = VariableDelay(self.span + self.minimum)
        self.phase = .25

    def process(self, samples, ratio, active):
        # Keep recent samples when bypassed so enabling tuning need not fill a new delay.
        if not active:
            history = self.delay.history
            count = min(len(history), len(samples))
            history[:-count] = history[count:]
            history[-count:] = samples[-count:]
            return samples
        step = (1 - ratio) / self.span
        phase = (self.phase + np.arange(len(samples)) * step) % 1
        self.phase = (self.phase + len(samples) * step) % 1
        second = (phase + .5) % 1
        blend = np.sin(pi * phase) ** 2
        middle = np.full_like(phase, .5)
        first_tap, second_tap, central = self.delay.read(
            samples, self.minimum + self.span * np.stack((phase, second, middle)),
        )
        shifted = first_tap * blend + second_tap * (1 - blend)
        mix = min(1., abs(ratio - 1.) / .016)
        return (central + mix * (shifted - central)).astype(np.float32)


class FeedbackDelay:
    def __init__(self, samples: int, feedback: float):
        self.buffer = np.zeros(max(1, samples), dtype=np.float32)
        self.feedback = feedback
        self.cursor = 0

    def process(self, samples):
        result = np.empty_like(samples)
        start = 0
        while start < len(samples):
            size = min(len(samples) - start, len(self.buffer) - self.cursor)
            end = start + size
            cells = self.buffer[self.cursor:self.cursor + size]
            result[start:end] = cells
            cells *= self.feedback
            cells += samples[start:end]
            self.cursor = (self.cursor + size) % len(self.buffer)
            start = end
        return result


class PresetEffects:
    def __init__(self, preset: Preset, rate: int):
        self.preset = preset
        self.rate = rate
        filters = []
        for freq, kind in ((preset.highpass, "highpass"), (preset.lowpass, "lowpass")):
            if freq:
                filters.extend(signal.butter(2, min(freq, rate * .45), kind, fs=rate, output="sos"))
        self.band = Filter(filters) if filters else None
        self.formants = []
        if preset.formant:
            ratio = 2 ** (preset.formant / 12)
            for freq, q in ((560, 2.2), (1470, 3.2), (2520, 4.)):
                for center, weight in ((freq, -.34), (freq * ratio, .48)):
                    resonator = signal.iirpeak(min(center, rate * .45), q, fs=rate)
                    self.formants.append((Filter(signal.tf2sos(*resonator)), weight))
        self.chorus = VariableDelay(rate * .026) if preset.chorus else None
        self.echo = FeedbackDelay(round(preset.echo_ms * rate / 1000), .34) if preset.echo else None
        self.combs = [
            FeedbackDelay(round(rate * seconds), .49 + preset.reverb * .22)
            for seconds in (.0297, .0371, .0411, .0437)
        ] if preset.reverb else []
        self.reverb_filter = (
            Filter(signal.butter(2, min(4300, rate * .45), fs=rate, output="sos"))
            if self.combs else None
        )
        self.sample_index = 0

    def process(self, samples):
        preset = self.preset
        index = self.sample_index + np.arange(len(samples))
        self.sample_index += len(samples)
        if self.formants:
            source = samples
            samples = source.copy()
            for resonator, weight in self.formants:
                samples += weight * resonator.process(source)
        if self.band is not None:
            samples = self.band.process(samples)
        if preset.ring_mix:
            carrier = np.sin(2 * pi * preset.ring_hz / self.rate * index).astype(np.float32)
            if preset.ring_hz < 15:
                carrier = .5 + .5 * carrier
            samples = samples * (1. - preset.ring_mix + preset.ring_mix * carrier)
        if preset.drive:
            saturation = np.tanh(samples * (1. + 6 * preset.drive))
            samples = samples * (1. - preset.drive) + saturation * preset.drive
        if self.chorus is not None:
            lfo = 2 * pi * preset.chorus_rate / self.rate * index
            delays = self.rate * np.stack((.014 + .004 * np.sin(lfo), .020 + .004 * np.cos(lfo)))
            voices = self.chorus.read(samples, delays)
            samples = samples * (1 - .4 * preset.chorus) + preset.chorus * .45 * (voices[0] + voices[1])
        if self.echo is not None:
            samples = samples + preset.echo * self.echo.process(samples)
        if self.reverb_filter is not None:
            wet = sum(comb.process(samples) for comb in self.combs) * .25
            samples = samples * (1. - .25 * preset.reverb) + self.reverb_filter.process(wet) * preset.reverb
        return samples


class VoiceProcessor:
    def __init__(self, rate: int, initial_gain_db: float = 0.0):
        self.rate = rate
        self.gain = 10 ** (initial_gain_db / 20)
        self.level = 0.0
        self.tracker = PitchTracker(rate)
        self.shifter = PitchShifter(rate)
        self.correction = 0.0
        self.was_tuning = False

    def process(self, samples, effects: PresetEffects, controls: Controls):
        desired_gain = 10 ** (controls.gain_db / 20)
        boosted = samples * np.linspace(self.gain, desired_gain, len(samples), dtype=np.float32)
        self.gain = desired_gain
        self.level = float(np.sqrt(np.dot(boosted, boosted) / len(boosted)))
        target = 0.0
        tuning = controls.tune and controls.amount > 0
        if tuning:
            hz = self.tracker.update(boosted)
            if hz is not None:
                target = nearest_note(hz, controls.root, controls.scale) * controls.amount
        elif self.was_tuning:
            self.tracker.reset()
        self.was_tuning = tuning
        response = .095 - .077 * controls.amount
        self.correction += (target - self.correction) * (1 - exp(-len(samples) / (self.rate * response)))
        correction = self.correction if tuning else 0.
        shift = effects.preset.pitch + correction
        voice = self.shifter.process(boosted, 2 ** (shift / 12), bool(effects.preset.pitch) or tuning)
        voice = effects.process(voice)
        magnitude = np.abs(voice)
        knee = .78
        limited = knee + (1 - knee) * np.tanh(np.maximum(magnitude - knee, 0) / (1 - knee))
        return np.where(magnitude > knee, np.copysign(limited, voice), voice)
