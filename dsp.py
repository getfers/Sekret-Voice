"""Block-based, mono DSP. Each processor belongs to one audio stream."""

from dataclasses import dataclass
from functools import lru_cache
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


@lru_cache(maxsize=16)
def sample_indices(size):
    return np.arange(size, dtype=np.float32)


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
        self.window = np.zeros(round(self.rate * .036) * self.decimation, np.float32)
        self.fft_size = fft.next_fast_len(2 * len(self.window) // self.decimation)
        self.min_lag = max(2, int(self.rate / 1000))
        self.max_lag = int(self.rate / 65)
        self.lags = np.arange(1, self.max_lag + 2)
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
        lags = self.lags
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
        return float(self.rate / (lag + offset))


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
        self.buffer = np.empty(len(self.history) + 960, np.float32)

    def push(self, samples):
        count = min(len(self.history), len(samples))
        self.history[:-count] = self.history[count:]
        self.history[-count:] = samples[-count:]

    def read(self, samples, *delays):
        size = len(self.history) + len(samples)
        if size > len(self.buffer):
            self.buffer = np.empty(size, np.float32)
        combined = self.buffer[:size]
        combined[:len(self.history)] = self.history
        combined[len(self.history):] = samples
        self.history[:] = combined[-len(self.history):]
        positions = len(self.history) + sample_indices(len(samples))
        return tuple(self._tap(combined, positions - delay) for delay in delays)

    @staticmethod
    def _tap(combined, positions):
        integral = np.floor(positions)
        left = integral.astype(np.intp)
        fraction = positions - integral
        right = np.minimum(left + 1, len(combined) - 1)
        return combined[left] + fraction * (combined[right] - combined[left])


class PitchShifter:
    """Continuous two-head delay with overlapping Hann envelopes."""

    def __init__(self, rate):
        self.rate = rate
        self.nominal_span = rate * .022
        self.span = self.nominal_span
        self.maximum_span = rate * .033
        self.minimum = round(rate * .0015)
        self.delay = VariableDelay(self.maximum_span + self.minimum)
        self.phase = .25

    def process(self, samples, ratio, active, voiced_hz):
        # Keep recent samples when bypassed so enabling tuning need not fill a new delay.
        if not active:
            self.delay.push(samples)
            return samples
        span = self.nominal_span
        if voiced_hz:
            two_periods = 2 * self.rate / voiced_hz
            # Adjacent heads stay in phase on vowels when separated by whole periods.
            span = min(self.maximum_span, max(1, round(span / two_periods)) * two_periods)
        self.span += (span - self.span) * (1 - exp(-len(samples) / (self.rate * .024)))
        step = (1 - ratio) / self.span
        phase = (self.phase + sample_indices(len(samples)) * step) % 1
        self.phase = (self.phase + len(samples) * step) % 1
        second = (phase + .5) % 1
        blend = np.sin(pi * phase) ** 2
        taps = self.minimum + self.span * phase, self.minimum + self.span * second
        if abs(ratio - 1.) >= .016:
            first, second = self.delay.read(samples, *taps)
            return first * blend + second * (1 - blend)

        # Near unison, fade to a single delayed head to avoid comb filtering and preserve timing.
        first_tap, second_tap, central = self.delay.read(samples, *taps, self.minimum + self.span * .5)
        shifted = first_tap * blend + second_tap * (1 - blend)
        return central + abs(ratio - 1.) / .016 * (shifted - central)


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
            for freq, q, weight in ((560, .8, 1.15), (1470, 1.6, .85), (2520, 2.3, .6)):
                resonator = signal.iirpeak(min(freq * ratio, rate * .45), q, fs=rate)
                self.formants.append((Filter(signal.tf2sos(*resonator)), weight))
        self.chorus = VariableDelay(rate * .014) if preset.chorus else None
        self.echo = FeedbackDelay(round(preset.echo_ms * rate / 1000), .22 + .44 * preset.echo) if preset.echo else None
        self.combs = [
            FeedbackDelay(round(rate * seconds), .29 + preset.reverb * .50)
            for seconds in (.0043, .0117, .0191, .0277)
        ] if preset.reverb else []
        self.reverb_filter = (
            Filter(signal.butter(2, min(4300, rate * .45), fs=rate, output="sos"))
            if self.combs else None
        )
        self.ring_phase = 0.
        self.chorus_phase = 0.

    def process(self, samples):
        preset = self.preset
        if self.formants:
            samples = sum(weight * resonator.process(samples) for resonator, weight in self.formants)
        if self.band is not None:
            samples = self.band.process(samples)
        if preset.ring_depth:
            step = 2 * pi * preset.ring_hz / self.rate
            carrier = np.sin(self.ring_phase + step * sample_indices(len(samples)))
            self.ring_phase = (self.ring_phase + step * len(samples)) % (2 * pi)
            if preset.ring_hz < 15:
                carrier = (.5 + .5 * carrier) ** (1 + 3 * preset.ring_depth)
            else:
                carrier = np.tanh((1 + 3 * preset.ring_depth) * carrier)
            samples = samples * carrier
        if preset.drive:
            amount = 1. + 7 * preset.drive
            samples = np.tanh(samples * amount) / amount**.5
        if self.chorus is not None:
            step = 2 * pi * preset.chorus_rate / self.rate
            lfo = self.chorus_phase + step * sample_indices(len(samples))
            self.chorus_phase = (self.chorus_phase + step * len(samples)) % (2 * pi)
            depth = .001 + .003 * preset.chorus
            first, second = self.chorus.read(
                samples, self.rate * (.005 + depth * np.sin(lfo)), self.rate * (.010 + depth * np.cos(lfo)),
            )
            samples = .62 * first + .55 * second
        if self.echo is not None:
            samples = self.echo.process(samples) * (.80 + .20 * preset.echo)
        if self.reverb_filter is not None:
            wet = sum(weight * comb.process(samples) for weight, comb in zip((.57, .22, .19, .17), self.combs))
            samples = self.reverb_filter.process(wet)
        return samples


class VoiceProcessor:
    def __init__(self, rate: int, initial_gain_db: float = 0.0):
        self.rate = rate
        self.gain = 10 ** (initial_gain_db / 20)
        self.gain_db = initial_gain_db
        self.level = 0.0
        self.tracker = PitchTracker(rate)
        self.shifter = PitchShifter(rate)
        self.correction = 0.0
        self.was_tracking = False
        self.last_effects = None
        self.last_sample = self.anchor = 0.
        self.transition = 0
        self.fade_frames = max(1, round(rate * .003))

    def process(self, samples, effects: PresetEffects, controls: Controls):
        if not len(samples):
            return samples
        if self.gain_db == controls.gain_db:
            boosted = samples if self.gain == 1. else samples * self.gain
        else:
            self.gain_db = controls.gain_db
            desired_gain = 10 ** (self.gain_db / 20)
            boosted = samples * np.linspace(self.gain, desired_gain, len(samples), dtype=np.float32)
            self.gain = desired_gain
        self.level = float(np.sqrt(np.dot(boosted, boosted) / len(boosted)))
        target = 0.0
        tuning = controls.tune and controls.amount > 0
        tracking = bool(effects.preset.pitch) or tuning
        hz = self.tracker.update(boosted) if tracking else None
        if not tracking and self.was_tracking:
            self.tracker.reset()
        self.was_tracking = tracking
        if tuning:
            if hz is not None:
                target = nearest_note(hz, controls.root, controls.scale) * controls.amount
            response = .095 - .077 * controls.amount
            self.correction += (target - self.correction) * (1 - exp(-len(samples) / (self.rate * response)))
        else:
            self.correction = 0.
        shift = effects.preset.pitch + self.correction
        voice = self.shifter.process(boosted, 2 ** (shift / 12) if shift else 1., tracking, hz)
        voice = effects.process(voice)
        magnitude = np.abs(voice)
        knee = .78
        if magnitude.max() > knee:
            above = magnitude > knee
            limited = knee + (1 - knee) * np.tanh((magnitude[above] - knee) / (1 - knee))
            voice = voice.copy()
            voice[above] = np.copysign(limited, voice[above])

        if effects is not self.last_effects:
            self.transition = self.fade_frames if self.last_effects is not None else 0
            self.anchor = self.last_sample
            self.last_effects = effects
        if self.transition:
            voice = voice.copy()
            length = min(len(voice), self.transition)
            progress = (self.fade_frames - self.transition + sample_indices(length)) / self.fade_frames
            voice[:length] = self.anchor + progress * (voice[:length] - self.anchor)
            self.transition -= length
        self.last_sample = float(voice[-1])
        return voice
