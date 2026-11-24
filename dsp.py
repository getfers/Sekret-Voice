"""Block-based, mono DSP. Each processor belongs to one audio stream."""

from dataclasses import dataclass
from functools import lru_cache
from math import exp, pi, sqrt

import numpy as np
from scipy import fft, signal

from presets import Preset, Room


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
    def __init__(self, samples: int, feedback: float, damping=0.):
        self.buffer = np.zeros(max(1, samples), dtype=np.float32)
        self.feedback = feedback
        self.cursor = 0
        self.decay_filter = None
        if damping:
            self.decay_filter = Filter([[1 - damping, 0, 0, 1, -damping, 0]])

    def process(self, samples):
        result = np.empty_like(samples)
        start = 0
        while start < len(samples):
            size = min(len(samples) - start, len(self.buffer) - self.cursor)
            end = start + size
            cells = self.buffer[self.cursor:self.cursor + size]
            result[start:end] = cells
            if self.decay_filter is not None:
                cells[:] = self.decay_filter.process(cells)
            cells *= self.feedback
            cells += samples[start:end]
            self.cursor = (self.cursor + size) % len(self.buffer)
            start = end
        return result


class SpectralVoice:
    """Change the vocal envelope or excite it with a new carrier using streaming overlap-add."""

    def __init__(self, rate, preset):
        self.rate = rate
        self.size = 1 << (max(256, round(rate * .019)) - 1).bit_length()
        self.hop = self.size // 4
        phases = 2 * pi * sample_indices(self.size) / self.size
        self.window = (.5 - .5 * np.cos(phases)).astype(np.float32)
        self.analysis = np.zeros(self.size, np.float32)
        self.synthesis = np.zeros_like(self.analysis)
        self.ready = np.zeros(self.hop, np.float32)
        self.pending = self.frame_offset = 0
        self.bins = sample_indices(self.size // 2 + 1)
        formant = preset.formant if preset.formant is not None else preset.pitch
        self.warp = 2 ** ((formant - preset.pitch) / 12)
        self.warp_bins = self.bins / self.warp
        frequencies = self.bins * rate / self.size
        self.warp_weight = np.clip((9000 - frequencies) / 4200, 0, 1)
        lifter_width = min(self.size // 2, max(4, round(rate * .0022)))
        cepstral_indices = np.minimum(sample_indices(self.size), self.size - sample_indices(self.size))
        self.lifter = .5 + .5 * np.cos(pi * np.minimum(1, cepstral_indices / lifter_width))
        self.carrier = preset.carrier
        self.hz = preset.carrier_hz
        self.random = np.random.default_rng(1837)
        self.noise = np.zeros_like(self.analysis)

    def _wave(self, frequency):
        step = frequency / self.rate
        start = (self.frame_offset * step) % 1.
        return (start + sample_indices(self.size) * step) % 1

    def _carrier(self):
        self.noise[:-self.hop] = self.noise[self.hop:]
        self.noise[-self.hop:] = self.random.standard_normal(self.hop, dtype=np.float32)
        if self.carrier == "noise":
            return self.noise
        phase = self._wave(self.hz)
        if self.carrier == "pulse":
            excitation = np.where(phase < .31, np.float32(.69), np.float32(-.31))
        elif self.carrier == "chord":
            excitation = (2 * phase - 1) + .38 * (2 * self._wave(1.5 * self.hz) - 1)
            excitation += .26 * (2 * self._wave(2 * self.hz) - 1)
        else:
            excitation = 2 * phase - 1
        return excitation + .085 * self.noise

    def _frame(self):
        windowed = self.analysis * self.window
        power = float(np.dot(windowed, windowed))
        if power < self.size * 1e-11:
            return None
        spectrum = fft.rfft(windowed)
        magnitude = np.abs(spectrum)
        floor = max(1e-6, float(magnitude.max()) * .0018)
        log_power = np.log(np.maximum(magnitude, floor))
        cepstrum = fft.irfft(log_power, n=self.size) * self.lifter
        envelope = fft.rfft(cepstrum).real
        warped = np.interp(self.warp_bins, self.bins, envelope).astype(np.float32)
        shifted = envelope + self.warp_weight * (warped - envelope)
        if self.carrier:
            carrier = fft.rfft(self._carrier() * self.window)
            response = np.exp(np.maximum(-7., .76 * (shifted - shifted.max())))
            frame = fft.irfft(carrier * response, n=self.size)
            scale = min(24., sqrt(power / max(1e-9, float(np.dot(frame, frame)))))
            return frame * scale
        response = np.exp(np.clip(shifted - envelope, -1.85, 1.85))
        return fft.irfft(spectrum * response, n=self.size)

    def process(self, samples):
        output = np.empty_like(samples)
        start = 0
        while start < len(samples):
            count = min(len(samples) - start, self.hop - self.pending)
            end = self.pending + count
            output[start:start + count] = self.ready[self.pending:end]
            offset = self.size - self.hop
            self.analysis[offset + self.pending:offset + end] = samples[start:start + count]
            self.pending = end
            start += count
            if self.pending == self.hop:
                frame = self._frame()
                if frame is not None:
                    self.synthesis += frame * self.window * (2 / 3)
                self.ready[:] = self.synthesis[:self.hop]
                self.synthesis[:-self.hop] = self.synthesis[self.hop:]
                self.synthesis[-self.hop:] = 0
                self.analysis[:-self.hop] = self.analysis[self.hop:]
                self.frame_offset += self.hop
                self.pending = 0
        return output


class BitCrusher:
    def __init__(self, rate, bits, target_rate):
        self.hold = max(1, round(rate / (target_rate or rate)))
        self.levels = (1 << (bits - 1)) - 1
        self.phase = 0
        self.last = 0.

    def process(self, samples):
        next_sample = (-self.phase) % self.hold
        picked = samples[next_sample::self.hold]
        # Companding keeps soft syllables above the first quantization step.
        compressed = np.sqrt(np.minimum(1., 1.9 * np.abs(picked)))
        steps = np.rint(compressed * self.levels) / self.levels
        quantized = np.copysign(steps * steps, picked)
        output = np.empty_like(samples)
        output[:next_sample] = self.last
        if len(quantized):
            output[next_sample:] = np.repeat(quantized, self.hold)[:len(samples) - next_sample]
            self.last = float(quantized[-1])
        self.phase = (self.phase + len(samples)) % self.hold
        return output


class RoomReverb:
    def __init__(self, room: Room, rate):
        damping = exp(-2 * pi * min(room.damping_hz, rate * .45) / rate)
        self.early = [
            (weight, FeedbackDelay(round(rate * (room.early_ms / 1000 + offset * room.size)), 0.))
            for offset, weight in ((0., .69), (.0103, .32), (.0197, -.19))
        ]
        self.late = []
        for seconds in (.0211, .0297, .0367, .0433, .0503):
            interval = seconds * room.size
            feedback = min(.988, 10 ** (-3 * interval / room.decay))
            self.late.append(FeedbackDelay(round(rate * interval), feedback, damping))
        self.diffusion = [FeedbackDelay(round(rate * dt * min(1.8, room.size)), .54) for dt in (.0047, .0067)]
        self.damping = Filter(signal.butter(2, min(room.damping_hz, rate * .45), fs=rate, output="sos"))

    def process(self, samples):
        early = sum(weight * tap.process(samples) for weight, tap in self.early)
        tail = sum(tap.process(samples) for tap in self.late) * .105
        for diffuser in self.diffusion:
            delayed = diffuser.process(tail)
            tail = (1 - diffuser.feedback**2) * delayed - diffuser.feedback * tail
        return self.damping.process(early + tail)


def peaking_eq(rate, hz, gain_db, q):
    """Constant-Q peaking filter coefficients in SOS order."""
    angle = 2 * pi * min(hz, .45 * rate) / rate
    gain = 10 ** (gain_db / 40)
    alpha = np.sin(angle) / (2 * q)
    middle = -2 * np.cos(angle)
    coefficients = np.array((1 + alpha * gain, middle, 1 - alpha * gain,
                             1 + alpha / gain, middle, 1 - alpha / gain))
    return coefficients / coefficients[3]


class PresetEffects:
    def __init__(self, preset: Preset, rate: int):
        self.preset = preset
        self.active = bool(preset.effects)
        self.rate = rate
        filters = []
        for freq, kind in ((preset.highpass, "highpass"), (preset.lowpass, "lowpass")):
            if freq:
                filters.extend(signal.butter(2, min(freq, rate * .45), kind, fs=rate, output="sos"))
        filters.extend(peaking_eq(rate, hz, gain, q) for hz, gain, q in preset.eq)
        self.band = Filter(filters) if filters else None
        self.spectral = SpectralVoice(rate, preset) if preset.formant is not None or preset.carrier else None
        self.drive_gain = 1 + 11 * preset.drive
        self.dc_filter = (
            Filter(signal.butter(1, 26, "highpass", fs=rate, output="sos"))
            if preset.drive and preset.distortion != "tube" else None
        )
        self.flange = VariableDelay(rate * .014) if preset.flange else None
        self.flange_weight = (.32 + .52 * abs(preset.flange)) * (-1 if preset.flange < 0 else 1)
        self.flutter = VariableDelay(rate * (.002 + .002 * preset.flutter_ms)) if preset.flutter_ms else None
        self.chorus = VariableDelay(rate * .040) if preset.chorus else None
        self.crusher = BitCrusher(rate, preset.bits, preset.crush_rate) if preset.bits else None
        self.echo = FeedbackDelay(round(preset.echo_ms * rate / 1000), .18 + .8 * preset.echo) if preset.echo else None
        self.reverb = RoomReverb(preset.room, rate) if preset.room else None
        self.sample_index = 0

    def _cycles(self, hz, indices):
        step = hz / self.rate
        return (self.sample_index * step) % 1 + step * indices

    def _distort(self, samples):
        driven = samples * self.drive_gain
        kind = self.preset.distortion
        if kind == "clip":
            shaped = np.clip(driven, -.64, .85)
        elif kind == "fold":
            shaped = 1 - np.abs(((driven + 1) % 4) - 2)
        else:
            shaped = np.tanh(driven)
        return shaped / sqrt(self.drive_gain)

    def process(self, samples):
        preset = self.preset
        indices = sample_indices(len(samples))
        if self.spectral is not None:
            samples = self.spectral.process(samples)
        if self.flutter is not None:
            phase = 2 * pi * self._cycles(preset.flutter_hz, indices)
            depth = preset.flutter_ms / 1000
            delay = self.rate * (.0008 + depth * (1 + .65 * np.sin(phase) + .23 * np.sin(2 * phase + 1.2)))
            samples, = self.flutter.read(samples, delay)
        if self.band is not None:
            samples = self.band.process(samples)
        if preset.ring_depth:
            phase = 2 * pi * self._cycles(preset.ring_hz, indices)
            carrier = np.tanh((1 + 4 * preset.ring_depth) * np.sin(phase))
            samples = samples * carrier
        if preset.drive:
            samples = self._distort(samples)
        if self.crusher is not None:
            samples = self.crusher.process(samples)
        if self.dc_filter is not None:
            samples = self.dc_filter.process(samples)
        if preset.tremolo_depth:
            phase = 2 * pi * self._cycles(preset.tremolo_hz, indices)
            samples = samples * (1 - .5 * preset.tremolo_depth * (1 - np.sin(phase)))
        if preset.chop_hz:
            phase = self._cycles(preset.chop_hz, indices) % 1
            gate = np.clip(phase * 45, 0, 1) * np.clip((.68 - phase) * 35, 0, 1)
            samples = samples * gate
        if self.flange is not None:
            phase = 2 * pi * self._cycles(preset.flange_rate, indices)
            sweep = .0014 + .0048 * abs(preset.flange) * (.5 + .5 * np.sin(phase))
            reference, moving = self.flange.read(samples, self.rate * .00072, self.rate * sweep)
            samples = .64 * reference + self.flange_weight * moving
        if self.chorus is not None:
            lfo = 2 * pi * self._cycles(preset.chorus_rate, indices)
            depth = .0015 + .0055 * preset.chorus
            first, second, third = self.chorus.read(
                samples, self.rate * (.010 + depth * np.sin(lfo)),
                self.rate * (.017 + depth * np.cos(lfo)),
                self.rate * (.026 + depth * np.sin(lfo + 2.1)),
            )
            samples = .53 * first + .39 * second + .28 * third
        if self.echo is not None:
            samples = self.echo.process(samples)
        if self.reverb is not None:
            samples = self.reverb.process(samples)
        self.sample_index += len(samples)
        return samples


class OutputLevel:
    """Match the smoothed wet level to the post-gain mic while keeping peak headroom."""

    def __init__(self, rate):
        self.rate = rate
        self.reference = self.wet = 0.
        self.gain = 1.

    def _envelope(self, seconds, previous, value, attack, release):
        tau = attack if value > previous else release
        return previous + (value - previous) * (1 - exp(-seconds / tau))

    def process(self, samples, input_rms):
        seconds = len(samples) / self.rate
        block_rms = sqrt(max(0., float(np.dot(samples, samples)) / len(samples)))
        self.reference = self._envelope(seconds, self.reference, input_rms, .026, .52)
        self.wet = self._envelope(seconds, self.wet, block_rms, .056, .28)
        if self.wet > .0002 and block_rms > .00012:
            target_rms = min(.17, .95 * self.reference)
            target_gain = max(.12, min(10., target_rms / self.wet))
            if self.reference < .002:
                target_gain = min(1., target_gain)
            goal = self._envelope(seconds, self.gain, target_gain, .18, .032)
            if abs(goal - self.gain) > .0001:
                ramp = self.gain + (goal - self.gain) / len(samples) * sample_indices(len(samples))
                self.gain = goal
                return samples * ramp
        return samples * self.gain


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
        self.output_level = OutputLevel(rate)

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
        amount = controls.amount if controls.tune else effects.preset.tune
        tuning = amount > 0
        tracking = bool(effects.preset.pitch) or tuning
        hz = self.tracker.update(boosted) if tracking else None
        if not tracking and self.was_tracking:
            self.tracker.reset()
        self.was_tracking = tracking
        if tuning:
            if hz is not None:
                target = nearest_note(hz, controls.root, controls.scale) * amount
            response = (.008 if effects.preset.tune else .018) + .077 * (1 - amount)
            self.correction += (target - self.correction) * (1 - exp(-len(samples) / (self.rate * response)))
        else:
            self.correction = 0.
        shift = effects.preset.pitch + self.correction
        voice = self.shifter.process(boosted, 2 ** (shift / 12) if shift else 1., tracking, hz)
        if effects is not self.last_effects:
            self.transition = self.fade_frames if self.last_effects is not None else 0
            self.anchor = self.last_sample
            self.last_effects = effects
            self.output_level = OutputLevel(self.rate)
        voice = effects.process(voice)
        if effects.active:
            voice = self.output_level.process(voice, self.level)
        magnitude = np.abs(voice)
        knee, ceiling = .72, .9
        if magnitude.max() > knee:
            above = magnitude > knee
            limited = knee + (ceiling - knee) * np.tanh((magnitude[above] - knee) / (ceiling - knee))
            np.minimum(limited, np.float32(ceiling), out=limited)
            voice = voice.copy()
            voice[above] = np.copysign(limited, voice[above])
        if self.transition:
            voice = voice.copy()
            length = min(len(voice), self.transition)
            progress = (self.fade_frames - self.transition + sample_indices(length)) / self.fade_frames
            voice[:length] = self.anchor + progress * (voice[:length] - self.anchor)
            self.transition -= length
        self.last_sample = float(voice[-1])
        return voice
