from dataclasses import replace
import unittest

import numpy as np

from audio_engine import MonitorBuffer
from dsp import Controls, PitchTracker, PresetEffects, VoiceProcessor, nearest_note
from presets import DRY, PRESETS


RATE = 48000
BLOCK = 256


def tone(freq, seconds=1, rate=RATE):
    time = np.arange(round(rate * seconds)) / rate
    return (.15 * np.sin(2 * np.pi * freq * time)).astype(np.float32)


def process(samples, preset=DRY, controls=Controls(), rate=RATE):
    processor = VoiceProcessor(rate, controls.gain_db)
    effects = PresetEffects(preset, rate)
    return np.concatenate([
        processor.process(samples[start:start + BLOCK], effects, controls)
        for start in range(0, len(samples), BLOCK)
    ])


def dominant_hz(samples, rate=RATE):
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    return np.argmax(spectrum) * rate / len(samples)


class DspTest(unittest.TestCase):
    def test_dry_bypass_and_gain(self):
        original = tone(220, .08)
        np.testing.assert_array_equal(process(original), original)
        np.testing.assert_allclose(process(original, controls=Controls(gain_db=6)),
                                   original * (10 ** (6 / 20)), rtol=1e-6)

    def test_fifty_distinct_presets(self):
        self.assertEqual(len(PRESETS), 50)
        self.assertTrue({
            "Робот", "Демон", "Бурундук", "Анонимус", "Радио",
            "Эхо", "Киборг", "Глубокий бас", "Гелий",
        }.issubset(PRESETS))
        rng = np.random.default_rng(42)
        sample = tone(193, .45)
        sample += (.005 * rng.standard_normal(len(sample))).astype(np.float32)
        sample[round(.25 * RATE):] = 0
        fingerprints = set()
        for name, preset in PRESETS.items():
            with self.subTest(preset=name):
                out = process(sample, preset)
                self.assertEqual(out.shape, sample.shape)
                self.assertTrue(np.isfinite(out).all())
                self.assertLessEqual(float(np.abs(out).max()), 1.)
                self.assertFalse(np.allclose(out, sample))
                fingerprints.add(out.round(4).tobytes())
        self.assertEqual(len(fingerprints), 50)

    def test_pitch_shift_and_scale_correction(self):
        for rate in (44100, 48000):
            with self.subTest(rate=rate):
                source = tone(220, 1.4, rate)
                low = process(source, replace(DRY, pitch=-7), rate=rate)
                high = process(source, replace(DRY, pitch=12), rate=rate)
                self.assertAlmostEqual(dominant_hz(low[-rate:], rate), 146.8, delta=4)
                self.assertAlmostEqual(dominant_hz(high[-rate:], rate), 440, delta=4)
                off_key = tone(452, 1.4, rate)
                fixed = process(off_key, controls=Controls(tune=True, scale="Мажор", amount=1), rate=rate)
                self.assertAlmostEqual(dominant_hz(fixed[-rate:], rate), 440, delta=3)
        c_sharp = 440 * 2 ** ((61 - 69) / 12)
        self.assertAlmostEqual(nearest_note(c_sharp, 1, "Мажор"), 0.)
        self.assertAlmostEqual(abs(nearest_note(c_sharp, 0, "Мажор")), 1.)

    def test_tracker_keeps_silence_unvoiced(self):
        tracker = PitchTracker(RATE)
        vocal = tone(112.7, .1)
        for start in range(0, len(vocal), BLOCK):
            estimate = tracker.update(vocal[start:start + BLOCK])
        self.assertAlmostEqual(estimate, 112.7, delta=1)
        for _ in range(12):
            estimate = tracker.update(np.zeros(BLOCK, dtype=np.float32))
        self.assertIsNone(estimate)

    def test_echo_tail_spans_blocks(self):
        impulse = np.zeros(RATE, dtype=np.float32)
        impulse[0] = .3
        out = process(impulse, PRESETS["Эхо"])
        repeat_at = round(RATE * PRESETS["Эхо"].echo_ms / 1000)
        self.assertGreater(out[repeat_at], .1)
        self.assertGreater(out[2 * repeat_at], .025)

    def test_monitor_consumes_each_sample_once_and_inserts_silence(self):
        monitor = MonitorBuffer()
        monitor.blocks.extend([np.array([.1, .2, .3]), np.array([.4])])
        out = np.empty((2, 2))
        monitor.fill(out)
        np.testing.assert_allclose(out, [[.1, .1], [.2, .2]])
        out = np.empty((3, 1))
        monitor.fill(out)
        np.testing.assert_allclose(out[:, 0], [.3, .4, 0.])


if __name__ == "__main__":
    unittest.main()
