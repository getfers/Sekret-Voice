from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import sounddevice as sd

from audio_devices import Device, Endpoint, default_devices, group_devices
from audio_engine import AudioBridge, AudioEngine
from dsp import Controls, PitchTracker, PresetEffects, VoiceProcessor, nearest_note
from presets import DEFAULT_PRESET, PRESETS, Preset


RATE = 48000
BLOCK = 64
BYPASS = Preset("Test fixture for an isolated DSP stage")


def tone(freq, seconds=1, rate=RATE):
    time = np.arange(round(rate * seconds)) / rate
    return (.15 * np.sin(2 * np.pi * freq * time)).astype(np.float32)


def process(samples, preset=BYPASS, controls=Controls(), rate=RATE):
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
    def test_gain_in_isolated_processor(self):
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
                self.assertEqual(out.dtype, np.float32)
                self.assertTrue(np.isfinite(out).all())
                self.assertGreater(float(np.abs(out).max()), .001)
                self.assertLessEqual(float(np.abs(out).max()), 1.)
                self.assertFalse(np.allclose(out, sample))
                fingerprints.add(out.round(4).tobytes())
        self.assertEqual(len(fingerprints), 50)

    def test_pitch_shift_and_scale_correction(self):
        for rate in (44100, 48000):
            with self.subTest(rate=rate):
                source = tone(220, 1.4, rate)
                low = process(source, replace(BYPASS, pitch=-7), rate=rate)
                high = process(source, replace(BYPASS, pitch=12), rate=rate)
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
        for _ in range(round(.08 * RATE / BLOCK)):
            estimate = tracker.update(np.zeros(BLOCK, dtype=np.float32))
        self.assertIsNone(estimate)

    def test_echo_tail_spans_blocks_without_a_dry_signal(self):
        impulse = np.zeros(RATE, dtype=np.float32)
        impulse[0] = .3
        out = process(impulse, PRESETS["Эхо"])
        repeat_at = round(RATE * PRESETS["Эхо"].echo_ms / 1000)
        self.assertFalse(np.any(out[:repeat_at]))
        self.assertGreater(out[repeat_at], .1)
        self.assertGreater(out[2 * repeat_at], .025)

    def test_chorus_and_reverb_have_no_immediate_dry_signal(self):
        impulse = np.zeros(RATE // 10, dtype=np.float32)
        impulse[0] = .5
        for parameters in ({"chorus": .5}, {"reverb": .5}):
            with self.subTest(effect=parameters):
                wet = process(impulse, replace(BYPASS, **parameters))
                self.assertFalse(np.any(wet[:round(.002 * RATE)]))
                self.assertGreater(np.max(np.abs(wet)), .02)

    def test_preset_change_starts_at_previous_output_level(self):
        processor = VoiceProcessor(RATE)
        robot = PresetEffects(PRESETS["Робот"], RATE)
        echo = PresetEffects(PRESETS["Эхо"], RATE)
        samples = tone(170)[:BLOCK]
        previous = processor.process(samples, robot, Controls())
        changed = processor.process(samples, echo, Controls())
        self.assertAlmostEqual(float(changed[0]), float(previous[-1]))
        processor.process(samples, echo, Controls())
        processor.process(samples, echo, Controls())
        np.testing.assert_array_equal(processor.process(samples, echo, Controls()), 0.)


class DeviceTest(unittest.TestCase):
    def test_system_default_virtual_choice_and_backend_variants(self):
        entries = [
            ("MME", "Переназначение звуковых устр. - Output", 0, 2),
            ("MME", "Микрофон USB", 1, 0),
            ("MME", "Громкоговорители (Voice Changer", 0, 2),
            ("Windows DirectSound", "Первичный драйвер записи звука", 1, 0),
            ("Windows DirectSound", "Первичный звуковой драйвер", 0, 2),
            ("Windows DirectSound", "Громкоговорители (Voice Changer Audio)", 0, 2),
            ("Windows WASAPI", "Другой микрофон", 1, 0),
            ("Windows WASAPI", "Микрофон USB", 1, 0),
            ("Windows WASAPI", "Наушники USB", 0, 2),
            ("Windows WASAPI", "Громкоговорители (Voice Changer Audio)", 0, 2),
            ("Windows WASAPI", "CABLE Input (VB-Audio Virtual Cable)", 0, 2),
            ("Windows WDM-KS", "Output 2", 0, 2),
            ("Windows WDM-KS", "Аппаратный поток", 0, 2),
            ("Windows DirectSound", "Output 1", 0, 2),
            ("Windows DirectSound", "Наушники DirectSound", 0, 2),
        ]
        host_apis = [
            dict(name=api, devices=[i for i, entry in enumerate(entries) if entry[0] == api],
                 default_input_device=default_in, default_output_device=default_out)
            for api, default_in, default_out in (
                ("MME", 1, 0), ("Windows DirectSound", 3, 4), ("Windows WASAPI", 6, 8), ("Windows WDM-KS", -1, 11),
            )
        ]
        raw = [dict(name=name, max_input_channels=inputs, max_output_channels=outputs, default_samplerate=RATE)
               for _, name, inputs, outputs in entries]
        inputs, outputs = group_devices(raw, host_apis, (1, 0))
        self.assertEqual(len(inputs), 2)
        self.assertEqual(len(outputs), 4)
        microphone, cable, system_output = default_devices(inputs, outputs)
        self.assertEqual(microphone.label, "Другой микрофон")
        self.assertEqual(cable.label, "CABLE Input (VB-Audio Virtual Cable)")
        self.assertEqual(system_output.label, "Наушники USB")
        voice_changer = next(device for device in outputs if "Voice Changer" in device.label)
        self.assertEqual([endpoint.index for endpoint in voice_changer.endpoints], [9, 5, 2])
        self.assertEqual(default_devices(inputs, (system_output,))[1], system_output)


class FakeStream:
    def __init__(self, *, callback, blocksize, latency, **settings):
        self.callback = callback
        self.blocksize = blocksize
        self.latency = latency
        self.active = False

    def start(self):
        self.active = True

    def abort(self):
        self.active = False

    def close(self):
        self.active = False


class RoutingTest(unittest.TestCase):
    def test_bridge_resamples_and_silences_underruns(self):
        bridge = AudioBridge(8000, 16000, stable=False)
        out = np.ones((100, 2), dtype=np.float32)
        bridge.fill(out)
        np.testing.assert_array_equal(out, 0.)
        bridge.blocks.append(np.arange(115, dtype=np.float32) / 115)
        bridge.fill(out)
        np.testing.assert_allclose(out[:, 0], np.arange(100) / 230, atol=1e-4)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        bridge.fill(np.empty((400, 2), dtype=np.float32))
        self.assertFalse(bridge.primed)
        bridge.fill(out)
        np.testing.assert_array_equal(out, 0.)

    def test_duplex_output_is_wet_and_callback_failure_is_silent(self):
        mic = Device("Mic", (Endpoint(4, "Mic", 0, "Windows WASAPI", 1, RATE),))
        cable = Device("Cable", (Endpoint(5, "Cable", 0, "Windows WASAPI", 2, RATE),))
        engine = AudioEngine()
        self.assertEqual(engine.preset, PRESETS[DEFAULT_PRESET])
        with self.assertRaises(ValueError):
            engine.set_preset(BYPASS)
        with patch("audio_engine.sd.Stream", side_effect=lambda **params: FakeStream(**(params | {"latency": (.01, .01)}))):
            engine.start(mic, cable)
        self.addCleanup(engine.stop)
        stream = engine._streams[0]
        samples = tone(190)[:BLOCK, None]
        out = np.full((BLOCK, 2), np.nan, dtype=np.float32)
        stream.callback(samples, out, BLOCK, None, None)
        np.testing.assert_array_equal(out[:, 0], out[:, 1])
        self.assertFalse(np.allclose(out[:, 0], samples[:, 0]))
        self.assertEqual(stream.blocksize, 64)
        with patch.object(engine._effects, "process", side_effect=ValueError("invalid samples")):
            with self.assertRaises(sd.CallbackAbort):
                stream.callback(samples, out, BLOCK, None, None)
        np.testing.assert_array_equal(out, 0.)

    def test_separate_endpoints_with_different_backends_keep_processing(self):
        mic = Device("Mic", (Endpoint(4, "Mic", 0, "Windows WASAPI", 1, RATE),))
        cable = Device("Cable", (Endpoint(5, "Cable", 1, "MME", 2, 44100),))
        engine = AudioEngine()
        with patch("audio_engine.sd.InputStream", side_effect=lambda **params: FakeStream(**(params | {"latency": .01}))), \
             patch("audio_engine.sd.OutputStream", side_effect=lambda **params: FakeStream(**(params | {"latency": .02}))), \
             patch("audio_engine.sd.Stream") as duplex:
            engine.start(mic, cable)
        self.addCleanup(engine.stop)
        duplex.assert_not_called()
        capture, playback = engine._streams
        self.assertTrue(engine.running)
        for start in range(0, BLOCK * 12, BLOCK):
            samples = tone(160)[start:start + BLOCK, None]
            capture.callback(samples, BLOCK, None, None)
        out = np.full((BLOCK, 2), np.nan, dtype=np.float32)
        playback.callback(out, BLOCK, None, None)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreater(np.max(np.abs(out)), .001)
        self.assertFalse(np.allclose(out[:, 0], samples[:, 0]))


if __name__ == "__main__":
    unittest.main()
