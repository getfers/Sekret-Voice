from collections import Counter
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import sounddevice as sd

from audio_devices import Device, Endpoint, default_devices, group_devices
from audio_engine import AudioBridge, AudioEngine
from dsp import Controls, PitchTracker, PresetEffects, VoiceProcessor, nearest_note
from presets import CATEGORIES, DEFAULT_PRESET, PRESETS, Preset, Room


RATE = 48000
BLOCK = 64
BYPASS = Preset("Test fixture for an isolated DSP stage")


def tone(freq, seconds=1, rate=RATE):
    time = np.arange(round(rate * seconds)) / rate
    return (.15 * np.sin(2 * np.pi * freq * time)).astype(np.float32)


def process(samples, preset=BYPASS, controls=Controls(), rate=RATE, block=BLOCK):
    processor = VoiceProcessor(rate, controls.gain_db)
    effects = PresetEffects(preset, rate)
    return np.concatenate([
        processor.process(samples[start:start + block], effects, controls)
        for start in range(0, len(samples), block)
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
        self.assertEqual(CATEGORIES, ("Персонажи", "Монстры", "Роботы", "Связь", "Пространство"))
        self.assertEqual(Counter(preset.category for preset in PRESETS.values()), dict.fromkeys(CATEGORIES, 10))
        self.assertEqual(min(preset.pitch for preset in PRESETS.values()), -12)
        self.assertEqual(max(preset.pitch for preset in PRESETS.values()), 12)
        rng = np.random.default_rng(42)
        sample = tone(193, .55)
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
                self.assertLessEqual(float(np.abs(out).max()), .9)
                self.assertFalse(np.allclose(out, sample))
                fingerprints.add(out.round(4).tobytes())
        self.assertEqual(len(fingerprints), 50)

    def test_all_presets_keep_headroom_with_loud_input_and_maximum_boost(self):
        for rate, block in ((32000, 127), (44100, 257), (48000, 64)):
            time = np.arange(round(.77 * rate)) / rate
            narrow = np.sin(2 * np.pi * 187 * time)
            signal = np.clip(.86 * narrow + .62 * np.sin(2 * np.pi * 1070 * time), -1, 1).astype(np.float32)
            signal[round(.44 * rate):] = 0
            for name, preset in PRESETS.items():
                with self.subTest(rate=rate, preset=name):
                    wet = process(signal, preset, Controls(gain_db=24), rate, block)
                    self.assertEqual(wet.shape, signal.shape)
                    self.assertTrue(np.isfinite(wet).all())
                    self.assertLessEqual(float(np.abs(wet).max()), .9)
                    self.assertGreater(np.max(np.abs(wet)), .003)

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
        echo = replace(BYPASS, echo=.62, echo_ms=300)
        out = process(impulse, echo)
        repeat_at = round(RATE * echo.echo_ms / 1000)
        self.assertFalse(np.any(out[:repeat_at]))
        self.assertGreater(out[repeat_at], .1)
        self.assertGreater(out[2 * repeat_at], .025)

    def test_chorus_and_reverb_have_no_immediate_dry_signal(self):
        impulse = np.zeros(RATE // 10, dtype=np.float32)
        impulse[0] = .5
        for parameters in ({"chorus": .5}, {"room": Room(.7, 1.8, 6, 5000)}):
            with self.subTest(effect=parameters):
                wet = process(impulse, replace(BYPASS, **parameters))
                self.assertFalse(np.any(wet[:round(.002 * RATE)]))
                self.assertGreater(np.max(np.abs(wet)), .02)

    def test_preset_change_starts_at_previous_output_level(self):
        processor = VoiceProcessor(RATE)
        robot = PresetEffects(PRESETS[DEFAULT_PRESET], RATE)
        echo = PresetEffects(PRESETS["Эхо в пещере"], RATE)
        audio = tone(170, .10)
        for start in range(0, len(audio), BLOCK):
            previous = processor.process(audio[start:start + BLOCK], robot, Controls())
        self.assertGreater(abs(float(previous[-1])), .001)
        samples = audio[:BLOCK]
        changed = processor.process(samples, echo, Controls())
        self.assertAlmostEqual(float(changed[0]), float(previous[-1]))
        processor.process(samples, echo, Controls())
        processor.process(samples, echo, Controls())
        np.testing.assert_array_equal(processor.process(samples, echo, Controls()), 0.)

    def test_formant_shift_moves_vowel_envelope_without_changing_fundamental(self):
        fundamental = 112
        time = np.arange(RATE) / RATE
        source = np.zeros(RATE)
        for harmonic in range(1, 35):
            hz = fundamental * harmonic
            vowel = np.exp(-.5 * ((hz - 625) / 235)**2) + .46 * np.exp(-.5 * ((hz - 1550) / 370)**2)
            source += vowel * np.sin(2 * np.pi * hz * time) / harmonic**.35
        source = (.17 * source / np.max(np.abs(source))).astype(np.float32)
        down = process(source, replace(BYPASS, formant=-6.))
        up = process(source, replace(BYPASS, formant=6.))

        def centroid(samples):
            steady = samples[RATE // 2:]
            weights = np.abs(np.fft.rfft(steady * np.hanning(len(steady))))**2
            bins = np.fft.rfftfreq(len(steady), 1 / RATE)
            return float(np.dot(weights, bins) / weights.sum())

        self.assertGreater(centroid(up), centroid(down) * 1.28)
        for processed in (down, up):
            tracker = PitchTracker(RATE)
            estimate = tracker.update(processed[-len(tracker.window):])
            self.assertAlmostEqual(estimate, fundamental, delta=1.0)

    def test_trap_preset_enables_pitch_correction_with_global_switch_off(self):
        source = tone(452, 1.4)
        corrected = process(source, PRESETS["Трэп-автотюн"])
        self.assertAlmostEqual(dominant_hz(corrected[-RATE:]), 440, delta=3)

    def test_level_matcher_brings_quiet_and_loud_effects_closer_without_flattening_gain(self):
        quiet = replace(BYPASS, eq=((200, -19, .65),))
        loud = replace(BYPASS, eq=((200, 17, .65),))
        source = tone(200, 1.4)
        tail = RATE // 3
        quiet_out = process(source, quiet)[-tail:]
        loud_out = process(source, loud)[-tail:]
        ratio = np.sqrt(np.mean(loud_out**2) / np.mean(quiet_out**2))
        self.assertLess(ratio, 1.55)
        quieter_input = process(.25 * source, quiet)[-tail:]
        self.assertGreater(np.sqrt(np.mean(quiet_out**2) / np.mean(quieter_input**2)), 3.3)


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
        outputs = []
        original = tone(160, .10)
        for start in range(0, BLOCK * 24, BLOCK):
            samples = original[start:start + BLOCK, None]
            capture.callback(samples, BLOCK, None, None)
            if start >= BLOCK * 8:
                chunk = np.full((BLOCK, 2), np.nan, dtype=np.float32)
                playback.callback(chunk, BLOCK, None, None)
                outputs.append(chunk)
        out = np.concatenate(outputs)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreater(np.max(np.abs(out)), .001)
        self.assertFalse(np.allclose(out[:, 0], original[:len(out)]))


if __name__ == "__main__":
    unittest.main()
