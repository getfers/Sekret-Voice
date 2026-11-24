"""Negotiate compatible PortAudio streams and route only the processed voice."""

from collections import deque
from math import ceil, floor
import sys
from time import monotonic

import numpy as np
import sounddevice as sd

from dsp import Controls, PresetEffects, VoiceProcessor, sample_indices
from presets import DEFAULT_PRESET, PRESETS


def audio_worker(engine, requests, results):
    """Keep device setup and teardown in one COM apartment on Windows."""
    apartment = None
    if sys.platform == "win32":
        import ctypes

        ole32 = ctypes.WinDLL("ole32")
        if ole32.CoInitializeEx(None, 0) >= 0:
            apartment = ole32
    try:
        for task in iter(requests.get, None):
            action, complete = task
            try:
                results.put((complete, action(), None))
            except Exception as error:
                results.put((complete, None, error))
    finally:
        try:
            engine.stop()
        finally:
            if apartment is not None:
                apartment.CoUninitialize()


class AudioBridge:
    """Bounded buffer with resampling for independently clocked audio streams."""

    def __init__(self, source_rate, output_rate, stable):
        self.blocks = deque(maxlen=32)
        self.pending = np.empty(0, dtype=np.float32)
        self.ratio = source_rate / output_rate
        self.reserve = round(source_rate * (.024 if stable else .008))
        self.position = 0.
        self.primed = False
        self.adjustment = 0.

    def fill(self, out):
        out.fill(0)
        queued = len(self.pending) + sum(map(len, self.blocks.copy()))
        demand = len(out) * self.ratio
        if not self.primed:
            if queued < demand + self.reserve + 1:
                return False
            self.primed = True

        # Absorb clock drift gradually, with queue depth kept near one buffer plus the reserve.
        error = (queued - demand - self.reserve) / max(1, self.reserve)
        adjustment = max(-.004, min(.004, .002 * error))
        self.adjustment += (adjustment - self.adjustment) * .04
        step = self.ratio * (1 + self.adjustment)
        required = ceil(self.position + len(out) * step) + 1
        chunks = [self.pending] if len(self.pending) else []
        available = len(self.pending)
        while available < required:
            try:
                block = self.blocks.popleft()
            except IndexError:
                break
            chunks.append(block)
            available += len(block)
        count = min(len(out), max(0, floor((available - 1 - self.position) / step) + 1))
        if count:
            samples = chunks[0] if len(chunks) == 1 else np.concatenate(chunks)
            positions = self.position + step * sample_indices(count)
            left = positions.astype(np.intp)
            right = np.minimum(left + 1, available - 1)
            voice = samples[left] + (positions - left) * (samples[right] - samples[left])
            out[:count, :] = voice[:, None]
            consumed = self.position + count * step
            self.pending = samples[int(consumed):]
            self.position = consumed % 1
        if count < len(out):
            self.primed = False
            self.position = 0.
            return True
        return False


def sample_rates(*rates):
    return dict.fromkeys(rate for rate in (*rates, 48000, 44100, 32000) if rate and rate >= 8000)


def stream_profiles(endpoint, stable):
    if endpoint.wasapi:
        if stable:
            return ((256, .030), (0, "high"))
        return ((64, 0.), (0, "low"), (256, .030))
    return ((0, "high"),) if stable else ((0, "low"), (0, "high"))


def stream_options(endpoint, rate, block, latency):
    return dict(samplerate=rate, blocksize=block, latency=latency, dtype="float32",
                extra_settings=sd.WasapiSettings(auto_convert=True) if endpoint.wasapi else None)


class AudioEngine:
    def __init__(self):
        self.controls = Controls()
        self.preset = PRESETS[DEFAULT_PRESET]
        self.level = 0.0
        self.rate = 0
        self.xruns = 0
        self.error = ""
        self.output = None
        self.stable = False
        self.latency_ms = 0
        self._streams = ()
        self._monitor = None
        self._monitor_bridge = None
        self._effects = None
        self._recent_xruns = deque(maxlen=6)

    @property
    def running(self):
        return bool(self._streams) and all(stream.active for stream in self._streams)

    @property
    def monitoring(self):
        return self._monitor is not None and self._monitor.active

    @property
    def needs_more_buffer(self):
        return not self.stable and len(self._recent_xruns) == 6 and monotonic() - self._recent_xruns[0] < 1.25

    def set_preset(self, preset):
        if not preset.effects:
            raise ValueError("Выберите пресет с эффектом.")
        if self.rate:
            self._effects = PresetEffects(preset, self.rate)
        self.preset = preset

    def _note_status(self, status):
        if status and (status.input_overflow or status.output_underflow):
            self._record_xrun()

    def _record_xrun(self):
        self.xruns += 1
        self._recent_xruns.append(monotonic())

    def _processor(self, rate):
        self._effects = PresetEffects(self.preset, rate)
        return VoiceProcessor(rate, self.controls.gain_db)

    def _processed(self, processor, indata):
        try:
            samples = indata[:, 0]
            if indata.shape[1] > 1:
                samples = np.mean(indata, axis=1)
            voice = processor.process(samples, self._effects, self.controls)
            self.level = processor.level
            bridge = self._monitor_bridge
            if bridge is not None:
                bridge.blocks.append(voice)
            return voice
        except Exception as error:
            self.error = f"Ошибка обработки голоса: {error}"
            raise sd.CallbackAbort from error

    @staticmethod
    def _close(stream):
        if stream is not None:
            try:
                stream.abort()
            except sd.PortAudioError:
                pass
            finally:
                try:
                    stream.close()
                except sd.PortAudioError:
                    pass

    def _open_duplex(self, microphone, output, rate, block, latency):
        stream = None
        processor = None

        def callback(indata, outdata, frames, time, status):
            self._note_status(status)
            try:
                outdata[:] = self._processed(processor, indata)[:, None]
            except sd.CallbackAbort:
                outdata.fill(0)
                raise

        options = stream_options(microphone, rate, block, latency)
        options["extra_settings"] = options["extra_settings"], stream_options(output, rate, block, latency)["extra_settings"]
        for inputs in dict.fromkeys((1, min(2, microphone.channels))):
            try:
                stream = sd.Stream(device=(microphone.index, output.index), channels=(inputs, min(2, output.channels)),
                                   callback=callback, **options)
                processor = self._processor(rate)
                stream.start()
                return stream
            except (sd.PortAudioError, ValueError) as error:
                self._close(stream)
                stream = None
                failure = error
            except Exception:
                self._close(stream)
                raise
        raise failure

    def _open_playback(self, device, source_rate, *, stable, main_output):
        failure = None
        for endpoint in device.endpoints:
            for rate in sample_rates(source_rate, endpoint.rate):
                bridge = AudioBridge(source_rate, rate, stable)

                def callback(outdata, frames, time, status, bridge=bridge):
                    if main_output:
                        self._note_status(status)
                    if self.error:
                        outdata.fill(0)
                    elif bridge.fill(outdata) and main_output:
                        self._record_xrun()

                for block, latency in stream_profiles(endpoint, stable):
                    stream = None
                    try:
                        stream = sd.OutputStream(device=endpoint.index, channels=min(2, endpoint.channels),
                                                 callback=callback, **stream_options(endpoint, rate, block, latency))
                        stream.start()
                        return stream, bridge
                    except (sd.PortAudioError, ValueError) as error:
                        self._close(stream)
                        failure = failure or error
        raise RuntimeError(f"Не удалось открыть «{device.label}». {failure or ''}")

    def _open_capture(self, endpoint, rate, block, latency, output, stable):
        stream = playback = None
        processor = bridge = None

        def callback(indata, frames, time, status):
            self._note_status(status)
            bridge.blocks.append(self._processed(processor, indata))

        try:
            options = stream_options(endpoint, rate, block, latency)
            for channels in dict.fromkeys((1, min(2, endpoint.channels))):
                try:
                    stream = sd.InputStream(device=endpoint.index, channels=channels, callback=callback, **options)
                    break
                except (sd.PortAudioError, ValueError) as error:
                    failure = error
            if stream is None:
                raise failure
            playback, bridge = self._open_playback(output, rate, stable=stable, main_output=True)
            processor = self._processor(rate)
            stream.start()
            return stream, playback, bridge.reserve / rate
        except Exception:
            self._close(stream)
            self._close(playback)
            raise

    def start(self, microphone, output, *, stable=False):
        self.stop()
        self.error = ""
        self.xruns = 0
        self._recent_xruns.clear()
        self.stable = stable
        failure = None
        for source in microphone.endpoints:
            for sink in output.endpoints:
                if source.api != sink.api:
                    continue
                for rate in sample_rates(sink.rate, source.rate):
                    for block, latency in stream_profiles(source, stable):
                        try:
                            stream = self._open_duplex(source, sink, rate, block, latency)
                        except (sd.PortAudioError, ValueError) as error:
                            failure = failure or error
                            continue
                        self._streams = (stream,)
                        self.latency_ms = round(1000 * sum(stream.latency))
                        self.output = output
                        self.rate = rate
                        return
        for source in microphone.endpoints:
            for rate in sample_rates(source.rate):
                for block, latency in stream_profiles(source, stable):
                    try:
                        capture, playback, reserve = self._open_capture(source, rate, block, latency, output, stable)
                    except (sd.PortAudioError, ValueError, RuntimeError) as error:
                        failure = failure or error
                        continue
                    self._streams = capture, playback
                    self.latency_ms = round(1000 * (capture.latency + playback.latency + reserve))
                    self.output = output
                    self.rate = rate
                    return
        raise RuntimeError(f"Не удалось начать обработку. Проверьте доступ к микрофону и выберите рабочий выход. {failure or ''}")

    def start_monitor(self, device):
        if not self.running:
            raise ValueError("Сначала включите обработку голоса.")
        if device == self.output:
            raise ValueError("Этот выход уже воспроизводит обработанный голос.")
        self.stop_monitor()
        self._monitor, self._monitor_bridge = self._open_playback(
            device, self.rate, stable=self.stable, main_output=False,
        )

    def stop_monitor(self):
        self._monitor_bridge = None
        self._close(self._monitor)
        self._monitor = None

    def stop(self):
        self.stop_monitor()
        for stream in self._streams:
            self._close(stream)
        self._streams = ()
        self._effects = None
        self.level = 0.0
        self.rate = 0
        self.latency_ms = 0
        self.output = None
