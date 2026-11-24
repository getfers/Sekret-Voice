"""PortAudio device selection and independent local monitoring."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from dsp import Controls, PresetEffects, VoiceProcessor
from presets import DRY, Preset


@dataclass(frozen=True, slots=True)
class Device:
    index: int
    name: str
    api: int
    api_name: str
    inputs: int
    outputs: int
    rate: int

    @property
    def label(self):
        return f"{self.name}  ·  {self.api_name} [{self.index}]"

    @property
    def virtual(self):
        name = self.name.casefold()
        return any(word in name for word in ("cable", "voicemeeter", "virtual", "steam streaming"))


def devices() -> list[Device]:
    apis = sd.query_hostapis()
    return [
        Device(index, info["name"], info["hostapi"], apis[info["hostapi"]]["name"],
               info["max_input_channels"], info["max_output_channels"], round(info["default_samplerate"]))
        for index, info in enumerate(sd.query_devices())
        if info["max_input_channels"] or info["max_output_channels"]
    ]


def default_devices(available: list[Device]):
    input_id, output_id = sd.default.device
    inputs = [device for device in available if device.inputs]
    outputs = [device for device in available if device.outputs]
    windows_pairs = [
        (mic, speaker) for mic in inputs for speaker in outputs
        if mic.api == speaker.api and "WASAPI" in mic.api_name
    ]
    if windows_pairs:
        # Resolve the physical defaults again in WASAPI when PortAudio defaults to MME.
        names = {device.index: device.name.casefold() for device in available}
        return max(windows_pairs, key=lambda pair: (
            not pair[0].virtual,
            not pair[1].virtual,
            bool(names.get(input_id)) and pair[0].name.casefold().startswith(names[input_id]),
            bool(names.get(output_id)) and pair[1].name.casefold().startswith(names[output_id]),
        ))
    microphone = next((device for device in inputs if device.index == input_id), inputs[0] if inputs else None)
    matching = [device for device in outputs if microphone and device.api == microphone.api]
    speakers = matching or outputs
    output = next((device for device in speakers if device.index == output_id), speakers[0] if speakers else None)
    return microphone, output


class MonitorBuffer:
    """A bounded single-producer/single-consumer queue between audio callbacks."""

    def __init__(self):
        self.blocks = deque(maxlen=4)
        self.remaining = np.empty(0, dtype=np.float32)

    def fill(self, out):
        out.fill(0)
        start = 0
        while start < len(out):
            if not len(self.remaining):
                try:
                    self.remaining = self.blocks.popleft()
                except IndexError:
                    break
            count = min(len(self.remaining), len(out) - start)
            out[start:start + count, :] = self.remaining[:count, None]
            self.remaining = self.remaining[count:]
            start += count


class AudioEngine:
    def __init__(self):
        self.controls = Controls()
        self.preset = DRY
        self.level = 0.0
        self.rate = 0
        self.xruns = 0
        self.error = ""
        self.output = None
        self._stream = None
        self._monitor = None
        self._monitor_buffer = None
        self._effects = None

    @property
    def running(self):
        return self._stream is not None and self._stream.active

    @property
    def monitoring(self):
        return self._monitor is not None and self._monitor.active

    @property
    def latency_ms(self):
        return round(1000 * sum(self._stream.latency)) if self._stream is not None else 0

    def set_preset(self, preset: Preset):
        if self.rate:
            self._effects = PresetEffects(preset, self.rate)
        self.preset = preset

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

    def start(self, microphone: Device, output: Device, block_size: int):
        if microphone.api != output.api:
            raise ValueError("Выберите Input и Output с одинаковым звуковым API, например Windows WASAPI.")
        self.stop()
        self.error = ""
        self.xruns = 0
        failures = []
        channels = (1, min(2, output.outputs))
        for rate in dict.fromkeys((48000, output.rate, microphone.rate, 44100, 32000)):
            stream = None
            try:
                sd.check_input_settings(device=microphone.index, channels=channels[0], dtype="float32", samplerate=rate)
                sd.check_output_settings(device=output.index, channels=channels[1], dtype="float32", samplerate=rate)
                processor = VoiceProcessor(rate, self.controls.gain_db)
                self._effects = PresetEffects(self.preset, rate)

                def callback(indata, outdata, frames, time, status, processor=processor):
                    if status:
                        self.xruns += 1
                    try:
                        voice = processor.process(indata[:, 0], self._effects, self.controls)
                        self.level = processor.level
                        outdata[:] = voice[:, None]
                        monitor_buffer = self._monitor_buffer
                        if monitor_buffer is not None:
                            monitor_buffer.blocks.append(voice)
                    except Exception as error:
                        self.error = f"Ошибка обработки звука ({type(error).__name__}): {error}"
                        outdata.fill(0)
                        raise sd.CallbackAbort from error

                stream = sd.Stream(
                    device=(microphone.index, output.index), channels=channels, samplerate=rate,
                    dtype="float32", blocksize=block_size, latency="low", callback=callback,
                )
                stream.start()
            except (sd.PortAudioError, ValueError) as error:
                self._close(stream)
                failures.append(f"{rate} Гц: {error}")
                continue
            self._stream = stream
            self.output = output
            self.rate = rate
            return
        raise RuntimeError("Не удалось открыть аудиопоток. " + failures[0])

    def start_monitor(self, device: Device, block_size: int):
        if not self.running:
            raise ValueError("Сначала запустите основной поток.")
        if device.index == self.output.index:
            raise ValueError("Этот выход уже получает обработанный голос. Выберите другие наушники.")
        self.stop_monitor()
        buffer = MonitorBuffer()
        stream = None

        def callback(outdata, frames, time, status):
            buffer.fill(outdata)

        try:
            stream = sd.OutputStream(
                device=device.index, channels=min(2, device.outputs), samplerate=self.rate,
                dtype="float32", blocksize=block_size, latency="low", callback=callback,
            )
            self._monitor_buffer = buffer
            stream.start()
        except (sd.PortAudioError, ValueError):
            self._monitor_buffer = None
            self._close(stream)
            raise
        self._monitor = stream

    def stop_monitor(self):
        self._monitor_buffer = None
        self._close(self._monitor)
        self._monitor = None

    def stop(self):
        self.stop_monitor()
        self._close(self._stream)
        self._stream = None
        self._effects = None
        self.level = 0.0
        self.rate = 0
        self.output = None
