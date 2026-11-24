"""Group PortAudio backends into named input and output choices."""

from dataclasses import dataclass
import re
import unicodedata

import sounddevice as sd


def backend_rank(name):
    return {"windows wasapi": 0, "windows directsound": 1, "mme": 2}.get(name.casefold(), 3)


def name_key(name):
    return " ".join(unicodedata.normalize("NFKC", name).casefold().split())


def system_alias(name):
    key = name_key(name)
    return key.startswith((
        "переназначение звуковых", "первичный звуковой драйвер", "первичный драйвер записи",
        "microsoft sound mapper", "sound mapper", "primary sound driver", "primary sound capture driver",
        "primary capture driver", "pilote de son principal", "pilote de capture audio principal",
        "primärer soundtreiber", "primärer soundaufnahmetreiber", "controlador primario de sonido",
        "driver audio principale",
    )) or bool(re.fullmatch(r"(?:input|output|вход|выход)\s*\d+(?:\s*\(\s*\))?", key))


@dataclass(frozen=True, slots=True)
class Endpoint:
    index: int
    name: str
    api: int
    api_name: str
    channels: int
    rate: int

    @property
    def wasapi(self):
        return backend_rank(self.api_name) == 0


@dataclass(frozen=True, slots=True)
class Device:
    label: str
    endpoints: tuple[Endpoint, ...]
    system_default: bool = False

    @property
    def cable_rank(self):
        name = name_key(self.label)
        if re.search(r"\b(?:cable|vb-audio|virtual audio cable|vac)\b", name):
            return 3
        if any(term in name for term in ("voice changer", "voicemeeter", "blackhole", "virtual audio", "виртуальн")):
            return 2
        return int(any(term in name for term in ("virtual", "loopback", "steelseries sonar")))

    @property
    def virtual(self):
        return bool(self.cable_rank) or "steam streaming" in name_key(self.label)

    @property
    def headphones(self):
        return any(word in name_key(self.label) for word in ("headphones", "headset", "наушник", "гарнитура"))


def group_devices(raw_devices, host_apis, default_ids):
    """Keep native endpoints visible and retain unambiguous legacy aliases for fallback."""
    ordered_apis = sorted(range(len(host_apis)), key=lambda api: backend_rank(host_apis[api]["name"]))
    grouped = []
    for direction, default_id in zip(("input", "output"), default_ids):
        groups = {}
        channels_key = f"max_{direction}_channels"
        default_key = f"default_{direction}_device"
        for api in ordered_apis:
            api_name = host_apis[api]["name"]
            if re.sub(r"[^a-z]", "", api_name.casefold()).endswith("wdmks"):
                continue
            for index in host_apis[api]["devices"]:
                entry = raw_devices[index]
                clean_name = " ".join(entry["name"].strip("\x00 ").split())
                if not entry[channels_key] or not clean_name or system_alias(clean_name):
                    continue
                key = name_key(clean_name)
                endpoint = Endpoint(index, clean_name, api, api_name, entry[channels_key],
                                    round(entry["default_samplerate"]))
                if api_name.casefold() == "mme" and key not in groups and len(clean_name) >= 30:
                    # MME's name buffer can cut off the end of the Windows friendly name.
                    matches = [name for name in groups if name.startswith(key)]
                    if len(matches) > 1:
                        continue
                    if matches:
                        key = matches[0]
                candidates = groups.setdefault(key, [])
                if not candidates or candidates[0][0].api == api:
                    candidates.append([endpoint])
                elif len(candidates) == 1 and all(item.api != api for item in candidates[0]):
                    candidates[0].append(endpoint)

        visible = {endpoint.index for candidates in groups.values() for variants in candidates for endpoint in variants}
        system_id = next(
            (host_apis[api][default_key] for api in ordered_apis
             if backend_rank(host_apis[api]["name"]) == 0 and host_apis[api][default_key] in visible),
            default_id,
        )
        if system_id not in visible:
            system_id = next((host_apis[api][default_key] for api in ordered_apis
                              if host_apis[api][default_key] in visible), None)
        devices = [
            Device(f"{variants[0].name} ({number})" if len(candidates) > 1 else variants[0].name,
                   tuple(variants), any(item.index == system_id for item in variants))
            for candidates in groups.values()
            for number, variants in enumerate(candidates, 1)
        ]
        grouped.append(tuple(sorted(devices, key=lambda device: (not device.system_default, name_key(device.label)))))
    return tuple(grouped)


def devices(*, refresh=False):
    if refresh:
        # PortAudio enumerates endpoints during initialization. The UI refreshes only with streams stopped.
        sd._terminate()
        sd._initialize()
    return group_devices(sd.query_devices(), sd.query_hostapis(), sd.default.device)


def default_devices(inputs, outputs):
    microphone = next((device for device in inputs if device.system_default), next(iter(inputs), None))
    system_output = next((device for device in outputs if device.system_default), next(iter(outputs), None))
    output = max(outputs, key=lambda device: (device.cable_rank, device.system_default), default=None)
    return microphone, output, system_output
