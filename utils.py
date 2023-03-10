"""Utilities"""
from enum import Enum, auto
import time
import pyaudiowpatch as pyaudio


class DeviceType(Enum):
    """Audio device type"""

    INPUT = auto()
    OUTPUT = auto()


class DeviceWrapper:
    """Wrapper around device dict"""

    def __init__(self, device: dict) -> None:
        self.raw = device
        self.rate = int(device["defaultSampleRate"])
        self.channels = device["maxInputChannels"]
        self.index = device["index"]

    def __repr__(self) -> str:
        return f"{self.raw['name']} ({self.rate} Hz, {self.channels} chnl)"


def get_default_wasapi_device(p_audio: pyaudio.PyAudio, _type: DeviceType) -> DeviceWrapper:
    """Get default "input" device"""
    audio_device = None
    wasapi_info = p_audio.get_host_api_info_by_type(pyaudio.paWASAPI)
    key = "defaultOutputDevice" if _type == DeviceType.OUTPUT else "defaultInputDevice"
    audio_device = p_audio.get_device_info_by_index(wasapi_info[key])
    if _type == DeviceType.OUTPUT:
        # Get loopback if default device is a speaker
        for loopback in p_audio.get_loopback_device_info_generator():
            if audio_device["name"] in loopback["name"]:
                audio_device = loopback
                break
    assert audio_device is not None
    audio_device = DeviceWrapper(audio_device)
    print(f"Got audio device {audio_device}")
    return audio_device


class MyTimer:
    def __init__(self, name) -> None:
        self.name = name
        self.duration = 0
        self.n_calls = 0
        self._start = 0

    def start(self):
        self._start = time.perf_counter()

    def stop(self):
        assert self._start > 0
        self.duration += time.perf_counter() - self._start
        self._start = 0
        self.n_calls += 1

    def reset(self):
        self.duration = 0
        self.n_calls = 0
        self._start = 0

    def report(self):
        avg = self.duration / (self.n_calls + 1e-6)
        print(f"Avg. Time for {self.name}: {avg*1000:.2f} ms.")
