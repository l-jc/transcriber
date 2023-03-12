"""Utilities"""
import os
from enum import Enum, auto
import functools
import time
import pyaudiowpatch as pyaudio


class DeviceType(Enum):
    """Audio device type"""

    MICROPHONE = auto()
    SPEAKER = auto()

    def __str__(self) -> str:
        return self.name.lower()

    @staticmethod
    def argparse(arg: str) -> "DeviceType":
        """hack for argparse command line arguments"""
        arg = arg.upper()
        try:
            return DeviceType[arg]
        except KeyError:
            return arg


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
    key = "defaultOutputDevice" if _type == DeviceType.SPEAKER else "defaultInputDevice"
    audio_device = p_audio.get_device_info_by_index(wasapi_info[key])
    if _type == DeviceType.SPEAKER:
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
    """Timer class"""

    def __init__(self, name) -> None:
        self.name = name
        self.duration = 0
        self.n_calls = 0
        self._start = 0

    def start(self):
        """Start timer"""
        self._start = time.perf_counter()

    def stop(self):
        """Stop timer"""
        assert self._start > 0
        self.duration += time.perf_counter() - self._start
        self._start = 0
        self.n_calls += 1

    def reset(self):
        """Reset timer"""
        self.duration = 0
        self.n_calls = 0
        self._start = 0

    def report(self):
        """Print timer metrics"""
        avg = self.duration / (self.n_calls + 1e-6)
        print(f"Avg. Time for {self.name}: {avg*1000:.2f} ms.")


@functools.lru_cache(maxsize=10)
def format_t(seconds: float) -> str:
    """foramt seconds to minutes:seconds"""
    seconds = round(seconds)
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"


def printline(output: str, end: str = None) -> None:
    """Print a entire line in terminal"""
    width = os.get_terminal_size().columns
    padding = max(0, width - len(output))
    output += " " * padding
    if len(output) > width:
        output = output[: width - 3] + "..."
    print(output, end=end)
