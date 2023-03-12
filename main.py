"""Main"""
import argparse
from multiprocessing import Queue, Process, Value
import os
import time
import wave
import pyaudiowpatch as pyaudio
from whisper.tokenizer import LANGUAGES

from wproc import run as w_run
from utils import get_default_wasapi_device, DeviceType
from constants import DATA_FORMAT, RECORDER_BUFFER_SIZE, RECOGNIZER_STEP


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Transcriber")
    parser.add_argument(
        "--input",
        required=True,
        type=DeviceType.argparse,
        choices=tuple(DeviceType),
        help="The input device to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tiny",
        choices=("tiny", "base", "small", "medium", "large"),
        help="Whisper model type",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        choices=tuple(LANGUAGES.keys()) + ("",),
        help="Language to transcribe. Defaults to multilingual",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task to perform, transcribe or translate.",
    )
    args = parser.parse_args()

    print(f"Recorder running P{os.getpid()}")

    with pyaudio.PyAudio() as p, wave.open("test.wav", "wb") as wave_file:

        # region get audio device
        audio_device = get_default_wasapi_device(p, args.input)
        channels = audio_device.channels
        rate = audio_device.rate
        # endregion

        # region setup recording file
        wave_file: wave.Wave_write
        wave_file.setnchannels(channels)
        wave_file.setsampwidth(pyaudio.get_sample_size(DATA_FORMAT))
        wave_file.setframerate(rate)
        # endregion

        # region start whisper process
        output_queue = Queue()
        ready = Value("h", 0)
        proc = Process(
            target=w_run,
            name="Whisper",
            args=(output_queue, ready, audio_device, args.model, args.language or None, args.task),
            daemon=True,
        )
        proc.start()
        while ready.value == 0:
            time.sleep(1)
        # endregion

        # region start recording stream
        def callback(in_data, frame_count: int, time_info: dict, status):
            """Callback for audio streaming"""
            output_queue.put(in_data)
            wave_file.writeframes(in_data)
            if output_queue.qsize() > 4 * RECOGNIZER_STEP / RECORDER_BUFFER_SIZE:
                print("RECORDING BUFFER OVERFLOW!!!")
                raise KeyboardInterrupt
            return (in_data, pyaudio.paContinue)

        stream = p.open(
            format=DATA_FORMAT,
            channels=channels,
            rate=rate,
            frames_per_buffer=int(rate * RECORDER_BUFFER_SIZE),
            input=True,
            input_device_index=audio_device.index,
            stream_callback=callback,
        )
        print("Start recording...\n\n")
        # endregion

        try:
            proc.join()
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()
