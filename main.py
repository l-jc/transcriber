"""Main"""
import argparse
from multiprocessing import Queue, Process, Value
import os
import time
import contextlib
import wave
import pyaudiowpatch as pyaudio
from whisper.tokenizer import LANGUAGES
from whisper.utils import str2bool

from wproc import run as w_run
from utils import get_default_wasapi_device, DeviceType, printline, format_t, srt_format_time
from constants import DATA_FORMAT, RECORDER_BUFFER_SIZE, RECOGNIZER_STEP


if __name__ == "__main__":
    # region argparse
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
    parser.add_argument(
        "--save_audio",
        type=str2bool,
        default=True,
        help="whether to save audio to file; True by default",
    )
    args = parser.parse_args()
    # endregion

    print(f"Recorder running P{os.getpid()}")

    wave_file_ctx = wave.open("test.wav", "wb") if args.save_audio else contextlib.nullcontext()

    with pyaudio.PyAudio() as p, wave_file_ctx as wave_file, open(
        "test.srt", "w", encoding="utf-8"
    ) as srt_file:

        # region get audio device
        audio_device = get_default_wasapi_device(p, args.input)
        channels = audio_device.channels
        rate = audio_device.rate
        # endregion

        # region setup recording file
        if wave_file is not None:
            wave_file: wave.Wave_write
            wave_file.setnchannels(channels)
            wave_file.setsampwidth(pyaudio.get_sample_size(DATA_FORMAT))
            wave_file.setframerate(rate)
        # endregion

        # region start whisper process
        audio_queue = Queue()
        output_queue = Queue()
        ready = Value("h", 0)
        proc = Process(
            target=w_run,
            name="Whisper",
            args=(
                audio_queue,
                output_queue,
                ready,
                audio_device,
                args.model,
                args.language or None,
                args.task,
            ),
            daemon=True,
        )
        proc.start()
        while ready.value == 0:
            time.sleep(1)
        # endregion

        # region start recording stream
        def callback(in_data, frame_count: int, time_info: dict, status):
            """Callback for audio streaming"""
            audio_queue.put(in_data)
            if wave_file is not None:
                wave_file.writeframes(in_data)
            if audio_queue.qsize() > 4 * RECOGNIZER_STEP / RECORDER_BUFFER_SIZE:
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
            i = 0
            while True:
                time_info, language, output = output_queue.get()
                start, t1, t2 = time_info
                text1, text2 = output
                if text1:
                    printline(f"  [{format_t(start)} -- {format_t(start+t1)}] ({language}) {text1}")
                    i += 1
                    lines = [
                        str(i) + "\n",
                        f"{srt_format_time(start)} --> {srt_format_time(start+t1)}\n",
                        text1 + "\n\n",
                    ]
                    srt_file.writelines(lines)
                text2 = text2 or "NO SPEECH"
                print(f"{t2:05.2f}s ({language}) {text2}", end="\r")
        except KeyboardInterrupt:
            pass
        finally:
            stream.stop_stream()
            stream.close()
