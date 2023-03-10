"""Main"""
from multiprocessing import Queue, Process, Value
import os
import time
import wave
import pyaudiowpatch as pyaudio

# from wproc import run as w_run
from wproc import run as w_run
from utils import get_default_wasapi_device
from constants import DATA_FORMAT, RECORDER_BUFFER_SIZE


if __name__ == "__main__":
    print(f"Recorder running P{os.getpid()}")

    with pyaudio.PyAudio() as p, wave.open("test.wav", "wb") as wave_file:

        # region get audio device
        audio_device = get_default_wasapi_device(p)
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
        proc = Process(target=w_run, args=(output_queue, rate, channels, ready), daemon=True)
        proc.start()
        while ready.value == 0:
            time.sleep(1)
        # endregion

        # region start recording stream
        def callback(in_data, frame_count, time_info, status):
            """Callback for audio streaming"""
            output_queue.put(in_data)
            wave_file.writeframes(in_data)
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
