"""Whispering process"""
from multiprocessing import Queue, Value
from typing import Union
import os
import numpy as np
import torch
import torchaudio
import whisper

from constants import MODEL_INPUT_SIZE, MODEL_CARD, STD_RATE, BUFFER_SIZE
from utils import MyTimer


def run(output_queue: Queue, rate: int, ready: Value):
    """Subprocess to run rtt"""
    model = whisper.load_model(MODEL_CARD, device=torch.device(0))
    print(f"Loaded model {MODEL_CARD}\n\n")
    resampler = torchaudio.transforms.Resample(rate, STD_RATE, dtype=torch.float32).to(model.device)
    print("Created resmpler\n\n")
    with ready.get_lock():
        ready.value = 1

    dsizes = []
    total = 0
    window = []
    start_chunk = 0

    print("Start transcription\n\n")
    preproc_timer = MyTimer("preprocessing")
    decoding_timer = MyTimer("decoding")
    resample_timer = MyTimer("resample")

    try:
        while True:
            data = b""
            dsize = 0
            while not output_queue.empty():
                dsize += 1
                data += output_queue.get()
            if len(data) == 0:
                continue
            dsizes.append(dsize)
            total += dsize

            # Construct waveform from buffer
            preproc_timer.start()
            waveform = np.frombuffer(data, np.int16).flatten().astype(np.float32)
            waveform = waveform[::2]  # 2 channels
            waveform = torch.from_numpy(waveform).to(model.device) / 32768.0
            preproc_timer.stop()
            resample_timer.start()
            waveform = resampler(waveform)
            resample_timer.stop()

            # Resizing window
            window.append(waveform)
            while total > MODEL_INPUT_SIZE / BUFFER_SIZE:  # seconds
                window.pop(0)
                x = dsizes.pop(0)
                total -= x
                start_chunk += x

            # Run decoding
            if len(window) == 0:
                continue
            decoding_timer.start()
            audio = torch.cat(window)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            options = whisper.DecodingOptions(language="en", fp16=True)
            result = whisper.decode(model, mel, options)
            decoding_timer.stop()

            # Print result
            tlength = len(result.text)
            cmd_width = os.get_terminal_size().columns
            print(" " * cmd_width, end="\r")
            padding = cmd_width - tlength
            if padding < 0:
                text = result.text[-cmd_width:]
            else:
                text = " " * padding + result.text
            print(text, end="\r")

    except KeyboardInterrupt:
        print("\n\n")
        preproc_timer.report()
        decoding_timer.report()
        resample_timer.report()
