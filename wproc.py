"""Whispering process"""
from multiprocessing import Queue, Value
import os
import numpy as np
import torch
import torchaudio
import whisper
from whisper.tokenizer import get_tokenizer

from constants import MODEL_CARD, SAMPLE_RATE, RECORDER_BUFFER_SIZE
from utils import MyTimer


def run(output_queue: Queue, rate: int, channels: int, ready: Value):
    """Subprocess to run rtt"""
    print(f"Whisper running P{os.getpid()}")
    model = whisper.load_model(MODEL_CARD, device=torch.device(0))
    print(f"Loaded model {MODEL_CARD}\n\n")
    tokenizer = get_tokenizer(model.is_multilingual, language="en", task="transcribe")
    TIME_BEGIN = tokenizer.timestamp_begin
    print("Got tokenizer\n\n")
    resampler = torchaudio.transforms.Resample(rate, SAMPLE_RATE, dtype=torch.float32).to(
        model.device
    )
    print("Created resmpler\n\n")
    with ready.get_lock():
        ready.value = 1

    print("Start transcription\n\n")
    preproc_timer = MyTimer("preprocessing")
    decoding_timer = MyTimer("decoding")
    resample_timer = MyTimer("resample")
    data = b""
    dsize = 0
    segment = None
    prompt = None
    seg_start_t = 0
    try:
        while True:
            # get data from buffer
            # is there's more than 2 seconds of new data, process,
            # otherwise continue
            while not output_queue.empty():
                data += output_queue.get()
                dsize += 1

            if dsize < 2 / RECORDER_BUFFER_SIZE:
                continue

            waveform = np.frombuffer(data, np.int16).flatten().astype(np.float32)
            waveform = waveform[::channels]
            waveform = torch.from_numpy(waveform).to(model.device) / 32768.0
            waveform = resampler(waveform)

            data = b""
            dsize = 0

            segment = torch.cat([segment, waveform]) if segment is not None else waveform
            segment_for_model = whisper.pad_or_trim(segment)
            mel = whisper.log_mel_spectrogram(segment_for_model).to(model.device)
            options = whisper.DecodingOptions(language="en", prompt=prompt)
            result = whisper.decode(model, mel, options)

            tokens = result.tokens

            # find consecutive time
            consecutive_t = None
            for t in range(1, len(tokens)):
                if tokens[t] >= tokens[t - 1] >= TIME_BEGIN:
                    consecutive_t = t - 1
                    break

            if consecutive_t:
                elapsed = (tokens[consecutive_t] - TIME_BEGIN) * 0.02
                seg_start_t += elapsed
                # discard segment before seg_start_t
                segment = segment[int(SAMPLE_RATE * elapsed) :]
                output = tokenizer.decode(tokens[: consecutive_t + 1])
                prompt = output
                width = os.get_terminal_size().columns
                padding = max(0, width - len(output))
                print(output + " " * padding)
            else:
                output = result.text
                width = os.get_terminal_size().columns
                padding = max(0, width - len(output))
                print(" " * padding + output, end="\r")

    except KeyboardInterrupt:
        print("\n\n")
        preproc_timer.report()
        decoding_timer.report()
        resample_timer.report()
