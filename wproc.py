"""Whispering process"""
from multiprocessing import Queue, Value
import os
import numpy as np
import torch
import torchaudio
import whisper
from whisper.utils import exact_div
from whisper.tokenizer import get_tokenizer

from constants import (
    MODEL_CARD,
    SAMPLE_RATE,
    RECORDER_BUFFER_SIZE,
    RECOGNIZER_STEP,
    N_FRAMES,
    HOP_LENGTH,
    LANGUAGE,
    NO_SPEECH_THRESHOLD,
    CHUNK_LENGTH,
)
from utils import MyTimer


def run(output_queue: Queue, rate: int, channels: int, ready: Value):
    """Subprocess to run rtt"""
    print(f"Whisper running P{os.getpid()}")
    model = whisper.load_model(MODEL_CARD, device=torch.device(0))

    input_stride = exact_div(N_FRAMES, model.dims.n_audio_ctx)  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)

    print(f"Loaded model {MODEL_CARD}")
    tokenizer = get_tokenizer(model.is_multilingual, language=LANGUAGE, task="transcribe")
    TIME_BEGIN = tokenizer.timestamp_begin
    print("Got tokenizer")
    resampler = torchaudio.transforms.Resample(rate, SAMPLE_RATE, dtype=torch.float32).to(
        model.device
    )
    print("Created resmpler")
    with ready.get_lock():
        ready.value = 1

    print("Start transcription")
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
            preproc_timer.start()
            while not output_queue.empty():
                data += output_queue.get()
                dsize += 1

            if dsize < RECOGNIZER_STEP / RECORDER_BUFFER_SIZE:
                preproc_timer.stop()
                continue

            waveform = np.frombuffer(data, np.int16).flatten().astype(np.float32)
            waveform = waveform[::channels]
            waveform = torch.from_numpy(waveform).to(model.device) / 32768.0
            preproc_timer.stop()
            resample_timer.start()
            waveform = resampler(waveform)
            resample_timer.stop()

            data = b""
            dsize = 0

            decoding_timer.start()
            segment = torch.cat([segment, waveform]) if segment is not None else waveform
            segment_for_model = whisper.pad_or_trim(segment)
            mel = whisper.log_mel_spectrogram(segment_for_model).to(model.device)
            # options = whisper.DecodingOptions(language=LANGUAGE, prompt=prompt)
            options = whisper.DecodingOptions(language=LANGUAGE)
            result = whisper.decode(model, mel, options)

            if result.no_speech_prob > NO_SPEECH_THRESHOLD:
                seg_start_t += segment.size(0) / SAMPLE_RATE
                segment = None
                print("NO SPEECH", end="\r")
                continue

            tokens = result.tokens

            # find consecutive time
            consecutive_t = None
            # for t in range(1, len(tokens)):  # first segs
            for t in reversed(range(1, len(tokens))):  # all segs
                if tokens[t] >= tokens[t - 1] >= TIME_BEGIN:
                    consecutive_t = t - 1
                    break
            decoding_timer.stop()

            commit_seconds = 0
            if segment.size(0) >= SAMPLE_RATE * (CHUNK_LENGTH - RECOGNIZER_STEP):
                # if segment is longer than ? seconds,
                commit_seconds = segment.size(0) / SAMPLE_RATE
            elif consecutive_t:
                commit_seconds = (tokens[consecutive_t] - TIME_BEGIN) * time_precision

            if commit_seconds > 0:
                # discard segment before seg_start_t
                segment = segment[int(SAMPLE_RATE * commit_seconds) :]
                if consecutive_t is not None:
                    output = tokenizer.decode(tokens[: consecutive_t + 1])
                else:
                    output = result.text
                prompt = output
                prefix = f"[{seg_start_t:06.2f} -- {(seg_start_t+commit_seconds):06.2f}] "
                output = prefix + output
                seg_start_t += commit_seconds
                width = os.get_terminal_size().columns
                padding = max(0, width - len(output) - 4)
                print("    " + output + " " * padding)
            else:
                segment_seconds = segment.size(0) / SAMPLE_RATE
                output = f"{segment_seconds:.2f} seconds - " + result.text
                width = os.get_terminal_size().columns
                padding = max(0, width - len(output))
                # print(" " * padding + output, end="\r")
                print(output, end="\r")

            if seg_start_t > 600:
                print("Stop after 10 minutes.")
                break

    except KeyboardInterrupt:
        print("\n\n")
        preproc_timer.report()
        resample_timer.report()
        decoding_timer.report()
