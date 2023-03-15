# pylint: disable=no-member
# pylint: disable=not-callable
"""
Whisper process

1. get audio buffer (at least x seconds) from queue
2. append to current segment
3. decode current segment
4. commit or don't commit
    a. no speech -> commit all
    b. complete segment(s) -> commit first/all segments
    c. too long -> commit all
    d. silent for x seconds -> commit all
"""
from multiprocessing import Queue, Value
from typing import Optional, Tuple
import os
import numpy as np
import torch
import torchaudio
import whisper
from whisper.utils import exact_div
from whisper.tokenizer import get_tokenizer, Tokenizer
from whisper.model import Whisper

from constants import (
    SAMPLE_RATE,
    RECORDER_BUFFER_SIZE,
    RECOGNIZER_STEP,
    N_FRAMES,
    HOP_LENGTH,
    NO_SPEECH_THRESHOLD,
    LOGPROB_THRESHOLD,
    DETECT_TAIL,
    CHUNK_LENGTH,
    COMPRESSION_RATIO_THRESHOLD,
    TEMPERATURES,
    BEAM_SIZE,
)
from utils import MyTimer, DeviceWrapper, format_t, printline


def get_audio_from_queue(output_queue: Queue, channels: int, device: torch.device) -> torch.Tensor:
    """get audio from buffer"""
    data = b""
    n_buffer = 0
    while n_buffer < RECOGNIZER_STEP / RECORDER_BUFFER_SIZE:
        while not output_queue.empty():
            data += output_queue.get()
            n_buffer += 1
    waveform = np.frombuffer(data, np.int16).flatten().astype(np.float32)
    waveform = waveform[::channels]
    waveform = torch.from_numpy(waveform).to(device) / 32768.0
    return waveform


def commit(
    result: whisper.DecodingResult,
    segment: torch.Tensor,
    tokenizer: Tokenizer,
    inverted_time_precision: float,
    vad_model: Whisper,
) -> Tuple[int, str]:
    """Return the number of samples to commit and the committed text"""
    if result.no_speech_prob > NO_SPEECH_THRESHOLD and result.avg_logprob < LOGPROB_THRESHOLD:
        return segment.size(0), ""

    tokens = torch.tensor(result.tokens)
    timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
    consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0].add_(1)
    if len(consecutive) > 1:  # if the output contains two consecutive timestamp tokens
        # option a. commit first segment
        # t = tokens[consecutive[0]] - tokenizer.timestamp_begin
        # output = tokenizer.decode(tokens[: consecutive[0]])
        # option b. commit all segment
        t = tokens[consecutive[-1]] - tokenizer.timestamp_begin
        output = tokenizer.decode(tokens[: consecutive[-1]])
        remain = tokenizer.decode(tokens[consecutive[-1] :])
        return exact_div(t.item() * SAMPLE_RATE, inverted_time_precision), (output, remain)

    if segment.size(0) > (CHUNK_LENGTH - RECOGNIZER_STEP) * SAMPLE_RATE:
        return segment.size(0), result.text

    # use tail to detect
    tail_is_silent = False
    if segment.size(0) > DETECT_TAIL * SAMPLE_RATE:
        tail = segment[-min(3, DETECT_TAIL // 2) * SAMPLE_RATE :]
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(tail))
        tail_result = whisper.decode(
            vad_model, mel, whisper.DecodingOptions(language=result.language)
        )
        if (
            tail_result.no_speech_prob > NO_SPEECH_THRESHOLD
            and tail_result.avg_logprob < LOGPROB_THRESHOLD
        ):
            tail_is_silent = True

    if tail_is_silent:
        return segment.size(0), result.text

    return 0, result.text


def run(
    output_queue: Queue,
    ready: Value,
    audio_device: DeviceWrapper,
    model_card: str,
    language: Optional[str],
    task: str = "transcribe",
):
    """Subprocess to run rtt"""
    print(f"Whisper running P{os.getpid()}")
    if language == "en":
        model_card += ".en"
    model = whisper.load_model(model_card)
    vad_model = whisper.load_model("tiny.en" if language == "en" else "tiny")

    input_stride = exact_div(N_FRAMES, model.dims.n_audio_ctx)  # mel frames per output token: 2
    inverted_time_precision = exact_div(SAMPLE_RATE, input_stride * HOP_LENGTH)
    print(f"Loaded model {model_card}")

    tokenizer = get_tokenizer(model.is_multilingual, language=language, task=task)
    print("Got tokenizer")

    resampler = torchaudio.transforms.Resample(
        audio_device.rate, SAMPLE_RATE, dtype=torch.float32
    ).to(model.device)
    resampler(torch.zeros(audio_device.rate, dtype=torch.float32, device=model.device))
    print("Created resmpler")

    # Tell recorder ready to record
    with ready.get_lock():
        ready.value = 1

    print("Start transcription")
    preproc_timer = MyTimer("preprocessing")
    resample_timer = MyTimer("resample")
    decoding_timer = MyTimer("decoding")
    output_timer = MyTimer("output")
    segment = None
    # prompt = None
    seg_start_t = 0
    should_detect_language = language is None
    try:
        while True:
            preproc_timer.start()
            # region get data from audio buffer
            waveform = get_audio_from_queue(output_queue, audio_device.channels, model.device)
            # endregion
            preproc_timer.stop()

            resample_timer.start()
            # region resample to whisper audio sample rate
            waveform = resampler(waveform)
            # endregion
            resample_timer.stop()

            decoding_timer.start()
            # region decode segment
            segment = torch.cat([segment, waveform]) if segment is not None else waveform
            segment_for_model = whisper.pad_or_trim(segment)
            mel = whisper.log_mel_spectrogram(segment_for_model).to(model.device)
            for temperature in TEMPERATURES:
                options = whisper.DecodingOptions(
                    task=task,
                    language=language,
                    temperature=temperature,
                    beam_size=BEAM_SIZE if temperature != 0.0 else None,
                    # prompt=prompt,
                )
                result = whisper.decode(model, mel, options)
                if (
                    result.no_speech_prob > NO_SPEECH_THRESHOLD
                    and result.avg_logprob < LOGPROB_THRESHOLD
                ):
                    break
                if (
                    result.compression_ratio < COMPRESSION_RATIO_THRESHOLD
                    and result.avg_logprob > LOGPROB_THRESHOLD
                ):
                    break
            # endregion
            decoding_timer.stop()

            output_timer.start()
            # region commit
            samples_to_commit, output = commit(
                result, segment, tokenizer, inverted_time_precision, vad_model
            )
            if language is None:
                language = result.language
            # endregion

            # region print output and update segment
            if samples_to_commit > 0:
                seconds_to_commit = samples_to_commit / SAMPLE_RATE
                if output == "":
                    printline(
                        f"{segment.size(0)/SAMPLE_RATE:.2f}s NO SPEECH {result.no_speech_prob:.2f}",
                        end="\r",
                    )
                else:
                    remain = ""
                    if isinstance(output, tuple):
                        output, remain = output

                    printline(
                        f"  [{format_t(seg_start_t)} -- {format_t(seg_start_t+seconds_to_commit)}] "
                        f"({language}) {output}"
                    )
                    print(f"{segment.size(0)/SAMPLE_RATE:05.2f}s ({language}) {remain}", end="\r")
                seg_start_t += seconds_to_commit
                segment = segment[samples_to_commit:]
            else:
                print(f"{segment.size(0)/SAMPLE_RATE:05.2f}s ({language}) {output}", end="\r")
            if samples_to_commit > 0 and should_detect_language:
                language = None
            # endregion
            output_timer.stop()

    except KeyboardInterrupt:
        print("\n\n")
        preproc_timer.report()
        resample_timer.report()
        decoding_timer.report()
        output_timer.report()
