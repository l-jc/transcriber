"""Constants used in project"""
import pyaudiowpatch as pyaudio
from whisper.audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH, CHUNK_LENGTH

DATA_FORMAT = pyaudio.paInt16
RECORDER_BUFFER_SIZE = 0.2  # seconds

RECOGNIZER_STEP = 2  # seconds
NO_SPEECH_THRESHOLD = 0.6
LOGPROB_THRESHOLD = -1.0
COMPRESSION_RATIO_THRESHOLD = 2.4
DETECT_TAIL = 3 * RECOGNIZER_STEP  # 6 seconds
TEMPERATURES = (0.0,)
# TEMPERATURES = (0.0,0.2)
# TEMPERATURES = (0.0,0.2,0.4)
BEAM_SIZE = None
