"""Constants used in project"""
import pyaudiowpatch as pyaudio
from whisper.audio import SAMPLE_RATE, N_FRAMES, HOP_LENGTH

DATA_FORMAT = pyaudio.paInt16
RECORDER_BUFFER_SIZE = 0.2  # seconds

RECOGNIZER_STEP = 2  # seconds
MODEL_CARD = "tiny.en"
