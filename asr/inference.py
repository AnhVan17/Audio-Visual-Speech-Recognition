import random
from pathlib import Path
from typing import Optional

import librosa
import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1

asr = pipeline(
    task="automatic-speech-recognition",
    model="vinai/PhoWhisper-small",         # tiny/base/small/medium/large
    chunk_length_s=30,                      
    stride_length_s=(4, 4),
    generate_kwargs={"task": "transcribe", "language": "vi"}, 
    device=device,
)


def extract_random_segment(audio_path: Path, segment_length_s: float = 30.0, target_sr: int = 16000):
    """Load audio and return a random segment."""
    if segment_length_s <= 0:
        raise ValueError("Segment length must be positive.")

    y, sr = librosa.load(audio_path, sr=target_sr)
    if y.size == 0:
        raise ValueError(f"Audio file {audio_path} is empty or unreadable.")

    segment_samples = int(segment_length_s * sr)
    if segment_samples <= 0:
        raise ValueError("Segment length is too short for the given sampling rate.")

    if y.shape[0] <= segment_samples:
        start_sample = 0
        segment = y
    else:
        max_start = y.shape[0] - segment_samples
        start_sample = random.randint(0, max_start)
        segment = y[start_sample : start_sample + segment_samples]

    end_sample = start_sample + segment.shape[0]
    return segment, sr, start_sample, end_sample


def transcribe_random_segment(audio_path: Path, segment_length_s: float = 30.0, seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)

    segment, sr, start_sample, end_sample = extract_random_segment(audio_path, segment_length_s)
    result = asr({"array": segment, "sampling_rate": sr})
    text = result.get("text", "")

    start_time = start_sample / sr
    end_time = end_sample / sr
    return text, start_time, end_time


def main(audio_path: Path, segment_length_s: float = 30.0, seed: Optional[int] = None):
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file {audio_path} does not exist.")

    text, start_time, end_time = transcribe_random_segment(audio_path, segment_length_s, seed)

    print(f"Random segment: {start_time:.2f}s → {end_time:.2f}s")
    print(text)
    return text, start_time, end_time


if __name__ == "__main__":
    # Update these defaults before running directly
    AUDIO_PATH = Path("/Volumes/Kingston_XS1000_Media/Sách nói/hanh-trinh-ve-phuong-dong.mp3")
    SEGMENT_LENGTH = 30.0
    SEED = None

    main(AUDIO_PATH, SEGMENT_LENGTH, SEED)
