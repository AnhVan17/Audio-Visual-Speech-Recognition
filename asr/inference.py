import torch
from transformers import pipeline
import librosa

device = 0 if torch.cuda.is_available() else -1

asr = pipeline(
    task="automatic-speech-recognition",
    model="vinai/PhoWhisper-small",         # tiny/base/small/medium/large
    chunk_length_s=30,                      
    stride_length_s=(4, 4),
    generate_kwargs={"task": "transcribe", "language": "vi"}, 
    device=device,
)


# y, sr = librosa.load("audio.mp3", sr=16000)
# print(asr({"array": y, "sampling_rate": sr})["text"])


# result = asr({"array": y, "sampling_rate": sr}, return_timestamps="word", batch_size=1)
# for w in result["chunks"]:   
#     print(w)
