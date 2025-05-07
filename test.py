from dataclasses import dataclass
from pathlib import Path
from time import time

import melspecx
from IPython.display import Image
from melspecx import create_mel_config

AUDIO_PATH = Path("./data/suzanne.wav")
audio, sr = melspecx.read_wav(str(AUDIO_PATH), normalize=True)


@dataclass
class MelConfig:
    sample_rate: float
    n_fft: int
    win_length: int
    hop_length: int
    f_min: float
    f_max: float
    n_mels: int
    top_db: float
    onesided: bool
    chunk_size: int


config = create_mel_config(
    sample_rate=sr,
    n_fft=2048,
    win_length=512,
    hop_length=1024,
    f_min=0,
    f_max=8000,
    n_mels=128,
    top_db=60,
    onesided=True,
    chunk_size=4096,
)

audio, sr = melspecx.read_wav(str(AUDIO_PATH), normalize=True)
# get the number of samples in the audio
num_samples = len(audio)
print(num_samples)
start = time()
config = MelConfig(**config, chunk_size=4096)


mel_spec = melspecx.mel_spectrogram_db_py(config, audio)
# get the number of mel bins
num_mel_bins = len(mel_spec)
print(num_mel_bins)
print("Generated mel spectrogram in {:0.2f} seconds".format(time() - start))
image = melspecx.plot_mel_spec_py(
    mel_spec, cmap="inferno", width_px=num_mel_bins, height_px=512
)
with open("test.png", "wb") as f:
    f.write(image)
