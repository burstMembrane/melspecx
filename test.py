from dataclasses import dataclass
from pathlib import Path
from time import time

import melspecx
from IPython.display import Image
from melspecx import create_mel_config

AUDIO_PATH = Path("./data/suzannetrimmed.wav")
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


config = create_mel_config(
    sample_rate=sr,
    n_fft=2048,
    win_length=512,
    hop_length=1024,
    f_min=0,
    f_max=sr / 2,
    n_mels=128,
    top_db=80,
    onesided=True,
)
# convert to named tuple

start = time()
config = MelConfig(**config)
mel_spec = melspecx.mel_spectrogram_db_py(config, audio)
print("Generated mel spectrogram in {:0.2f} seconds".format(time() - start))
image = melspecx.plot_mel_spec_py(mel_spec, "inferno", 1024, 256)
with open("test.png", "wb") as f:
    f.write(image)


# test plot from path

start = time()
out = melspecx.mel_spec_from_path(
    str(AUDIO_PATH), 2048, 512, 1024, 0, sr / 2, 128, "inferno", 1024, 256, 80
)
print("Generated mel spectrogram in {:0.2f} seconds".format(time() - start))
