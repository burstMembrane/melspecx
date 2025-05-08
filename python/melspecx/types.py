from dataclasses import dataclass
from typing import Optional


@dataclass
class MelConfig:
    """
    Configuration for mel spectrogram generation.
    """

    sample_rate: Optional[float] = 22050
    n_fft: Optional[int] = 2048
    win_length: Optional[int] = 512
    hop_length: Optional[int] = 1024
    f_min: Optional[float] = 0
    f_max: Optional[float] = 11025
    n_mels: Optional[int] = 128
    top_db: Optional[float] = 60
    onesided: Optional[bool] = True
    chunk_size: Optional[int] = 4096
