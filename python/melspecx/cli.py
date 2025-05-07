from pathlib import Path

import typer
from halo import Halo
from rich import print

from melspecx.melspecx import mel_spec_from_path

app = typer.Typer(help="Generate a mel spectrogram from an audio file.")


@app.command()
def generate(
    input_file: Path = typer.Argument(
        ..., help="Path to the input audio file (supports WAV format)"
    ),
    output_file: Path = typer.Argument(
        ..., help="Path where the output spectrogram image will be saved"
    ),
    n_fft: int = typer.Option(
        2048,
        help="Number of FFT components. Higher values give better frequency resolution but lower time resolution",
    ),
    win_length: int = typer.Option(
        512,
        help="Length of the window function. Should be <= n_fft. Smaller values give better time resolution",
    ),
    hop_length: int = typer.Option(
        1024,
        help="Number of samples between successive frames. Controls overlap between windows",
    ),
    f_min: float = typer.Option(
        0, help="Minimum frequency (Hz) to include in the mel spectrogram"
    ),
    f_max: float = typer.Option(
        8000, help="Maximum frequency (Hz) to include in the mel spectrogram"
    ),
    n_mels: int = typer.Option(
        128,
        help="Number of mel bands to generate. Higher values give more detailed frequency representation",
    ),
    colormap: str = typer.Option(
        "greys",
        help="Matplotlib colormap to use for visualization (e.g., 'inferno', 'viridis', 'magma', 'plasma')",
    ),
    width_px: int = typer.Option(1024, help="Width of the output image in pixels"),
    height_px: int = typer.Option(256, help="Height of the output image in pixels"),
    top_db: float = typer.Option(
        60,
        help="Maximum decibel value for normalization. Values above this will be clipped",
    ),
    onesided: bool = typer.Option(
        True,
        help="If True, only return positive frequencies. If False, return both positive and negative frequencies",
    ),
    chunk_size: int = typer.Option(
        1024,
        help="Size of chunks to process at once. Larger values use more memory but may be faster",
    ),
):
    """
    Generate a mel spectrogram from an audio file.

    This command creates a visual representation of the frequency content of an audio file
    over time, using the mel scale which approximates human perception of sound.
    """
    print(f"[green]Generating mel spectrogram for:[/green] {input_file}")
    spinner = Halo(text="Generating mel spectrogram...", color="cyan")
    spinner.start()
    image = mel_spec_from_path(
        str(input_file),
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        colormap=colormap,
        width_px=width_px,
        height_px=height_px,
        top_db=top_db,
        chunk_size=chunk_size,
    )
    output_file.write_bytes(image)
    spinner.stop()
    print(f"[bold green]Saved to:[/bold green] {output_file}")


if __name__ == "__main__":
    app()
