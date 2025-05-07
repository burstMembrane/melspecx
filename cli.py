import argparse

import melspecx


def main():
    parser = argparse.ArgumentParser(
        description="Generate a mel spectrogram from an audio file"
    )
    parser.add_argument("input_file", type=str, help="The input audio file")
    parser.add_argument("output_file", type=str, help="The output image file")
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--win_length", type=int, default=512)
    parser.add_argument("--hop_length", type=int, default=1024)
    parser.add_argument("--f_min", type=float, default=0)
    parser.add_argument("--f_max", type=float, default=8000)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--colormap", type=str, default="greys")
    parser.add_argument("--width_px", type=int, default=1024)
    parser.add_argument("--height_px", type=int, default=256)
    parser.add_argument("--top_db", type=float, default=60)
    parser.add_argument("--onesided", type=bool, default=True)

    args = parser.parse_args()

    image = melspecx.mel_spec_from_path(
        args.input_file,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        f_min=args.f_min,
        f_max=args.f_max,
        n_mels=args.n_mels,
        colormap=args.colormap,
        width_px=args.width_px,
        height_px=args.height_px,
        top_db=args.top_db,
    )
    with open(args.output_file, "wb") as f:
        f.write(image)


if __name__ == "__main__":
    main()
