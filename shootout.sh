cmd_one="librosa data/suzannetrimmed.wav librosa.png --colormap inferno --width-px 4096 --height-px 512"
cmd_two="melspecx data/suzannetrimmed.wav out.png --colormap inferno --width-px 4096 --height-px 512"

hyperfine --warmup 3 "$cmd_one" "$cmd_two"
