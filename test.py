import melspecx

audio, sr = melspecx.read_wav("./data/suzannetrimmed.wav", normalize=True)

mel_spec = melspecx.mel_spec(audio, sr, 2048, 1024, 512, 0, 8000, 128, 20, True)
image = melspecx.plot_mel_spec_py(mel_spec, "inferno", 1000, 1000)

# save image to file
with open("mel_spec.png", "wb") as f:
    f.write(image)
