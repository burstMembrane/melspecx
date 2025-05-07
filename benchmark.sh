#!/bin/bash
hyperfine "python cli.py  data/suzannetrimmed.wav test.png --colormap inferno --chunk_size {chunk}" --parameter-scan chunk 4096 8192 -D 1024
