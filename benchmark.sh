#!/bin/bash
hyperfine "melspecx data/suzanne.mp3 test.png --colormap inferno --chunk-size {chunk}" --parameter-scan chunk 4096 16384 -D 1024
