#!/bin/bash

# Download and unpack SoloFace: A Single-Face Dataset for Resource-Constrained Face Detection and Tracking
# https://doi.org/10.5281/zenodo.14474899

url=https://zenodo.org/records/14474899/files/soloface-detection-dataset.zip?download=1

wget -V >>/dev/null 2>&1 && download_cmd="wget -O dataset.zip"
[ -z "${download_cmd}" ] && curl -V >>/dev/null 2>&1 && download_cmd="curl -o dataset.zip"
[ -z "${download_cmd}" ] && echo "Neither curl or wget is available. Stopping." && exit 1

unzip -v >>/dev/null 2>&1 || echo "unzip is not available." && exit 1

if ${download_cmd} "$url"; then
    unzip dataset.zip
fi

