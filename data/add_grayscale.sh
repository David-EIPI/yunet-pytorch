#!/bin/bash

# Optionally augment the dataset with some black and white variants. Requires ImageMagick installed.

src=soloface-detection-dataset/train

find $src/images -name *.0.jpg | while read x; do

    fname=${x%%0.jpg}5.jpg
    y=${x%%0.jpg}1.jpg
    fname2=${x%%0.jpg}6.jpg

    magick $x -grayscale Rec601Luma $fname 
    magick $y -grayscale Brightness $fname2

    echo -ne "Converting $x \r"
done

echo

find $src/labels -name *.0.json | while read x; do

    fname=${x%%0.json}5.json
    y=${x%%0.json}1.json
    fname2=${x%%0.json}6.json

    cp $x $fname
    cp $y $fname2

    echo -ne "Copying $x \r"
done

