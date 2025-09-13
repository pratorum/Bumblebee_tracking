#!/bin/bash

# Author: Abhishek Dutta <adutta@robots.ox.ac.uk>
# Date: 19-Nov-2021 (Updated on 12-Dec-2022)
# Updated by Cait Newport to allow for .MP4 files and to replace spaces with underscores in the output directory name. 17.07.2024


INDIR=$1
OUTDIR=$2

# Check if the correct number of arguments is provided
if [ $# -ne 2 ]; then
    echo "Usage: ${0} INDIR OUTDIR"
    exit 1
fi

echo "Extracting frames for videos contained in ${INDIR}"

# Find .mp4 files in INDIR and process each
find "$INDIR" -mindepth 1 -maxdepth 1 -type f -name '*.MP4' -print0 | while IFS= read -r -d '' file; do
    filename=$(basename "$file")
    filename_noext="${filename%.*}"
    framedir="$OUTDIR$filename_noext/"
    framedir=${framedir// /_} # replace spaces with underscores

    echo "${filename} -> ${framedir}"
    echo "Extracting frames from ${filename}"

    if [[ ! -d $framedir ]]; then
        mkdir -p "$framedir"
        if [[ $? -ne 0 ]]; then
            echo "Failed to create directory $framedir"
            continue
        fi

        </dev/null ffmpeg -nostdin -i "$file" -vsync 0 -frame_pts true -qscale:v 2 "${framedir}/f%5d.jpg"
        if [[ $? -ne 0 ]]; then
            echo "Failed to extract frames from $file"
            continue
        fi
    else
        echo "Directory $framedir already exists, skipping."
    fi
done