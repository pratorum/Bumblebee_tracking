#!/bin/bash

# Author: Abhishek Dutta <adutta@robots.ox.ac.uk>
# Date: 19-Nov-2021 (Updated on 12-Dec-2022)
# Updated by Cait Newport to allow for .MP4 files and to replace spaces with underscores in the output directory name. 17.07.2024
# Updated to support frame selection parameters (start, end, step)

# Always make sure to chmod +x the script before running it
# chmod +x extract-allframes2.sh
# ./extract-allframes2.sh /path/to/input/directory /path/to/output/directory

#MP4 files should be capitalised .MP4 not .mp4

# To run the script, use the following command:
# ./extract-allframes2.sh /path/to/input/directory /path/to/output/directory

INDIR=$1
OUTDIR=$2
START_FRAME=${3:-0}      # Default to 0 if not provided
END_FRAME=${4:-""}       # Default to empty (extract to end) if not provided
STEP=${5:-1}             # Default to 1 (every frame) if not provided

# Check if the correct number of arguments is provided
if [ $# -lt 2 ] || [ $# -gt 5 ]; then
    echo "Usage: ${0} INDIR OUTDIR [START_FRAME] [END_FRAME] [STEP]"
    echo "  INDIR: Input directory containing video files"
    echo "  OUTDIR: Output directory for extracted frames"
    echo "  START_FRAME: Starting frame number (default: 0)"
    echo "  END_FRAME: Ending frame number (default: extract to end)"
    echo "  STEP: Extract every Nth frame (default: 1)"
    echo ""
    echo "Examples:"
    echo "  ${0} ~/Downloads ~/output                    # Extract all frames"
    echo "  ${0} ~/Downloads ~/output 100 200            # Extract frames 100-200"
    echo "  ${0} ~/Downloads ~/output 0 "" 5             # Extract every 5th frame from start"
    echo "  ${0} ~/Downloads ~/output 50 150 2           # Extract every 2nd frame from 50-150"
    exit 1
fi

echo "Extracting frames for videos contained in ${INDIR}"

# Find .mp4 and .h264 files in INDIR and process each
find "$INDIR" -mindepth 1 -maxdepth 1 -type f \( -name '*.MP4' -o -name '*.h264' \) -print0 | while IFS= read -r -d '' file; do
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

        #Set desired frame rate (e.g., 30 fps)
        #FRAME_RATE=30 # Used with the changes below to stop frame duplication - but doesn't work

        # Build ffmpeg command with frame selection parameters
        FFMPEG_CMD="ffmpeg -nostdin -i \"$file\" -vsync 0 -frame_pts true -qscale:v 2"
        #FFMPEG_CMD="ffmpeg -nostdin -r $FRAME_RATE -i \"$file\" -vsync 0 -frame_pts true -qscale:v 2" #modification to suupposedly stop frame duplication

        # Add start frame if specified
        if [ "$START_FRAME" -gt 0 ]; then
            FFMPEG_CMD="$FFMPEG_CMD -start_number $START_FRAME"
        fi
        
        # Add end frame if specified
        if [ -n "$END_FRAME" ] && [ "$END_FRAME" -gt 0 ]; then
            FFMPEG_CMD="$FFMPEG_CMD -frames:v $((END_FRAME - START_FRAME + 1))"
        fi
        
        # Add step (select every Nth frame)
        if [ "$STEP" -gt 1 ]; then
            FFMPEG_CMD="$FFMPEG_CMD -vf \"select=not(mod(n\,$STEP))\""
        fi
        
        FFMPEG_CMD="$FFMPEG_CMD \"${framedir}/f%5d.jpg\""
        
        echo "Running: $FFMPEG_CMD"
        </dev/null eval $FFMPEG_CMD
        if [[ $? -ne 0 ]]; then
            echo "Failed to extract frames from $file"
            continue
        fi
    else
        echo "Directory $framedir already exists, skipping."
    fi
done