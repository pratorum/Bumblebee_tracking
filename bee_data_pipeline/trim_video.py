# This script will trim a video given a start and end time. It will also compress the video using ffmpeg.

import ffmpeg
import subprocess
import os

def test_ffmpeg():
    """
    Check if FFmpeg is installed and working.
    """
    try:
        # Run ffmpeg -version command
        result = subprocess.run(['ffmpeg', '-version'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
        
        if result.returncode == 0:
            print("FFmpeg is installed and working!")
            print("Version info:", result.stdout.decode().split('\n')[0])
            return True
        else:
            print("FFmpeg appears to be installed but returned an error.")
            print("Error:", result.stderr.decode())
            return False
            
    except FileNotFoundError:
        print("FFmpeg is not installed or not in system PATH")
        return False

def check_video(input_path):
    """
    Check if the video is valid.
    """
    if os.path.exists(input_path):
        print(f"The file {input_path} exists.")
        return True
    else:
        print(f"The file {input_path} does not exist.")
        return False

def get_video_length(input_path):
    """
    Get the length of a video in seconds using ffprobe.
    """
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except Exception as e:
        print(f"Error getting video length: {e}")
        return None

def trim_video(input_path, output_path, start_time, end_time):
    """
    Trim a video given a start and end time.
    """

    # Get the video length
    video_length = get_video_length(input_path)
    if video_length is None:
        print("Unable to determine video length. Exiting.")
    
    print(f"Video length: {video_length} seconds, maximum length: {end_time} seconds")
    
    # Convert start and end times to seconds
    try:
        start_seconds = convert_to_seconds(start_time)
        print(f"Start time: {start_seconds} seconds")
        end_seconds = convert_to_seconds(end_time)
        print(f"End time: {end_seconds} seconds")
    except ValueError as e:
        print(f"Invalid time format: {e}")
        return
    
    # Validate time ranges
    if start_seconds < 0 or end_seconds > video_length or start_seconds >= end_seconds:
        print("Invalid start or end time. Exiting.")
        return
    
    # Check if the output path is valid
    if not os.path.exists(output_path):
        print(f"Error: The output directory does not exist: {output_path}")
        return
    
    # Create an output file name
    output_file = os.path.join(output_path, os.path.basename(input_path).split('.')[0] + "_trimmed.mp4")
    print(f"Output file path: {output_file}")

    # ffmpeg settings
    command = [
        'ffmpeg',
        #'-hwaccel', 'videotoolbox',
        '-i', str(input_path),
        '-ss', str(start_seconds),  # Start time in seconds
        '-to', str(end_seconds),    # End time in seconds
        '-vf', 'fps=60,scale=1920:1080',
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'copy',  # Copy audio stream without re-encoding
        str(output_file)
    ]

    # Execute ffmpeg command
    try:
        subprocess.run(command, check=True)
        print(f"Video trimmed successfully. Output saved to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error trimming video: {e}")

  
def convert_to_seconds(time_str):
    """
    Convert a time string in HH:MM:SS or MM:SS format to seconds.
    """
    parts = list(map(int, time_str.split(":")))
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        return parts[0] * 60 + parts[1]
    else:
        raise ValueError("Invalid time format. Use HH:MM:SS or MM:SS.")


if __name__ == "__main__":
    test_ffmpeg()

    video_path = "../../../QR_code_videos/qpi4_Test_7.mp4"
    output_path = "../../../QR_code_videos/Output"  # Ensure this is a directory
    start_time = "01:56:20"
    end_time = "02:04:20"

    get_video_length(video_path)
    check_video(video_path)
    trim_video(video_path, output_path, start_time, end_time)
    print("Done")
   