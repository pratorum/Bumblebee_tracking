# This script will bring a human into the loop to correct the annotations.
# The aim is to automate the identification of images that might have incorrect annotations so the user doesn't have to go through all the images.

import cv2
import numpy as np
import pandas as pd
import os
from select_regions_of_interest_interactive import select_regions_of_interest

# Load a csv file that contains annotations and confidence scores
try:
    annotations = pd.read_csv("tracking_test/Test2_20sec_detections.csv")
    print(f"Loaded {len(annotations)} annotations from Test2_20sec_detections.csv")
except FileNotFoundError:
    print("Test2_20sec_detections.csv not found. Please check the file path.")
    exit(1)

def filter_annotations_by_existing_images(annotations_df, frames_folder):
    """
    Filter annotations to only include those with existing image files.
    Creates a new CSV with only the annotations that have matching images.
    
    Args:
        annotations_df (pd.DataFrame): The original annotations dataframe
        frames_folder (str): Path to the folder containing the frame images
    
    Returns:
        pd.DataFrame: Filtered dataframe with only existing images
    """
    print(f"Checking which images exist in {frames_folder}...")
    
    # Get list of existing image files
    if not os.path.exists(frames_folder):
        print(f"Error: Frames folder {frames_folder} does not exist!")
        return pd.DataFrame()
    
    existing_files = set()
    for file in os.listdir(frames_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
            existing_files.add(file)
    
    print(f"Found {len(existing_files)} image files in the folder")
    
    # Filter annotations to only include those with existing images
    filtered_annotations = annotations_df[annotations_df['frame_filename'].isin(existing_files)].copy()
    
    print(f"Filtered from {len(annotations_df)} to {len(filtered_annotations)} annotations with existing images")
    
    # Save the filtered annotations to a new CSV
    filtered_csv_path = "tracking_test/Test2_20sec_detections_filtered.csv"
    filtered_annotations.to_csv(filtered_csv_path, index=False)
    print(f"Saved filtered annotations to: {filtered_csv_path}")
    
    # Show which images are missing
    missing_images = set(annotations_df['frame_filename']) - existing_files
    if missing_images:
        print(f"\nMissing images ({len(missing_images)}):")
        for img in sorted(missing_images):
            print(f"  - {img}")
    
    return filtered_annotations

# Filter annotations to only include existing images
frames_folder = "tracking_test/Test2_20sec_frames"
annotations = filter_annotations_by_existing_images(annotations, frames_folder)

# Now use the filtered annotations for the correction process
print(f"\nUsing filtered annotations with {len(annotations)} entries that have existing images.")

# Create a function that list images with low confidence
def list_low_confidence_images():
    low_confidence_images = annotations[annotations["confidence"] < 0.5]
    return low_confidence_images

# Create a function that lists images with no confidence
def list_no_confidence_images():
    no_confidence_images = annotations[annotations["confidence"] == 0]
    return no_confidence_images

# Create a function that will display the images and allow the user to see the image, correct the annotations. A new csv file will be created with the updated annotations.
# TODO: make an option where you say there is no bee and you don't want a bbox

def correct_annotations():
    """
    Interactive annotation correction workflow.
    Shows images with low confidence and allows users to correct annotations using the ROI selection tool.
    """
    low_confidence_images = list_low_confidence_images()
    
    if low_confidence_images.empty:
        print("No images with low confidence found!")
        return
    
    print(f"Found {len(low_confidence_images)} images with low confidence that need correction.")
    
    corrected_annotations = annotations.copy()
    indices_to_drop = []
    
    for idx, row in low_confidence_images.iterrows():
        frame_filename = row['frame_filename']
        # Construct the full path to the image (frames are in tracking_test/Test2_20sec_frames folder)
        image_path = os.path.join("tracking_test", "Test2_20sec_frames", frame_filename)
        
        print(f"\nCorrecting annotation for: {frame_filename}")
        print(f"Frame: {row['frame']}, Track ID: {row['track_id']}")
        print(f"Current confidence: {row['confidence']:.4f}")
        print(f"Current coordinates: x1={row['x1']:.1f}, y1={row['y1']:.1f}, x2={row['x2']:.1f}, y2={row['y2']:.1f}")
        
        try:
            # Use the interactive ROI selection from select_regions_of_interest.py
            print("Please select the correct region of interest...")
            print("Options:")
            print("  - Draw a rectangle and press SPACEBAR to correct the annotation")
            print("  - Press 'n' to mark as 'no bee' and remove from list")
            print("  - Press 'c' to cancel/skip this image")
            print("  - Press ESC to exit")
            
            x, y, width, height = select_regions_of_interest(image_path, show_original_bbox=True, 
                                                           orig_bbox=(row['x1'], row['y1'], row['x2'], row['y2']), 
                                                           track_id=row['track_id'])
            
            # Convert from x,y,width,height to x1,y1,x2,y2 format
            x1 = x
            y1 = y
            x2 = x + width
            y2 = y + height
            
            # Update the annotations with the corrected coordinates
            corrected_annotations.loc[idx, 'x1'] = x1
            corrected_annotations.loc[idx, 'y1'] = y1
            corrected_annotations.loc[idx, 'x2'] = x2
            corrected_annotations.loc[idx, 'y2'] = y2
            corrected_annotations.loc[idx, 'confidence'] = 1.0  # Mark as manually corrected
            
            print(f"Updated coordinates: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
            
        except ValueError as e:
            if "no bee" in str(e).lower():
                # Mark for removal from the dataframe
                print(f"Marking for removal (no bee detected)")
                indices_to_drop.append(idx)
            else:
                print(f"Skipping image due to error: {e}")
                continue
        except KeyboardInterrupt:
            print("\nAnnotation correction interrupted by user.")
            break
    
    # Remove all annotations marked as "no bee"
    if indices_to_drop:
        print(f"\nRemoving {len(indices_to_drop)} annotations marked as 'no bee'")
        corrected_annotations = corrected_annotations.drop(indices_to_drop)
    
    # Save the corrected annotations
    output_file = "tracking_test/annotations_corrected.csv"
    corrected_annotations.to_csv(output_file, index=False)
    print(f"\nCorrected annotations saved to: {output_file}")
    
    return corrected_annotations

# Main execution
if __name__ == "__main__":
    print("Starting annotation correction workflow...")
    corrected_annotations = correct_annotations()


