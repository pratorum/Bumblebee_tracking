# Author: Cait Newport
# Date: 16-Sep-2025

# This script will create a function that allows the user to select the regions of interest from an image.
# The input for the function will be an image filepath, or a folder in which case it chooses the first image in the folder.
# The image will appear and the user will make a rectangle around the regions of interest.
# The user will then press the spacebar to save the regions of interest.
# The user will then press the escape key to exit the function.
# The function will return the regions of interest as a dataframe in the format x, y, w, h.
# It will also save an image with the regions of interest drawn on it.

# Import libraries
import cv2
import csv
import os
import glob
import numpy as np

# Create a function that allows the user to select the regions of interest from an image.
def select_regions_of_interest(image_path, image_number=1, save_with_roi=False, output_path=None, 
                              show_original_bbox=False, orig_bbox=None, track_id=None):
    """
    Select regions of interest from an image or folder of images.
    
    Args:
        image_path (str): Path to either a specific image file or a folder containing images
        image_number (int): Image number to select from folder (1-indexed, default is 1)
        save_with_roi (bool): Whether to save an image with ROI drawn (default: False)
        output_path (str): Path to save the image with ROI (optional, auto-generated if not provided)
    
    Returns:
        tuple: (x, y, w, h) coordinates of selected region
        str: Path to saved image with ROI (only if save_with_roi=True)
    """
    # Check if image_path is a file or directory
    if os.path.isfile(image_path):
        # It's a specific image file
        final_image_path = image_path
    elif os.path.isdir(image_path):
        # It's a folder, get list of image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_path, ext)))
            image_files.extend(glob.glob(os.path.join(image_path, ext.upper())))
        
        if not image_files:
            raise ValueError(f"No image files found in folder: {image_path}")
        
        # Sort files for consistent ordering
        image_files.sort()
        
        # Validate image_number
        if image_number < 1 or image_number > len(image_files):
            print(f"Available images in folder: {len(image_files)}")
            print("Image files:")
            for i, img_file in enumerate(image_files, 1):
                print(f"  {i}: {os.path.basename(img_file)}")
            raise ValueError(f"Image number {image_number} is out of range. Available images: 1-{len(image_files)}")
        
        # Select the specified image (convert to 0-indexed)
        final_image_path = image_files[image_number - 1]
        print(f"Selected image {image_number}: {os.path.basename(final_image_path)}")
    else:
        raise ValueError(f"Path does not exist: {image_path}")
    
    # Load the image
    image = cv2.imread(final_image_path)
    
    if image is None:
        raise ValueError(f"Could not load image: {final_image_path}")
    
    # Draw original bounding box if provided
    if show_original_bbox and orig_bbox and track_id is not None:
        orig_x1, orig_y1, orig_x2, orig_y2 = orig_bbox
        cv2.rectangle(image, (int(orig_x1), int(orig_y1)), (int(orig_x2), int(orig_y2)), (0, 0, 255), 2)
        cv2.putText(image, f"Track ID: {track_id}", (int(orig_x1), int(orig_y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    # Create a sidebar with instructions
    sidebar_width = 300
    sidebar_height = image.shape[0]
    sidebar = np.ones((sidebar_height, sidebar_width, 3), dtype=np.uint8) * 50  # Dark gray background
    
    # Add instructions text
    instructions = [
        "INSTRUCTIONS:",
        "",
        "1. Click and drag to",
        "   select a region",
        "",
        "2. Press SPACEBAR to",
        "   confirm selection",
        "",
        "3. Press 'n' for no bee",
        "",
        "4. Press 'c' to cancel",
        "",
        "5. Press ESC to exit",
        "",
        "You will see a green",
        "rectangle as you drag!"
    ]
    
    # Draw text on sidebar
    y_offset = 30
    for instruction in instructions:
        cv2.putText(sidebar, instruction, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
    
    # Variables for mouse callback
    drawing = False
    start_point = None
    end_point = None
    current_image = image.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, start_point, end_point, current_image
        
        # Ignore clicks on the sidebar
        if x >= image.shape[1]:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_point = (x, y)
            end_point = (x, y)
            print(f"Started selection at: {x}, {y}")
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                end_point = (x, y)
                # Create a copy of the original image
                temp_image = image.copy()
                # Draw the current selection rectangle
                cv2.rectangle(temp_image, start_point, end_point, (0, 255, 0), 2)
                # Combine with sidebar
                temp_combined = np.hstack((temp_image, sidebar))
                # Show the updated image
                cv2.imshow("Image", temp_combined)
                
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            end_point = (x, y)
            print(f"Finished selection at: {x}, {y}")
    
    # Set up the window and mouse callback
    cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Image", mouse_callback)
    
    # Display the initial image with sidebar
    combined_image = np.hstack((image, sidebar))
    cv2.imshow("Image", combined_image)
    
    print("Instructions displayed. Click and drag to select a region...")
    
    # Main loop
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Spacebar
            if start_point and end_point:
                # Calculate ROI coordinates
                x1 = min(start_point[0], end_point[0])
                y1 = min(start_point[1], end_point[1])
                w1 = abs(end_point[0] - start_point[0])
                h1 = abs(end_point[1] - start_point[1])
                
                if w1 > 0 and h1 > 0:  # Make sure we have a valid selection
                    print(f"ROI confirmed: x={x1}, y={y1}, w={w1}, h={h1}")
                    break
                else:
                    print("Please select a valid region (width and height must be > 0)")
            else:
                print("Please select a region first by clicking and dragging")
                
        elif key == ord('n'):  # No bee
            cv2.destroyAllWindows()
            raise ValueError("No bee - remove from list")
        elif key == ord('c'):  # Cancel
            cv2.destroyAllWindows()
            raise ValueError("ROI selection was cancelled by user")
        elif key == 27:  # ESC
            cv2.destroyAllWindows()
            raise ValueError("ROI selection was cancelled by user")
    
    # Draw final rectangle on the original image
    cv2.rectangle(image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)
    # Recreate the combined image with the final rectangle
    combined_image_with_roi = np.hstack((image, sidebar))
    # Display the updated combined image
    cv2.imshow("Image", combined_image_with_roi)
    print("Selection complete! Press any key to continue...")
    cv2.waitKey(0)
    # Close the image
    cv2.destroyAllWindows()
    
    # Save image with ROI if requested
    if save_with_roi:
        saved_path = save_image_with_roi(image_path, (x1, y1, w1, h1), output_path, image_number)
        return (x1, y1, w1, h1), saved_path
    
    # Return the regions of interest
    return x1, y1, w1, h1

def save_image_with_roi(image_path, roi, output_path=None, image_number=1):
    """
    Save an image with the regions of interest drawn as a box.
    
    Args:
        image_path (str): Path to either a specific image file or a folder containing images
        roi (tuple): Region of interest coordinates (x, y, width, height)
        output_path (str): Path to save the output image (optional)
        image_number (int): Image number to select from folder (1-indexed, default is 1)
    
    Returns:
        str: Path to the saved image
    """
    # Check if image_path is a file or directory
    if os.path.isfile(image_path):
        # It's a specific image file
        final_image_path = image_path
    elif os.path.isdir(image_path):
        # It's a folder, get list of image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(image_path, ext)))
            image_files.extend(glob.glob(os.path.join(image_path, ext.upper())))
        
        if not image_files:
            raise ValueError(f"No image files found in folder: {image_path}")
        
        # Sort files for consistent ordering
        image_files.sort()
        
        # Validate image_number
        if image_number < 1 or image_number > len(image_files):
            raise ValueError(f"Image number {image_number} is out of range. Available images: 1-{len(image_files)}")
        
        # Select the specified image (convert to 0-indexed)
        final_image_path = image_files[image_number - 1]
    else:
        raise ValueError(f"Path does not exist: {image_path}")
    
    # Load the image
    image = cv2.imread(final_image_path)
    
    if image is None:
        raise ValueError(f"Could not load image: {final_image_path}")
    
    # Unpack ROI coordinates
    x, y, width, height = roi
    
    # Draw rectangle on the image
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    
    # Add text label
    label = f"ROI: ({x}, {y}, {width}, {height})"
    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Create sidebar for saved image
    sidebar_width = 300
    sidebar_height = image.shape[0]
    sidebar = np.ones((sidebar_height, sidebar_width, 3), dtype=np.uint8) * 50  # Dark gray background
    
    # Add instructions text
    instructions = [
        "SELECTED REGION:",
        "",
        f"X: {x}",
        f"Y: {y}",
        f"Width: {width}",
        f"Height: {height}",
        "",
        "Green box shows",
        "the selected",
        "region of interest."
    ]
    
    # Draw text on sidebar
    y_offset = 30
    for instruction in instructions:
        cv2.putText(sidebar, instruction, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
    
    # Combine image and sidebar for saving
    combined_image = np.hstack((image, sidebar))
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(final_image_path))[0]
        output_path = f"{base_name}_with_roi.jpg"
    
    # Save the combined image
    cv2.imwrite(output_path, combined_image)
    
    return output_path

# Example usage when importing this module:
# from select_regions_of_interest_interactive import select_regions_of_interest
# 
# # Example 1: Select from a specific image file (no save)
# roi = select_regions_of_interest("path/to/specific/image.jpg")
# 
# # Example 2: Select and automatically save image with ROI
# roi, saved_path = select_regions_of_interest("path/to/specific/image.jpg", save_with_roi=True)
# print(f"Image with ROI saved to: {saved_path}")
# 
# # Example 3: Select and save with custom filename
# roi, saved_path = select_regions_of_interest("path/to/specific/image.jpg", 
#                                             save_with_roi=True, 
#                                             output_path="my_roi_result.jpg")
# 
# # Example 4: Select from folder and save
# roi, saved_path = select_regions_of_interest("path/to/folder/", 
#                                             image_number=2, 
#                                             save_with_roi=True)
