# Author: Cait Newport
# Date: 15-Sep-2025

# This script will identify the QR code in a video frame.
# The tags are from AprilTag: https://github.com/AprilRobotics/apriltag
# You will first need to install:

# 1. Get images to process from a csv file with a bounding box.
# 2. Then for each image, identify the QR code.
# 4. Then print the decoded QR code.

import cv2
import numpy as np
from pupil_apriltags import Detector
import csv
import os
import matplotlib.pyplot as plt


def identify_qr_code(image_path, bounding_box=None):
    """
    Identify and decode QR codes in an image using AprilTag detection.
    If bounding_box is provided, only search within that region.

    Args:
        image_path (str): Path to the image file
        bounding_box (str): Bounding box coordinates as "x1,y1,x2,y2" or None for full image

    Returns:
        list: List of detected QR codes with their data
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return []

    # Convert to grayscale for AprilTag detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # If bounding box is provided, crop the image
    crop_offset_x, crop_offset_y = 0, 0
    if bounding_box:
        try:
            # Parse bounding box coordinates
            coords = [int(x.strip()) for x in bounding_box.split(",")]
            if len(coords) != 4:
                print(
                    f"Warning: Invalid bounding box format: {bounding_box}. Expected x1,y1,x2,y2"
                )
                return []

            x1, y1, x2, y2 = coords

            # Ensure coordinates are within image bounds
            h, w = gray.shape
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # Crop the image to the bounding box
            gray = gray[y1:y2, x1:x2]
            crop_offset_x, crop_offset_y = x1, y1

            print(f"  Searching in bounding box: ({x1},{y1}) to ({x2},{y2})")

        except ValueError as e:
            print(f"Warning: Could not parse bounding box '{bounding_box}': {e}")
            return []

    # Circle AprilTags use the tagCircle21h7 family
    detector = Detector(families="tagCircle21h7")

    # Detect tags
    tags = detector.detect(gray)

    results = []
    for tag in tags:
        # Adjust coordinates back to original image space if we cropped
        adjusted_center = (tag.center[0] + crop_offset_x, tag.center[1] + crop_offset_y)
        adjusted_corners = [
            (corner[0] + crop_offset_x, corner[1] + crop_offset_y)
            for corner in tag.corners
        ]

        # Extract tag data
        tag_data = {
            "tag_id": tag.tag_id,
            "center": adjusted_center,
            "corners": adjusted_corners,
            "data": tag.tag_id,  # AprilTag ID serves as the "decoded" data
        }
        results.append(tag_data)

    return results


def visualize_detection(image_path, bounding_boxes, detected_tags, save_path=None):
    """
    Visualize the image with bounding boxes and detected tag IDs.

    Args:
        image_path (str): Path to the original image
        bounding_boxes (list): List of bounding box coordinates as "x1,y1,x2,y2"
        detected_tags (list): List of detected tag dictionaries
        save_path (str): Optional path to save the visualization
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title(f"QR Code Detection: {os.path.basename(image_path)}", fontsize=14)

    # Draw bounding boxes
    colors = ["red", "blue", "green", "orange", "purple"]
    for i, bbox in enumerate(bounding_boxes):
        try:
            coords = [int(x.strip()) for x in bbox.split(",")]
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                color = colors[i % len(colors)]

                # Draw rectangle
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, color=color, linewidth=2
                )
                ax.add_patch(rect)

                # Add label
                ax.text(
                    x1,
                    y1 - 5,
                    f"BBox {i+1}",
                    color=color,
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )
        except ValueError:
            print(f"Warning: Could not parse bounding box {bbox}")

    # Draw detected tags
    for tag in detected_tags:
        center = tag["center"]
        tag_id = tag["tag_id"]
        corners = tag["corners"]

        # Draw tag center
        ax.plot(center[0], center[1], "ro", markersize=8)

        # Draw tag corners
        corners_array = np.array(corners)
        ax.plot(corners_array[:, 0], corners_array[:, 1], "r-", linewidth=2)

        # Add tag ID label
        ax.text(
            center[0],
            center[1] + 15,
            f"ID: {tag_id}",
            color="red",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8),
            ha="center",
        )

    # Set axis properties
    ax.set_xlim(0, image_rgb.shape[1])
    ax.set_ylim(image_rgb.shape[0], 0)  # Flip y-axis for image coordinates
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.grid(True, alpha=0.3)

    # Add legend
    legend_elements = []
    for i, bbox in enumerate(bounding_boxes):
        color = colors[i % len(colors)]
        legend_elements.append(
            plt.Rectangle((0, 0), 1, 1, color=color, label=f"BBox {i+1}")
        )
    legend_elements.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=8,
            label="Detected Tags",
        )
    )
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()

    return fig


def process_images_from_csv(csv_file_path):
    """
    Process images listed in a CSV file with bounding box information.

    Args:
        csv_file_path (str): Path to CSV file containing image paths and bounding boxes
    """
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file {csv_file_path} not found")
        return

    # First, collect all rows and group by image_path
    image_data = {}

    with open(csv_file_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            image_path = row.get("image_path", "")
            if not image_path:
                continue

            if image_path not in image_data:
                image_data[image_path] = []
            image_data[image_path].append(row)

    # Process each unique image
    for image_path, rows in image_data.items():
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping...")
            continue

        print(f"\nProcessing: {image_path}")

        # Process each bounding box for this image
        bounding_boxes = []
        all_detected_tags = []  # Collect all detected tags for visualization

        # Process all rows for this image
        for row in rows:
            # Check for bbox1 (each row has bbox1 data)
            if (
                row.get("bbox1_x")
                and row.get("bbox1_y")
                and row.get("bbox1_w")
                and row.get("bbox1_h")
                and row.get("bbox1_x") != ""
                and row.get("bbox1_y") != ""
                and row.get("bbox1_w") != ""
                and row.get("bbox1_h") != ""
            ):
                try:
                    x = int(row["bbox1_x"])
                    y = int(row["bbox1_y"])
                    w = int(row["bbox1_w"])
                    h = int(row["bbox1_h"])
                    # Convert from x,y,w,h to x1,y1,x2,y2
                    bbox = f"{x},{y},{x+w},{y+h}"
                    bounding_boxes.append(bbox)
                except ValueError:
                    print(f"Warning: Could not parse bbox1 coordinates")

            # Check for bbox2 (if present in the same row)
            if (
                row.get("bbox2_x")
                and row.get("bbox2_y")
                and row.get("bbox2_w")
                and row.get("bbox2_h")
                and row.get("bbox2_x") != ""
                and row.get("bbox2_y") != ""
                and row.get("bbox2_w") != ""
                and row.get("bbox2_h") != ""
            ):
                try:
                    x = int(row["bbox2_x"])
                    y = int(row["bbox2_y"])
                    w = int(row["bbox2_w"])
                    h = int(row["bbox2_h"])
                    # Convert from x,y,w,h to x1,y1,x2,y2
                    bbox = f"{x},{y},{x+w},{y+h}"
                    bounding_boxes.append(bbox)
                except ValueError:
                    print(f"Warning: Could not parse bbox2 coordinates")

            # Check for additional bounding boxes (bbox3, bbox4, etc.) if they exist
            bbox_num = 3
            while True:
                bbox_x_key = f"bbox{bbox_num}_x"
                bbox_y_key = f"bbox{bbox_num}_y"
                bbox_w_key = f"bbox{bbox_num}_w"
                bbox_h_key = f"bbox{bbox_num}_h"

                if (
                    row.get(bbox_x_key)
                    and row.get(bbox_y_key)
                    and row.get(bbox_w_key)
                    and row.get(bbox_h_key)
                    and row.get(bbox_x_key) != ""
                    and row.get(bbox_y_key) != ""
                    and row.get(bbox_w_key) != ""
                    and row.get(bbox_h_key) != ""
                ):
                    try:
                        x = int(row[bbox_x_key])
                        y = int(row[bbox_y_key])
                        w = int(row[bbox_w_key])
                        h = int(row[bbox_h_key])
                        # Convert from x,y,w,h to x1,y1,x2,y2
                        bbox = f"{x},{y},{x+w},{y+h}"
                        bounding_boxes.append(bbox)
                        bbox_num += 1
                    except ValueError:
                        print(f"Warning: Could not parse bbox{bbox_num} coordinates")
                        break
                else:
                    break

        # If no bounding boxes found, search the entire image
        if not bounding_boxes:
            print("  No valid bounding boxes found, searching entire image")
            qr_codes = identify_qr_code(image_path, None)

            if qr_codes:
                print(f"Found {len(qr_codes)} QR code(s):")
                for i, qr in enumerate(qr_codes):
                    print(f"  QR Code {i+1}:")
                    print(f"    ID: {qr['tag_id']}")
                    print(f"    Center: ({qr['center'][0]:.2f}, {qr['center'][1]:.2f})")
                    print(f"    Decoded Data: {qr['data']}")
            else:
                print("  No QR codes detected")
        else:
            # Process each bounding box
            for i, bbox in enumerate(bounding_boxes):
                print(f"  Processing bounding box {i+1}: {bbox}")
                qr_codes = identify_qr_code(image_path, bbox)

                if qr_codes:
                    print(f"    Found {len(qr_codes)} QR code(s) in bbox {i+1}:")
                    for j, qr in enumerate(qr_codes):
                        print(f"      QR Code {j+1}:")
                        print(f"        ID: {qr['tag_id']}")
                        print(
                            f"        Center: ({qr['center'][0]:.2f}, {qr['center'][1]:.2f})"
                        )
                        print(f"        Decoded Data: {qr['data']}")
                    all_detected_tags.extend(qr_codes)
                else:
                    print(f"    No QR codes detected in bbox {i+1}")

            # Create visualization if any tags were detected
            if all_detected_tags:
                print(f"\n  Creating visualization...")
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                save_path = f"detection_visualization_{base_name}.png"
                visualize_detection(
                    image_path, bounding_boxes, all_detected_tags, save_path
                )


def visualize_single_image(image_path, csv_file_path="bounding_boxes.csv"):
    """
    Visualize a single image with its bounding boxes and detected QR codes.

    Args:
        image_path (str): Path to the image file
        csv_file_path (str): Path to CSV file with bounding box data
    """
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found")
        return

    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file {csv_file_path} not found")
        return

    # Find the row for this image
    with open(csv_file_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            if row.get("image_path") == image_path:
                print(f"Visualizing: {image_path}")

                # Process bounding boxes
                bounding_boxes = []
                all_detected_tags = []

                # Check for bbox1
                if (
                    row.get("bbox1_x")
                    and row.get("bbox1_y")
                    and row.get("bbox1_w")
                    and row.get("bbox1_h")
                    and row.get("bbox1_x") != ""
                    and row.get("bbox1_y") != ""
                    and row.get("bbox1_w") != ""
                    and row.get("bbox1_h") != ""
                ):
                    try:
                        x = int(row["bbox1_x"])
                        y = int(row["bbox1_y"])
                        w = int(row["bbox1_w"])
                        h = int(row["bbox1_h"])
                        bbox1 = f"{x},{y},{x+w},{y+h}"
                        bounding_boxes.append(bbox1)

                        # Detect QR codes in this bbox
                        qr_codes = identify_qr_code(image_path, bbox1)
                        if qr_codes:
                            all_detected_tags.extend(qr_codes)
                            print(f"  Found {len(qr_codes)} QR code(s) in bbox1")
                    except ValueError:
                        print(f"Warning: Could not parse bbox1 coordinates")

                # Check for bbox2
                if (
                    row.get("bbox2_x")
                    and row.get("bbox2_y")
                    and row.get("bbox2_w")
                    and row.get("bbox2_h")
                    and row.get("bbox2_x") != ""
                    and row.get("bbox2_y") != ""
                    and row.get("bbox2_w") != ""
                    and row.get("bbox2_h") != ""
                ):
                    try:
                        x = int(row["bbox2_x"])
                        y = int(row["bbox2_y"])
                        w = int(row["bbox2_w"])
                        h = int(row["bbox2_h"])
                        bbox2 = f"{x},{y},{x+w},{y+h}"
                        bounding_boxes.append(bbox2)

                        # Detect QR codes in this bbox
                        qr_codes = identify_qr_code(image_path, bbox2)
                        if qr_codes:
                            all_detected_tags.extend(qr_codes)
                            print(f"  Found {len(qr_codes)} QR code(s) in bbox2")
                    except ValueError:
                        print(f"Warning: Could not parse bbox2 coordinates")

                # Check for additional bounding boxes (bbox3, bbox4, etc.) if they exist
                bbox_num = 3
                while True:
                    bbox_x_key = f"bbox{bbox_num}_x"
                    bbox_y_key = f"bbox{bbox_num}_y"
                    bbox_w_key = f"bbox{bbox_num}_w"
                    bbox_h_key = f"bbox{bbox_num}_h"

                    if (
                        row.get(bbox_x_key)
                        and row.get(bbox_y_key)
                        and row.get(bbox_w_key)
                        and row.get(bbox_h_key)
                        and row.get(bbox_x_key) != ""
                        and row.get(bbox_y_key) != ""
                        and row.get(bbox_w_key) != ""
                        and row.get(bbox_h_key) != ""
                    ):
                        try:
                            x = int(row[bbox_x_key])
                            y = int(row[bbox_y_key])
                            w = int(row[bbox_w_key])
                            h = int(row[bbox_h_key])
                            bbox = f"{x},{y},{x+w},{y+h}"
                            bounding_boxes.append(bbox)

                            # Detect QR codes in this bbox
                            qr_codes = identify_qr_code(image_path, bbox)
                            if qr_codes:
                                all_detected_tags.extend(qr_codes)
                                print(
                                    f"  Found {len(qr_codes)} QR code(s) in bbox{bbox_num}"
                                )
                            bbox_num += 1
                        except ValueError:
                            print(
                                f"Warning: Could not parse bbox{bbox_num} coordinates"
                            )
                            break
                    else:
                        break

                # Create visualization
                if bounding_boxes:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]
                    save_path = f"visualization_{base_name}.png"
                    visualize_detection(
                        image_path, bounding_boxes, all_detected_tags, save_path
                    )
                else:
                    print("No valid bounding boxes found for this image")
                return

        print(f"Image {image_path} not found in CSV file")


def main():
    """
    Main function to demonstrate QR code identification.
    """
    print("QR Code Identification Script")
    print("=" * 40)

    # Use the actual CSV file with bounding box data
    csv_file = "bounding_boxes.csv"

    if os.path.exists(csv_file):
        process_images_from_csv(csv_file)
    else:
        print(f"CSV file '{csv_file}' not found.")
        print(
            "Please ensure the bounding_boxes.csv file exists in the current directory."
        )


if __name__ == "__main__":
    main()
