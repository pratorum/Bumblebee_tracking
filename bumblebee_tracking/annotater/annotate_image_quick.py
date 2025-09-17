import cv2
import csv
import os

# Load image
image_path = "qr_code_images/vlcsnap-2025-09-15-11h11m15s941c.png"
image = cv2.imread(image_path)

# Create a copy for drawing
display_image = image.copy()

# Select first ROI
print("Select first bounding box...")
r1 = cv2.selectROI("Select First ROI", display_image, fromCenter=False, showCrosshair=True)

# r1 returns (x, y, w, h)
x1, y1, w1, h1 = r1
print(f"Selected first bbox: x={x1}, y={y1}, w={w1}, h={h1}")

# Draw the first bounding box
cv2.rectangle(display_image, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

# Ask if user wants to add a second bounding box
add_second = input("Do you want to add a second bounding box? (y/n): ").lower().strip()

# Initialize second bounding box variables
x2, y2, w2, h2 = None, None, None, None

if add_second in ["y", "yes"]:
    # Select second ROI
    print("Select second bounding box...")
    r2 = cv2.selectROI("Select Second ROI", display_image, fromCenter=False, showCrosshair=True)

    # r2 returns (x, y, w, h)
    x2, y2, w2, h2 = r2
    print(f"Selected second bbox: x={x2}, y={y2}, w={w2}, h={h2}")

    # Draw the second bounding box
    cv2.rectangle(display_image, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)

# Show result
cv2.imshow("Bounding Boxes", display_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save to CSV
csv_filename = "bounding_boxes.csv"
file_exists = os.path.exists(csv_filename)

with open(csv_filename, "a", newline="") as csvfile:
    fieldnames = ["image_path", "bbox_id", "x", "y", "w", "h"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write header if file is new
    if not file_exists:
        writer.writeheader()

    # Write first bounding box
    writer.writerow({"image_path": image_path, "bbox_id": 1, "x": x1, "y": y1, "w": w1, "h": h1})

    # Write second bounding box only if it was selected
    if x2 is not None:
        writer.writerow(
            {"image_path": image_path, "bbox_id": 2, "x": x2, "y": y2, "w": w2, "h": h2}
        )

print(f"Bounding box data saved to {csv_filename}")
