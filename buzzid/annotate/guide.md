## User Examples
### Select Regions of Interest

Import the module:

```python
from buzzid.selector.regions import select_regions_of_interest
```

```python
# Example 1: Select from a specific image file and automatically save with ROI
roi, saved_path = select_regions_of_interest("f00400.jpg", save_with_roi=True)
```
```python
# Example 2: Select without saving (original behavior)
roi = select_regions_of_interest("f00400.jpg")
```

```python
# Example 3: Select and save with custom filename
roi, saved_path = select_regions_of_interest(
    "f00400.jpg",
    save_with_roi=True,
    output_path="my_custom_roi.jpg"
    )
```

```python
# Example 4: Select from folder and save
try:
    roi, saved_path = select_regions_of_interest(
        "qr_code_images",
        image_number=2,
        save_with_roi=True
        )

    # Print the ROI information
    x, y, width, height = roi
    print(f"Region: x={x}, y={y}, width={width}, height={height}")
    print(f"Image with ROI saved to: {saved_path}")

except ValueError as e:
    print(f"Selection cancelled: {e}")
```
