# BuzzID
Tracking and monitoring behaviour of bumblebees with QR codes and YOLO deteciton algorithm

## Installation

We recommend you use this library in a virtual environment:

```bash
python -m venv venv
```

To activate the environment, use:

```bash
source venv/bin/activate
```

Use `deactivate` to leave the enironment.

**This project requires `Python3.10` as minimum, please ensure you are using `Python3.10` or higher**

To create a virtualenv with a specific Python version e.g. `python3.10 -m venv venv` or use `<path-to-python> -m venv venv`.

To install the module, use:

```
pip install git+https://github.com/pratorum/buzzid.git
```

## Quickstart

To use the detection algorithm, simply parse your video file to the CLI as below:

```python
python3 -m bumblebee_tracking.detect_bees --video <path-to-video> --conf-thresh <thresh-value> --iou-thresh <thresh-value>
```

**Alternatively**, test with an example video:

```bash
buzzid --video test
```

The detector uses a default trained model, but you can use the `--model` flag to parse the path to a model weights file of your choice.

To output analysis files use the `--analyze` flag.

To see a available options and usage instructions, run:
```bash
buzzid -h
```

## Developer Installation
If you want to install the development version, please use the instructions below.
Once you have cloned down this repository using `git clone`, cd into the app directory:

```bash
git clone git@github.com:pratorum/buzzid.git
cd buzzid
```

Create a virtual environment and install the package:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e
```

If you want to create a virtualenv with a specific python version use <path-to-python> -m venv venv

If you want to install the development tools as well, use the following commands if you are are using **bash**:

```bash
pip install -e .[dev]
```

If you are using **zsh**:

```zsh
pip install -e ".[dev]"
```

To exit the virtual environment use `deactivate`.

To run the test suite, use `python -m pytest`.


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
