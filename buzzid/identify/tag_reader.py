# %%
import os
from pathlib import Path
import cv2
import numpy as np
import scipy as sp
from pupil_apriltags import Detector
import pandas as pd
from tqdm import tqdm


def detect_tags_with_rotations(gray: np.ndarray, n_rotatations: int, print: bool = False) -> list:
    """
    Detect AprilTags in a grayscale image, optionally rotating the image to improve detection.
    Returns a list of detected tags (best set found by confidence).
    """
    # Circle AprilTags use the tagCircle21h7 family
    detector = Detector(families="tagCircle21h7", quad_decimate=1.0, decode_sharpening=2.0)
    if n_rotatations > 0:
        confidence = 0.0
        tags = []
        for alpha in np.linspace(0, 90, n_rotatations):
            gray_rot = sp.ndimage.rotate(gray, alpha, mode="nearest", order=5).astype(np.uint8)
            tags_candidate = detector.detect(gray_rot)
            if len(tags_candidate) > 0:
                if tags_candidate[0].decision_margin > confidence:
                    if print:
                        print(
                            f"    Found tag {tags_candidate[0].tag_id} with confidence {tags_candidate[0].decision_margin:.2f} at rotation {alpha:.1f} degrees"
                        )
                    confidence = tags_candidate[0].decision_margin
                    tags = tags_candidate
        return tags
    else:
        return detector.detect(gray)


def load_detection(csv_file, data_dir=None) -> tuple:
    """
    Load detection data from a CSV file or DataFrame and return the DataFrame, a mapping from frame to image path, and the output CSV path.
    """
    if isinstance(csv_file, (str, Path)):
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            return df, {}, None
        if data_dir is None:
            data_dir = Path(csv_file).parent
        frame_dir = data_dir / (df.loc[0, "video_name"] + "_frames")
    elif isinstance(csv_file, pd.DataFrame):
        df = csv_file
        if len(df) == 0:
            return df, {}, None
        if data_dir is None:
            raise ValueError("data_dir must be provided when loading from dataframe")
        frame_dir = Path(data_dir) / (df.loc[0, "video_name"] + "_frames")
    assert frame_dir.exists(), f"Frame directory {frame_dir} does not exist."

    img_path = {
        int(group_name): frame_dir / row["frame_filename"]
        for group_name, row in df.groupby("frame").first().iterrows()
    }
    output_csv_file = data_dir / (df.loc[0, "video_name"] + "_detections_tagged.csv")
    return df, img_path, output_csv_file


# %%
def get_image(source, idx: int) -> np.ndarray | None:
    """
    Retrieve an image from a source (dict, ndarray, or path). Returns None if not found.
    """
    if isinstance(source, np.ndarray):
        img = source[idx]
    elif isinstance(source, dict):
        img = source.get(idx, None)
    if isinstance(img, (str, Path)):
        if Path(img).exists():
            img = cv2.imread(str(img))
        else:
            return None
    return img


def process_tag_detection(
    csv_fileORdataframe,
    apply_to_tracks: bool = True,
    save_csv: bool = True,
    n_rotations: int = 10,
    min_bbox_size: int = 5,
    min_confidence: float = 100.0,
    data_dir=None,
) -> pd.DataFrame:
    """
    Process tag detection for bounding boxes in a CSV or DataFrame.
    Optionally applies best tag selection per track and saves results.
    Returns the updated DataFrame.
    """

    df, img_path, output_csv_file = load_detection(csv_fileORdataframe, data_dir=data_dir)

    # add column for tag id and tag confidence
    df["tag_id"] = -1
    df["tag_confidence"] = 0.0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing frames"):
        frame = row["frame"]
        img = get_image(img_path, frame)
        if img is None:
            continue
        dx = row["x2"] - row["x1"]
        dy = row["y2"] - row["y1"]
        if dx < min_bbox_size or dy < min_bbox_size:
            continue
        bb_img = img[int(row["y1"]) : int(row["y2"]), int(row["x1"]) : int(row["x2"])]
        gray = cv2.cvtColor(bb_img, cv2.COLOR_BGR2GRAY)
        tags = detect_tags_with_rotations(gray, n_rotations)
        if len(tags) > 0 and tags[0].decision_margin >= min_confidence:
            row["tag_id"] = tags[0].tag_id
            row["tag_confidence"] = tags[0].decision_margin
            df.loc[idx] = row
            tqdm.write(
                f"Frame {frame}: Detected tag {tags[0].tag_id} with confidence {tags[0].decision_margin:.2f}"
            )

    if apply_to_tracks and "track_id" in df.columns:
        min_counts = 3

        def select_most_frequent_tag(group):
            tag_counts = group["tag_id"].value_counts()
            tag_id = -1
            if (
                not tag_counts.empty
                and tag_counts.iloc[0] >= min_counts
                and tag_counts.index[0] != -1
            ):
                tag_id = tag_counts.index[0]
            return pd.Series({"tag_id": tag_id})

        # Get the most frequent tag_id for each track_id (if it appears at least min_counts times)
        best_tags = df.groupby("track_id").apply(select_most_frequent_tag)
        # Map the selected tag_id back to the dataframe
        df["tag_id"] = df["track_id"].map(best_tags["tag_id"])

    if save_csv and output_csv_file is not None:
        df.to_csv(output_csv_file, index=False)
        print(f"Saved updated CSV with tags to {output_csv_file}")

    return df
