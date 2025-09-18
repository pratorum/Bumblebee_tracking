# %%
import os
from pathlib import Path
import cv2
import numpy as np

import scipy as sp
from pupil_apriltags import Detector
import pandas as pd
from tqdm import tqdm

def detect_tags_with_rotations(gray, n_rotatations, print=False):
    """
    Detect tags in an image, optionally rotating the image to improve detection.
    Returns the best set of tags found (highest confidence).
    """
    # Circle AprilTags use the tagCircle21h7 family
    detector = Detector(families='tagCircle21h7', quad_decimate=1.0, decode_sharpening=2.0)
    if n_rotatations > 0:
        confidence = 0.0
        tags = []
        for alpha in np.linspace(0, 90, n_rotatations):
            gray_rot = sp.ndimage.rotate(gray, alpha, mode="nearest", order=5).astype(np.uint8)
            tags_candidate = detector.detect(gray_rot)
            if len(tags_candidate) > 0:
                if tags_candidate[0].decision_margin > confidence:
                    if print:
                        print(f"    Found tag {tags_candidate[0].tag_id} with confidence {tags_candidate[0].decision_margin:.2f} at rotation {alpha:.1f} degrees")
                    confidence = tags_candidate[0].decision_margin
                    tags = tags_candidate
        return tags
    else:
        return detector.detect(gray)


# %%
def load_detection(csv_file, data_dir=None):
    if isinstance(csv_file, (str, Path)):
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            return df, {}, None
        if data_dir is None:
            data_dir = Path(csv_file).parent
        frame_dir = data_dir / (df.loc[0,"video_name"] + "_frames")
    elif isinstance(csv_file, pd.DataFrame):
        df = csv_file
        if len(df) == 0:
            return df, {}, None
        if data_dir is None:
            raise ValueError("data_dir must be provided when loading from dataframe")
        frame_dir = Path(data_dir) / (df.loc[0,"video_name"] + "_frames")
    assert frame_dir.exists(), f"Frame directory {frame_dir} does not exist."
    
    img_path = {
        int(group_name): frame_dir / row["frame_filename"]
        for group_name, row in df.groupby('frame').first().iterrows()
    }
    output_csv_file = data_dir / (df.loc[0,"video_name"] + "_detections_tagged.csv")
    return df, img_path, output_csv_file

#%% get image, use a chache to avoid reloading
def get_image(source, idx):
    # this can be extended for other sources
    if isinstance(source, np.ndarray):
        img = source[idx]
    elif isinstance(source, dict):
        img = source.get(idx, None)
    if isinstance(img, (str, Path)):
        if Path(img).exists():
            img = cv2.imread(img)
        else:
            return None    
    return img
# %%
def process_tag_detection(
    csv_fileORdataframe,
    apply_to_tracks=True,
    save_csv=True,
    n_rotations=10,
    min_bbox_size=5,
    data_dir=None,
):

    df, img_path, output_csv_file = load_detection(csv_fileORdataframe, data_dir=data_dir)

    # add column for tag id and tag confidence
    df['tag_id'] = -1
    df['tag_confidence'] = 0.0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing frames"):
        frame = row['frame']
        img = get_image(img_path, frame)
        if img is None:
            continue
        dx = row['x2'] - row['x1']
        dy = row['y2'] - row['y1']
        if dx < min_bbox_size or dy < min_bbox_size:
            continue
        bb_img = img[int(row['y1']):int(row['y2']), int(row['x1']):int(row['x2'])]
        gray = cv2.cvtColor(bb_img, cv2.COLOR_BGR2GRAY)
        tags = detect_tags_with_rotations(gray, n_rotations)
        if len(tags) > 0:
            row['tag_id'] = tags[0].tag_id
            row['tag_confidence'] = tags[0].decision_margin
            df.loc[idx] = row
            tqdm.write(f"Frame {frame}: Detected tag {tags[0].tag_id} with confidence {tags[0].decision_margin:.2f}")

    if apply_to_tracks and 'track_id' in df.columns:
        best_tags = df.loc[df.groupby('track_id')['tag_confidence'].idxmax()][['track_id', 'tag_id', 'tag_confidence']]
        best_tags = best_tags.set_index('track_id')
        df['tag_id'] = df['track_id'].map(best_tags['tag_id'])
        df['tag_confidence'] = df['track_id'].map(best_tags['tag_confidence'])

    if save_csv and output_csv_file is not None:
        df.to_csv(output_csv_file, index=False)
        print(f"Saved updated CSV with tags to {output_csv_file}")

    return df

def main(csv_file=None):
    # %%
    if csv_file is not None:
        apply_to_tracks = True
        save_csv = True
        df = process_tag_detection(csv_file, apply_to_tracks=apply_to_tracks, save_csv=save_csv)

#%%
if __name__ == "__main__":
    main()

# %%
