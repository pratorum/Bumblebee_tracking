import re
import datetime
from collections import Counter
import numpy as np
import scipy as sp
import cv2
import easyocr
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from bumblebee_tracking.qr_identifier.tag_reader import load_detection, get_image

class TimestampExtractor:
	def __init__(self, lang_list=['en'], gpu=False):
		self.reader = easyocr.Reader(lang_list, gpu=gpu)

	def extract_timestamp_from_image(
		self,
		frame,
		crop: tuple = (slice(0, 80), slice(0, 800)),
		timestamp_color: tuple = (0, 255, 0),
		allowlist: str = '0123456789:- qpi',
		date_fmt: str = r"%Y-%m-%d %H:%M:%S",
		debug: bool = False
	) -> dict:
		"""
		Extracts and parses timestamp from a single video frame using OCR.
		Returns a dict with keys: 'prefix', 'date_str', 'date_dt', 'postfix', 'ocr_result'.
		"""
		timestamp_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[crop[0], crop[1]]
		timestamp_bg = np.median(timestamp_img, axis=(0,1), keepdims=True)
		timestamp_mono = np.clip(np.sum(np.abs(timestamp_img - timestamp_color)**0.5, axis=-1), 0, np.sum(np.abs(timestamp_bg - timestamp_color)))
		timestamp_rgb = cv2.cvtColor(cv2.normalize(timestamp_mono, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
		result = self.reader.readtext(timestamp_rgb, detail=0, paragraph=True, allowlist=allowlist)
		if debug:
			print(f"OCR result: {result}")
			import matplotlib.pyplot as plt
			plt.imshow(timestamp_mono, cmap='gray')
			plt.show()
		results_str = " ".join(result)
		match = re.search(r"(.+?)\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(.*)", results_str)
		prefix = date_str = date_dt = postfix = None
		if match:
			prefix = match.group(1).strip()
			date_str = match.group(2)
			postfix = match.group(3).strip()
			try:
				date_dt = datetime.datetime.strptime(date_str, date_fmt)
			except ValueError as e:
				if debug:
					print(f"Error parsing date_str to datetime: {e}")
		else:
			if debug:
				print(f"No valid datetime found in OCR result: {results_str}")
		return {
			'prefix': prefix,
			'date_str': date_str,
			'date_dt': date_dt,
			'postfix': postfix,
			'ocr_result': result
		}

def validate_and_interpolate_timestamps(
	timestamps: dict,
	timestamp_idx: list | np.ndarray | None = None,
	frame_idx: list | np.ndarray | None = None
) -> tuple[np.ndarray, str, str]:
	"""
	Given a dict of OCR results (as from extract_timestamp_from_image),
	validates and interpolates timestamps.
	"""
	def most_common(lst):
		return Counter(lst).most_common(1)[0][0]
	prefix_common = most_common(timestamps['prefix'])
	postfix_common = most_common(timestamps['postfix'])
	timestamps_dt = np.array([np.datetime64(dt) if dt is not None else np.datetime64('NaT') for dt in timestamps['date_dt']])
	time_valid = np.diff(timestamps_dt, prepend=np.datetime64('NaT')) > np.timedelta64(1)
	idx_valid = np.nonzero(time_valid)[0]
	timestamp_idx = idx_valid if timestamp_idx is None else np.asarray(timestamp_idx)[idx_valid]
	if len(idx_valid) < 2:
		timestamps_interp = np.full_like(timestamps_dt, np.datetime64('NaT'))
	else:
		t0 = np.datetime64('1970-01-01T00:00:00', 'ns')
		ts_ns = (timestamps_dt[idx_valid] - t0).astype('timedelta64[ns]').astype(np.int64)
		idx_all = np.arange(len(timestamps_dt)) if frame_idx is None else frame_idx
		f_interp = sp.interpolate.interp1d(timestamp_idx, ts_ns, kind='linear', fill_value='extrapolate')
		ts_interp_ns = f_interp(idx_all)
		timestamps_interp = t0 + ts_interp_ns.astype('timedelta64[ns]')
	return timestamps_interp, prefix_common, postfix_common

def add_timestamps_to_dataframe(
	csv_file_or_df,
	data_dir=None,
	crop=(slice(0, 80), slice(0, 800)),
	n_blocks=4,
	block_length=20,
	timestamp_color=(0, 255, 0),
	allowlist='0123456789:- qpi',
	date_fmt=r"%Y-%m-%d %H:%M:%S",
	debug=False,
	extractor=None,
	save_csv=True,
):
    """
	Adds a 'timestamp' column to the DataFrame by extracting and interpolating timestamps for a sparse subset of frames.
	Only reads frames needed for OCR, not all frames.
	Parameters:
		df (pd.DataFrame): DataFrame with a 'frame' column.
		img_path (dict): Mapping from frame index to image path (see tag_reader.py:load_detection).
		fps (float): Frames per second of the video.
		ocr_interval (float): Interval in seconds between OCR windows.
		ocr_duration (float): Duration in seconds of each OCR window.
		extractor (TimestampExtractor, optional): Pass a TimestampExtractor instance to cache the OCR reader.
		... (other params): Passed to extract_timestamp_from_image.
	Returns:
		pd.DataFrame: DataFrame with an added 'timestamp' column (np.datetime64[ns]).
	"""
	
    df, img_path, csv_output_file = load_detection(csv_file_or_df, data_dir=data_dir, suffix="_timestamped")
    if len(df) == 0 or len(img_path) == 0:
        df['timestamp'] = pd.Series(dtype='datetime64[ns]')
        return df

    if extractor is None:
        extractor = TimestampExtractor()
    total_frames = df['frame'].max() + 1
    # Select n_blocks of consecutive indices of length block_length from sorted idx_to_read
    sorted_indices = np.sort(list(img_path.keys()))
    if len(sorted_indices) < n_blocks * block_length:
        # Not enough indices, just use all
        idx_to_read = sorted_indices.tolist()
    else:
        # Evenly space the start of each block
        block_starts = np.linspace(0, len(sorted_indices) - block_length, n_blocks, dtype=int)
        idx_to_read = np.concatenate([
            sorted_indices[start:start+block_length] for start in block_starts
        ]).tolist()

    # Only read images for idx_to_read
    frames_to_read = []
    idx_valid = []
    for i in idx_to_read:
        img = get_image(img_path, i)
        if img is not None:
            frames_to_read.append(img)
            idx_valid.append(i)
        else:
            frames_to_read.append(None)
            idx_valid.append(i)
    # OCR only on valid images
    ocr_results = []
    for img in tqdm(frames_to_read, desc="Extracting timestamps"):
        if img is not None:
            result = extractor.extract_timestamp_from_image(
                img,
                crop=crop,
                timestamp_color=timestamp_color,
                allowlist=allowlist,
                date_fmt=date_fmt,
                debug=debug,
            )
        else:
            result = {
                "prefix": None,
                "date_str": None,
                "date_dt": None,
                "postfix": None,
                "ocr_result": [],
            }
        ocr_results.append(result)
    # Build timestamps dict
    timestamps = {
        "prefix": [r["prefix"] for r in ocr_results],
        "date_str": [r["date_str"] for r in ocr_results],
        "date_dt": [r["date_dt"] for r in ocr_results],
        "postfix": [r["postfix"] for r in ocr_results],
        "ocr_result": [r["ocr_result"] for r in ocr_results],
    }
    # Interpolate for all frames
    timestamps_interp, _, _ = validate_and_interpolate_timestamps(
		timestamps,
		timestamp_idx=idx_valid,
		frame_idx=np.arange(total_frames)
	)
    # Map timestamps to DataFrame rows based on 'frame' column
    df['timestamp'] = df['frame'].apply(lambda idx: timestamps_interp[idx] if 0 <= idx < len(timestamps_interp) else np.datetime64('NaT'))
	
    if save_csv and csv_output_file is not None:
        df.to_csv(csv_output_file, index=False)
        if debug:
            print(f"Saved timestamped DataFrame to {csv_output_file}")
    return df
