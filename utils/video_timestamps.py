#
"""
video_timestamps.py
-------------------
High-level and utility functions for extracting and interpolating timestamps from video frames using OCR.

This module provides:
    - extract_timestamps_from_frames: OCR timestamp extraction from frames
    - validate_and_interpolate_timestamps: Validation and interpolation of timestamps
    - extract_and_interpolate_video_timestamps: High-level pipeline for efficient timestamp extraction/interpolation

Intended for use in video analysis pipelines where only a sparse subset of frames are OCR-processed and timestamps are interpolated for all frames.

Example usage:
timestamps, prefix, postfix, _ = extract_and_interpolate_video_timestamps(frames)
"""
import re
import datetime
import numpy as np
import scipy as sp
import cv2
import easyocr

def extract_timestamps_from_frames(
    frames: list|np.ndarray,
    crop: tuple = (slice(0, 50), slice(0, 600)),
    timestamp_color: tuple = (0, 255, 0),
    allowlist: str = '0123456789:- qpi',
    date_fmt: str = r"%Y-%m-%d %H:%M:%S",
    debug: bool = False
) -> dict:
    """
    Extracts and parses timestamps from a list of video frames using OCR.
    Returns a dict with keys: 'prefix', 'date_str', 'date_dt', 'postfix', 'ocr_result', each a list/tuple for all frames.
    """
    prefixes, date_strs, date_dts, postfixes, ocr_results = [], [], [], [], []
    reader = easyocr.Reader(['en'])
    for idx, frame in enumerate(frames):
        timestamp_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[crop[0], crop[1]]
        timestamp_bg = np.median(timestamp_img, axis=(0,1), keepdims=True)
        timestamp_mono = np.clip(np.sum(np.abs(timestamp_img - timestamp_color)**0.5, axis=-1), 0, np.sum(np.abs(timestamp_bg - timestamp_color)))
        timestamp_rgb = cv2.cvtColor(cv2.normalize(timestamp_mono, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        result = reader.readtext(timestamp_rgb, detail=0, paragraph=True, allowlist=allowlist)
        ocr_results.append(result)
        if debug:
            print(f"Frame {idx} OCR result:", result)
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
                    print(f"Frame {idx} error parsing date_str to datetime:", e)
        else:
            if debug:
                print(f"Frame {idx} no valid datetime found in OCR result:", results_str)
        prefixes.append(prefix)
        date_strs.append(date_str)
        date_dts.append(date_dt)
        postfixes.append(postfix)
    return {
        'prefix': prefixes,
        'date_str': date_strs,
        'date_dt': date_dts,
        'postfix': postfixes,
        'ocr_result': ocr_results
    }

def validate_and_interpolate_timestamps(
    timestamps: dict,
    timestamp_idx: list | np.ndarray | None = None,
    frame_idx: list | np.ndarray | None = None
) -> tuple[np.ndarray, str, str]:
    """
    Given a dict of OCR results (as from extract_timestamps_from_frames),
    validates and interpolates timestamps.

    Parameters:
        timestamps (dict): Dictionary with keys 'prefix', 'postfix', 'date_dt', etc.,
            as returned by extract_timestamps_from_frames.
        timestamp_idx (list or np.ndarray, optional): Idices of frames with entries in timestamps.
        frame_idx (list or np.ndarray, optional): Indices of frames to interpolate for.

    Returns:
        timestamps_interp (np.ndarray): Interpolated np.datetime64[ns] array for all frames.
        prefix_common (str): Most common prefix string.
        postfix_common (str): Most common postfix string.
    """
    from collections import Counter
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


def extract_and_interpolate_video_timestamps(
    frames: list|np.ndarray,
    ocr_interval: float = 120.0,
    ocr_duration: float = 2.0,
    fps: float = 30.0,
    crop: tuple = (slice(0, 50), slice(0, 600)),
    timestamp_color: tuple = (0, 255, 0),
    allowlist: str = '0123456789:- qpi',
    date_fmt: str = r"%Y-%m-%d %H:%M:%S",
    debug: bool = False
) -> tuple[np.ndarray, str, str, list]:
    """
    Run OCR on a sparse subset of frames and interpolate timestamps for all frames.

    Parameters:
        frames (list or np.ndarray): List of video frames.
        ocr_interval (float): Interval in seconds between OCR windows.
        ocr_duration (float): Duration in seconds of each OCR window.
        fps (float): Video frames per second.
        crop, timestamp_color, allowlist, date_fmt, debug: Passed to extract_timestamps_from_frames.

    Returns:
        timestamps_interp (np.ndarray): Interpolated np.datetime64[ns] array for all frames.
        prefix_common (str): Most common prefix string.
        postfix_common (str): Most common postfix string.
        idx_to_read (list): Indices of frames used for OCR.
    """
    total_frames = len(frames)
    n_intervals = int(total_frames // (ocr_interval*fps))+1
    start_frames = np.linspace(0, total_frames-ocr_duration*fps, n_intervals, dtype=int)
    idx_to_read = np.unique(np.concatenate([
        np.arange(int(start), int(start + ocr_duration*fps)).astype(int)
        for start in start_frames
    ])).tolist()
    frames_to_read = [frames[i] for i in idx_to_read]
    timestamps = extract_timestamps_from_frames(
        frames_to_read,
        crop=crop,
        timestamp_color=timestamp_color,
        allowlist=allowlist,
        date_fmt=date_fmt,
        debug=debug
    )
    timestamps_interp, prefix_common, postfix_common = validate_and_interpolate_timestamps(
        timestamps,
        timestamp_idx=idx_to_read,
        frame_idx=np.arange(total_frames)
    )
    return timestamps_interp, prefix_common, postfix_common, idx_to_read
