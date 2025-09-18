"""Module for YOLO detector model."""

from pathlib import Path

import pandas as pd
from ultralytics import YOLO


class BeeDetector:
    """Class for using YOLO model to detect bees in video data."""

    def __init__(
        self,
        model_path: str,
        video_path: str,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.70,
        output_dataframe=None,
    ):
        self.model_path = model_path
        self.video_path = video_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.output_dataframe = output_dataframe
        self.output_dir = None

    def track_bees_in_video(self):
        """Detect and track bees and save both video and CSV to same directory."""
        self.model_path = Path(self.model_path).resolve()
        self.video_path = Path(self.video_path).resolve()
        assert self.model_path.exists(), f"Model path {self.model_path} does not exist."
        assert self.video_path.exists(), f"Video path {self.video_path} does not exist."
        model = YOLO(self.model_path)
        rows = []

        # Get video name for consistent naming
        video_name = Path(self.video_path).stem  # "Test2_20sec"

        # Run tracking - YOLO handles the output directory
        results = model.track(
            source=self.video_path,
            tracker="bytetrack.yaml",
            save=True,
            save_frames=True,
            project="results",
            name=f"tracking_{video_name}",  # Use video name in folder
            stream=True,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
        )

        # Get output directory from the predictor
        self.output_dir = Path(model.predictor.save_dir)

        # Extract tracking data
        for frame_idx, r in enumerate(results):
            if r.boxes is None or len(r.boxes) == 0:
                continue

            xyxy = r.boxes.xyxy.tolist()
            conf = r.boxes.conf.tolist() if r.boxes.conf is not None else [None] * len(xyxy)
            cls = r.boxes.cls.int().tolist() if r.boxes.cls is not None else [-1] * len(xyxy)
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(xyxy)

            for (x1, y1, x2, y2), c, k, tid in zip(xyxy, conf, cls, ids):
                rows.append(
                    {
                        "video_name": video_name,  # Add this
                        "frame": frame_idx,
                        "frame_filename": f"{video_name}_{frame_idx}.jpg",  # Add this
                        "track_id": int(tid),
                        "class_id": int(k),
                        "confidence": float(c),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    }
                )

        # Save CSV with video name
        df = pd.DataFrame(rows)
        csv_path = self.output_dir / f"{video_name}_detections.csv"  # Name CSV after video
        df.to_csv(csv_path, index=False)
        self.output_dataframe = df

        print("Tracking complete!")
        print(f"- Total detections: {len(df)}")
        print(f"- Unique tracks: {df['track_id'].nunique()}")
        print(f"- Results saved to: {self.output_dir}")

        return self.output_dataframe


# if __name__ == "__main__":
#     df = track_bees_in_video("../models/YOLOv8s_10epochs.pt", "../data/Test2_5sec.MP4")
