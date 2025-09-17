"""Module for YOLO detector model."""


class BeeDetector:
    """Class for using YOLO model to detect bees in video data."""

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.25,
        output_data=None,
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.output_data = output_data

    def detect(self, video_path, output_dir, show):
        """Placeholder for functionality."""
        return self.output_data
