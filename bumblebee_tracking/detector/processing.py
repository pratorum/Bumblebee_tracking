"""Classes for processing inputs and outputs of YOLO model."""
from ..qr_identifier.tag_reader import process_tag_detection
from ..timestamp_reader.timestamp_reader import add_timestamps_to_dataframe

class PostProcessor:
    """Class for post processing outputs of YOLO model."""

    def __init__(self, data, data_dir=None):
        self.data = data
        self.data_dir = data_dir

    def process(self, tag_apply_to_tracks=True, tag_save_csv=True, read_timestamps=True):
        """Process output."""
        # Call the tag detection processing function
        self.data = process_tag_detection(self.data, data_dir=self.data_dir, apply_to_tracks=tag_apply_to_tracks, save_csv=tag_save_csv)

        if read_timestamps:
            self.data = add_timestamps_to_dataframe(self.data, data_dir=self.data_dir)
        return self.data
