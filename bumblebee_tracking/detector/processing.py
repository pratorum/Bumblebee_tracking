"""Classes for processing inputs and outputs of YOLO model."""
from ..qr_identifier.tag_reader import process_tag_detection

class PostProcessor:
    """Class for post processing outputs of YOLO model."""

    def __init__(self, data, data_dir=None):
        self.data = data
        self.data_dir = data_dir

    def process(self):
        """Process output."""
        return self.data
