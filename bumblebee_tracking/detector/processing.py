"""Classes for processing inputs and outputs of YOLO model."""


class PostProcessor:
    """Class for post processing outputs of YOLO model."""

    def __init__(self, data):
        self.data = data

    def process(self):
        """Process output."""
        return self.data
