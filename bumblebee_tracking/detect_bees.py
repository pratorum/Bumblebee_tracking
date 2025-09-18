"""Detect bees in video data with chosen model"""

import argparse
import importlib.resources
import os
import sys

# TODO: make modules for different processes
from bumblebee_tracking.detector.models import BeeDetector
from bumblebee_tracking.detector.processing import Analyzer, PostProcessor


class BeeDetectorApp:
    """Command line interface for bee detector app"""

    def __init__(self):
        self.args = self.parse_args()

    def parse_args(self):
        """argument parser for CLI."""
        parser = argparse.ArgumentParser(description="Bee Detection in Videos using YOLO")
        parser.add_argument(
            "--video", "-v", type=str, required=True, help="Path to the input video file"
        )
        parser.add_argument(
            "--model",
            "-m",
            type=str,
            default="bumblebee_tracking/detector/trained_model.pt",
            help="Path to the YOLO model weights file, defaults to pre-trained model",
        )
        parser.add_argument(
            "--conf-thresh",
            "-c",
            type=float,
            default=0.25,
            help="Confidence threshold for detections (0.0 to 1.0)",
        )
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            default="outputs/",
            help="Directory to save output results (frames, video, etc.)",
        )
        parser.add_argument(
            "--show",
            "-s",
            action="store_true",
            help="Display the video with detections while processing",
        )
        parser.add_argument("--analyze", "-a", default=True, help="Perfrom statistical analysis ")
        return parser.parse_args()

    def validate_args(self):
        """Check that input arguments are valid."""
        # Sorry for this horrible hack
        if self.args.video == "test":
            self.args.video = importlib.resources.path(
                "bumblebee_tracking.detector", "example_video.mp4"
            )

        if not os.path.isfile(self.args.video):
            print(f"Video file not found: {self.args.video}")
            sys.exit(1)

        if not os.path.isfile(self.args.model):
            print(f"Model file not found: {self.args.model}")
            sys.exit(1)

        os.makedirs(self.args.output, exist_ok=True)

    def run(self):
        """Run the detector with given arguments."""
        self.validate_args()

        print(f"Starting bee detection on: {self.args.video}")
        print(f"Using model: {self.args.model}")
        print(f"Confidence threshold: {self.args.conf_thresh}")
        print(f"Output directory: {self.args.output}")
        if self.args.show:
            print("Display enabled: Will show video while processing")

        detector = BeeDetector(model_path=self.args.model, video_path=self.args.video)
        detector.track_bees_in_video()
        processor = PostProcessor(data=detector.output_dataframe, data_dir=detector.output_dir)
        processor.process()

        if self.args.analyze:
            analyzer = Analyzer(df=processor.data, output_dir=self.args.output)
            analyzer.analyze()


if __name__ == "__main__":
    app = BeeDetectorApp()
    app.run()
