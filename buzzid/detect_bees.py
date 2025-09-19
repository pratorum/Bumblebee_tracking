"""Detect bees in video data with chosen model"""

import argparse
import os
import sys

from buzzid.detect.models import BeeDetector
from buzzid.detect.processing import Analyzer, PostProcessor
from buzzid.annotate.correct_annotations import correct_annotations


class BeeDetectorApp:
    """Command line interface for bee detector app"""

    def __init__(self):
        self.args = self.parse_args()

    @staticmethod
    def _get_data(path):
        """Private function to fix path for package data."""
        _ROOT = os.path.abspath(os.path.dirname(__file__))
        return os.path.join(_ROOT, path)

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
            default=self._get_data("detect/trained_model.pt"),
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
            "--iou-thresh",
            "-i",
            type=float,
            default=0.70,
            help="NMS IoU threshold - lower to reduce overlapping boxes (0.0 to 1.0)",
        )
        parser.add_argument(
            "--output",
            "-o",
            type=str,
            default="outputs/",
            help="Directory to save output results (frames, video, etc.)",
        )
        parser.add_argument("--analyze", "-a", action="store_true", help="Perfrom statistical analysis on results")
        parser.add_argument("--interactive", "-i", action="store_true", help="Run BuzzID in interactive mode to update selected regions")
        return parser.parse_args()

    def validate_args(self):
        """Check that input arguments are valid."""
        # Sorry for this horrible hack
        if self.args.video == "test":
            self.args.video = self._get_data("detect/example_video.mp4")

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
        print(f"NMS IoU threshold: {self.args.iou_thresh}")
        print(f"Output directory: {self.args.output}")

        detector = BeeDetector(model_path=self.args.model, video_path=self.args.video, conf_thresh=self.args.conf_thresh, iou_thresh=self.args.iou_thresh)
        detector.track_bees_in_video()
        processor = PostProcessor(data=detector.output_dataframe, data_dir=detector.output_dir)
        processor.process()

        if self.args.analyze:
            analyzer = Analyzer(df=processor.data, output_dir=self.args.output)
            analyzer.analyze()


def main():
    """Function to be run as 'buzzid' in pyproject.toml"""
    app = BeeDetectorApp()
    app.run()
