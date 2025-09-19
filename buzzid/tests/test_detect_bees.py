"""Tests for detect_bees.py"""

import pytest
from buzzid.detect_bees import BeeDetectorApp


class TestBeeDetectorApp:
    """Tests for the BeeDetectorApp class."""
    model = BeeDetectorApp

    def test_validate_args_no_video(self, monkeypatch, arguments_invalid_video):
        """Assert SystemError raised when invalid video file path parsed."""
        monkeypatch.setattr("sys.argv", arguments_invalid_video)
        test_app = self.model()

        with pytest.raises(SystemExit):
            test_app.validate_args()

    def test_validate_args_no_model(self, monkeypatch, arguments_invalid_model):
        """Assert SystemError raised when invalid model file path parsed."""
        monkeypatch.setattr("sys.argv", arguments_invalid_model)
        test_app = self.model()

        with pytest.raises(SystemExit):
            test_app.validate_args()
