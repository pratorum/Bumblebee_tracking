"""conftest.py for pytest fixtures"""

import pytest


@pytest.fixture
def invalid_video():
    """Fixture for video file that doesn't exist."""
    return "invalid_video_file_path.mp4"


@pytest.fixture
def invalid_model():
    """Fixture for model file that doesn't exist."""
    return "invalid_model_file_path.mp4"


@pytest.fixture
def valid_video():
    """Fixture for video file."""
    return "bumblebee_tracking/tests/test_video.mp4"


@pytest.fixture
def valid_model():
    """Fixture for model file."""
    return "bumblebee_tracking/tests/test_model.mp4"


@pytest.fixture
def arguments(valid_video, valid_model):
    """Fixture for CLI arguments"""
    return [
        "program_name",
        "--video",
        valid_video,
        "--model",
        valid_model,
        ]


@pytest.fixture
def arguments_invalid_video(invalid_video, valid_model):
    """Fixture for CLI arguments"""
    return [
        "program_name",
        "--video",
        invalid_video,
        "--model",
        valid_model,
        ]


@pytest.fixture
def arguments_invalid_model(valid_video, invalid_model):
    """Fixture for CLI arguments"""
    return [
        "program_name",
        "--video",
        valid_video,
        "--model",
        invalid_model,
        ]
