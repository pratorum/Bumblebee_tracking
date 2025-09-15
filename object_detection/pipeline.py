from ultralytics import YOLO

# Load your trained model
model = YOLO("models/yolo/yolov8n-bees.pt")

# Run inference on a video
results = model.track(
    source="C:/Users/you/Downloads/bee_video.mp4",  # path to your video
    tracker="bytetrack.yaml",  # enables tracking
    show=True,   # display live
    save=True    # save output video under runs/track/
)
