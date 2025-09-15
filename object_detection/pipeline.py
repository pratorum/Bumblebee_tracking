from ultralytics import YOLO

# Load your trained model
model = YOLO("../models/rachel_YOLOv8nano_feb25.pt")

# Run inference on a video
results = model.track(
    source="../data/Test2_20sec.MP4",  # path to your video
    tracker="bytetrack.yaml",  # enables tracking
    show=False,   # display live
    save=True,
    project="results",
    name="bee_tracking"    # save output video under runs/track/
)

print("Tracking complete. Results saved in the 'results/bee_tracking' directory.")