from ultralytics import YOLO
import pandas as pd
from pathlib import Path

def track_bees_in_video(model_path, video_path):
    """Detect and track bees and save both video and CSV to same directory."""
    model = YOLO(model_path)
    rows = []
    
    # Run tracking - YOLO handles the output directory
    results = model.track(
        source=video_path, 
        tracker="bytetrack.yaml",
        save=True,
        project="results",
        name="bee_tracking",
        stream=True
    )
    
    # Get output directory from the predictor (much cleaner!)
    output_dir = Path(model.predictor.save_dir)
    
    # Extract tracking data
    for frame_idx, r in enumerate(results):
        if r.boxes is None or len(r.boxes) == 0:
            continue
            
        xyxy = r.boxes.xyxy.tolist()
        conf = r.boxes.conf.tolist() if r.boxes.conf is not None else [None] * len(xyxy)
        cls = r.boxes.cls.int().tolist() if r.boxes.cls is not None else [-1] * len(xyxy)
        ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [-1] * len(xyxy)
        
        for (x1, y1, x2, y2), c, k, tid in zip(xyxy, conf, cls, ids):
            rows.append({
                "frame": frame_idx, "track_id": int(tid), "class_id": int(k),
                "confidence": float(c), "x1": float(x1), "y1": float(y1), 
                "x2": float(x2), "y2": float(y2)
            })
    
    # Save CSV to same directory as video
    df = pd.DataFrame(rows)
    csv_path = output_dir / "detections.csv"
    df.to_csv(csv_path, index=False)
    
    print(f"Tracking complete!")
    print(f"- Total detections: {len(df)}")
    print(f"- Unique tracks: {df['track_id'].nunique()}")
    print(f"- Results saved to: {output_dir}")
    
    return df

if __name__ == "__main__":
    df = track_bees_in_video("../models/rachel_YOLOv8nano_feb25.pt", "../data/Test2_20sec.MP4")