"""
YOLOv8 Tea Utensil Detection — Training Script

Uses YOLOv8n as base model, trains on 9 tea utensil classes.
Target: lightweight, real-time (>30 FPS on GPU)

Usage: python scripts/3_train.py
"""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "dataset"
DATA_YAML = BASE / "data.yaml"
OUTPUT_DIR = PROJECT / "models"

if not DATA_YAML.exists():
    print(f"ERROR: {DATA_YAML} not found")
    print("  Run scripts/2_split_dataset.py first")
    exit(1)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    from ultralytics import YOLO

    MODEL = PROJECT / "yolov8n.pt"
    EPOCHS = 300
    IMG_SIZE = 1280
    BATCH = 4
    DEVICE = 0
    WORKERS = 0  # Windows: must be 0

    print(f"=== Training YOLOv8 Tea Utensil Detector ===")
    print(f"   Dataset: {DATA_YAML}")
    print(f"   Base model: {MODEL}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Image size: {IMG_SIZE}")
    print(f"   Batch size: {BATCH}")
    print(f"   Workers: {WORKERS}")
    print(f"   Output: {OUTPUT_DIR}")
    print()

    model = YOLO(str(MODEL))

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        patience=20,
        save=True,
        save_period=10,
        project=str(OUTPUT_DIR),
        name="tea_ware_train3",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        cos_lr=True,
        close_mosaic=15,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=5.0,
        translate=0.1,
        scale=0.3,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )

    print("\n=== Training complete! ===")
    best_pt = OUTPUT_DIR / "tea_ware_train3" / "weights" / "best.pt"
    if best_pt.exists():
        print(f"   Best model: {best_pt}")
        print(f"   Model size: {best_pt.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"\nNext: python camera_demo.py")
    else:
        print("\nERROR: No model generated, check training log")
