"""
Dataset split script — split labeled data into 80/20 train/val
Usage: python scripts/2_split_dataset.py
"""
import random, shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "dataset"
IMG_DIR = BASE / "images"
LBL_DIR = BASE / "labels"

TRAIN_IMG = BASE / "train" / "images"
TRAIN_LBL = BASE / "train" / "labels"
VAL_IMG = BASE / "val" / "images"
VAL_LBL = BASE / "val" / "labels"

SPLIT_RATIO = 0.8
CLASSES_FILE = BASE / "labels.txt"

# Check
lbl_files = sorted(LBL_DIR.glob("*.txt"))
img_files = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))

print(f"Images: {len(img_files)}")
print(f"Labels: {len(lbl_files)}")

if len(lbl_files) == 0:
    print("ERROR: No label files found!")
    print("  Export YOLO format from makesense.ai to dataset/labels/")
    exit(1)

# Remove unlabeled images automatically
unlabeled = [f.stem for f in img_files if not (LBL_DIR / f"{f.stem}.txt").exists()]
if unlabeled:
    print(f"Removing {len(unlabeled)} unlabeled images: {unlabeled}")
    for u in unlabeled:
        for ext in ['.jpg', '.png']:
            p = IMG_DIR / f"{u}{ext}"
            if p.exists():
                p.unlink()

# Re-read after cleanup
img_files = sorted(IMG_DIR.glob("*.jpg")) + sorted(IMG_DIR.glob("*.png"))
lbl_files = [f for f in img_files if (LBL_DIR / f"{f.stem}.txt").exists()]
random.seed(42)
random.shuffle(lbl_files)

split_idx = int(len(lbl_files) * SPLIT_RATIO)
train_files = lbl_files[:split_idx]
val_files = lbl_files[split_idx:]

# Create dirs
for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
    d.mkdir(parents=True, exist_ok=True)
    for old in d.iterdir():
        old.unlink()

# Copy files
for f in train_files:
    shutil.copy2(f, TRAIN_IMG / f.name)
    lbl = LBL_DIR / f"{f.stem}.txt"
    if lbl.exists():
        shutil.copy2(lbl, TRAIN_LBL / lbl.name)

for f in val_files:
    shutil.copy2(f, VAL_IMG / f.name)
    lbl = LBL_DIR / f"{f.stem}.txt"
    if lbl.exists():
        shutil.copy2(lbl, VAL_LBL / lbl.name)

# Generate data.yaml
classes = CLASSES_FILE.read_text(encoding="utf-8").strip().split("\n")
nc = len(classes)
names_str = "\n".join(f"  {i}: {cn}" for i, cn in enumerate(classes))

yaml_content = f"""# Tea Utensil Detection Dataset
# Auto-generated

train: train/images
val: val/images

nc: {nc}
names:
{names_str}
"""

yaml_path = BASE / "data.yaml"
yaml_path.write_text(yaml_content, encoding="utf-8")

print(f"\nSplit complete!")
print(f"  Train: {len(train_files)} images")
print(f"  Val:   {len(val_files)} images")
print(f"  Classes: {nc}")
print(f"  Config: {yaml_path}")
print(f"\nNext: python scripts/3_train.py")
