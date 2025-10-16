import os
import shutil
import random
from pathlib import Path

# === CONFIGURATION ===
base_dir = Path("images")
output_dir = Path("splitted_data")

prefixes = [
    "maine_coon_",
    "pomeranian_",
    "american_bulldog_",
    "havanese_",
    "german_shorthaired_"
]

split_ratio = 0.8  # 80% train, 20% val

# === PREPARE OUTPUT FOLDERS ===
(output_dir / "train").mkdir(parents=True, exist_ok=True)
(output_dir / "val").mkdir(parents=True, exist_ok=True)

random.seed(42)

total_train, total_val = 0, 0

for prefix in prefixes:
    breed_imgs = list(base_dir.glob(f"{prefix}*.jpg"))
    if not breed_imgs:
        print(f"[WARN] No images found for prefix: {prefix}")
        continue
    random.seed(42)
    random.shuffle(breed_imgs)
    split_idx = int(len(breed_imgs) * split_ratio)
    train_imgs = breed_imgs[:split_idx]
    val_imgs = breed_imgs[split_idx:]

    # Copy images into the global train/val folders
    for img_path in train_imgs:
        shutil.copy(img_path, output_dir / "train" / img_path.name)
    for img_path in val_imgs:
        shutil.copy(img_path, output_dir / "val" / img_path.name)

    total_train += len(train_imgs)
    total_val += len(val_imgs)

    print(f"{prefix:<25} → {len(train_imgs)} train, {len(val_imgs)} val")

print(f"\n✅ Done! Total: {total_train} train, {total_val} val")