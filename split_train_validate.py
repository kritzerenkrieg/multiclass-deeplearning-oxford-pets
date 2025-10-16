import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import KFold

# === CONFIGURATION ===
base_dir = Path("images")
output_dir = Path("splitted_data_kfold")

breed_map = {
    "maine_coon_": "Maine_Coon",
    "pomeranian_": "Pomeranian",
    "american_bulldog_": "American_Bulldog",
    "havanese_": "Havanese",
    "german_shorthaired_": "German_Shorthaired"
}

k_folds = 5  # number of folds
random.seed(42)

# === CLEAN OUTPUT DIR (optional) ===
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# === PREPARE FOLD STRUCTURE ===
for i in range(1, k_folds + 1):
    for split in ["train", "val"]:
        for breed_name in breed_map.values():
            (output_dir / f"fold_{i}" / split / breed_name).mkdir(parents=True, exist_ok=True)

# === SPLIT EACH BREED INTO K FOLDS ===
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for prefix, folder_name in breed_map.items():
    breed_imgs = sorted(list(base_dir.glob(f"{prefix}*.jpg")))
    if not breed_imgs:
        print(f"[WARN] No images found for prefix: {prefix}")
        continue

    breed_imgs = list(breed_imgs)
    img_indices = list(range(len(breed_imgs)))

    # Generate folds
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(img_indices), 1):
        train_imgs = [breed_imgs[i] for i in train_idx]
        val_imgs = [breed_imgs[i] for i in val_idx]

        for img_path in train_imgs:
            shutil.copy(img_path, output_dir / f"fold_{fold_idx}" / "train" / folder_name / img_path.name)
        for img_path in val_imgs:
            shutil.copy(img_path, output_dir / f"fold_{fold_idx}" / "val" / folder_name / img_path.name)

    print(f"{folder_name:<25} → {len(breed_imgs)} total → {k_folds}-fold split complete")

print(f"\n✅ All {k_folds} folds created successfully!")
print(f"Location: {output_dir.resolve()}")
