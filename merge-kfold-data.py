import os
import shutil
from pathlib import Path
from tqdm import tqdm

"""
Script to merge all K-fold training data into a single 'full_train' directory.
This is needed for training the final model on all available training data.
"""

# === CONFIGURATION ===
base_dir = Path("splitted_data_kfold")
output_dir = base_dir / "full_train"
k_folds = 5

# Class names (adjust if needed)
classes = ['American_Bulldog', 'German_Shorthaired', 'Havanese', 'Maine_Coon', 'Pomeranian']

print("="*60)
print("🔄 MERGING K-FOLD TRAINING DATA")
print("="*60)

# Create output directory structure
print(f"\n📁 Creating output directory: {output_dir}")
output_dir.mkdir(exist_ok=True)

for class_name in classes:
    class_dir = output_dir / class_name
    class_dir.mkdir(exist_ok=True)
    print(f"   ✓ Created: {class_dir}")

# Copy files from each fold
print(f"\n📦 Copying training data from {k_folds} folds...")

copied_files = {class_name: 0 for class_name in classes}

for fold in range(1, k_folds + 1):
    print(f"\n🔹 Processing Fold {fold}/{k_folds}")
    train_dir = base_dir / f"fold_{fold}" / "train"
    
    if not train_dir.exists():
        print(f"   ⚠️ Warning: {train_dir} does not exist! Skipping...")
        continue
    
    for class_name in classes:
        source_class_dir = train_dir / class_name
        dest_class_dir = output_dir / class_name
        
        if not source_class_dir.exists():
            print(f"   ⚠️ Warning: {source_class_dir} does not exist! Skipping...")
            continue
        
        # Get all image files
        image_files = list(source_class_dir.glob("*.jpg")) + \
                     list(source_class_dir.glob("*.jpeg")) + \
                     list(source_class_dir.glob("*.png"))
        
        # Copy each file with fold prefix to avoid name conflicts
        for img_file in image_files:
            # Add fold number prefix to avoid overwriting duplicates
            new_filename = f"fold{fold}_{img_file.name}"
            dest_file = dest_class_dir / new_filename
            
            shutil.copy2(img_file, dest_file)
            copied_files[class_name] += 1
        
        print(f"   {class_name}: copied {len(image_files)} images")

# Summary
print("\n" + "="*60)
print("✅ MERGE COMPLETE!")
print("="*60)

print(f"\n📊 Summary:")
print(f"   Output directory: {output_dir}")
print(f"\n   Files per class:")

total_files = 0
for class_name in classes:
    count = copied_files[class_name]
    total_files += count
    print(f"      {class_name}: {count} images")

print(f"\n   Total images: {total_files}")

# Verify the merge
print(f"\n🔍 Verification:")
for class_name in classes:
    class_dir = output_dir / class_name
    actual_count = len(list(class_dir.glob("*.*")))
    expected_count = copied_files[class_name]
    
    if actual_count == expected_count:
        print(f"   ✓ {class_name}: {actual_count} files (OK)")
    else:
        print(f"   ✗ {class_name}: Expected {expected_count}, found {actual_count} (MISMATCH!)")

print(f"\n💡 You can now use '{output_dir}' for training the final model!")
print("="*60)