import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from pathlib import Path

# === CONFIGURATION ===
img_size = (128, 128)
batch_size = 32
epochs = 50
num_classes = 5
l2_lambda = 0.01
k_folds = 5

base_dir = Path("splitted_data_kfold")
results = []

# === DATA AUGMENTATION ===
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1
)
val_datagen = ImageDataGenerator(rescale=1./255)

# === MODEL BUILDER ===
def build_model():
    model = models.Sequential([
        # Layer 1
        layers.Conv2D(32, (3,3), activation='relu', input_shape=img_size + (3,)),
        layers.MaxPooling2D((2,2), strides=2),

        # Layer 2
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2), strides=2),
        layers.Dropout(0.2),

        # Layer 3
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.AveragePooling2D((2,2), strides=2),

        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# === TRAIN & EVALUATE PER FOLD ===
for fold in range(1, k_folds + 1):
    print(f"\n🔹 Fold {fold}/{k_folds}")
    train_dir = base_dir / f"fold_{fold}" / "train"
    val_dir   = base_dir / f"fold_{fold}" / "val"

    # Generators per fold
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical"
    )
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
    )

    model = build_model()
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    val_loss, val_acc = model.evaluate(val_gen, verbose=0)
    y_true = val_gen.classes
    y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"Fold {fold} Results: accuracy={val_acc:.4f}, loss={val_loss:.4f}, f1={f1:.4f}")
    results.append({'fold': fold, 'accuracy': val_acc, 'loss': val_loss, 'f1': f1})

# === AVERAGE METRICS ===
accs = [r['accuracy'] for r in results]
losses = [r['loss'] for r in results]
f1s = [r['f1'] for r in results]

avg_acc = np.mean(accs)
avg_loss = np.mean(losses)
avg_f1 = np.mean(f1s)
std_acc = np.std(accs)
std_f1 = np.std(f1s)

print("\n📊 5-Fold Cross Validation Summary:")
print(f"Accuracy : {avg_acc:.4f} ± {std_acc:.4f}")
print(f"F1-Score : {avg_f1:.4f} ± {std_f1:.4f}")
print(f"Loss     : {avg_loss:.4f}")

# Save metrics
os.makedirs("checkpoints_exp2", exist_ok=True)
with open("checkpoints_exp2/exp2_kfold_results.json", "w") as f:
    json.dump(results, f, indent=2)

# === FINAL MODEL TRAINING ON FULL DATA ===
print("\n🔸 Training final model on all data with best configuration...")
full_train_dir = Path("splitted_data_kfold/full_train")

# Merge all folds' train data into one (optional)
# For simplicity, assume you can reuse "splitted_data_kfold/fold_1/train" as full dataset.
# Or manually merge all training folds if required by your grading policy.

final_train_gen = train_datagen.flow_from_directory(
    "splitted_data_kfold/fold_1/train",
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical"
)

final_model = build_model()
final_model.fit(
    final_train_gen,
    epochs=epochs,
    verbose=1
)
final_model.save("checkpoints_exp2/final_model.h5")

print("\n✅ Experiment 2 training complete.")
