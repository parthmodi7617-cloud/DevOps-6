"""
EXP 3: Transfer Learning
- Uses MobileNetV2 pretrained on ImageNet
- Adapts for Cats vs Dogs (or CIFAR-10 subset)
- Compares frozen layers vs partial fine-tuning
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ── Load CIFAR-10 (use 2 classes: cats=3, dogs=5 → binary) ───────────────────
print("Loading CIFAR-10 (cats vs dogs subset) ...")
(x_train_all, y_train_all), (x_test_all, y_test_all) = cifar10.load_data()

# Filter cats (3) and dogs (5)
def filter_classes(x, y, classes):
    mask = np.isin(y.flatten(), classes)
    x_f  = x[mask]
    y_f  = y[mask].flatten()
    # Remap to 0 and 1
    for new_label, orig in enumerate(classes):
        y_f[y_f == orig] = new_label
    return x_f, y_f

x_train, y_train = filter_classes(x_train_all, y_train_all, [3, 5])
x_test,  y_test  = filter_classes(x_test_all,  y_test_all,  [3, 5])

# Subset
x_train, y_train = x_train[:2000], y_train[:2000]
x_test,  y_test  = x_test[:500],   y_test[:500]

# MobileNetV2 needs 96x96 minimum — resize
x_train_resized = tf.image.resize(x_train, (96, 96)).numpy() / 255.0
x_test_resized  = tf.image.resize(x_test,  (96, 96)).numpy() / 255.0

print(f"Train: {x_train_resized.shape}, Test: {x_test_resized.shape}")

EPOCHS = 10

# ── Helper: build transfer learning model ─────────────────────────────────────
def build_transfer_model(trainable_base=False, fine_tune_at=None):
    base_model = MobileNetV2(weights='imagenet',
                             include_top=False,
                             input_shape=(96, 96, 3))

    if not trainable_base:
        base_model.trainable = False          # Frozen (feature extraction)
    elif fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False           # Partial fine-tuning

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1,  activation='sigmoid'),      # Binary: cat or dog
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ── Model A: Fully Frozen (Feature Extraction) ────────────────────────────────
print("\nTraining with FROZEN base (feature extraction) ...")
model_frozen = build_transfer_model(trainable_base=False)
print(f"  Trainable params: {model_frozen.count_params():,}")
h_frozen = model_frozen.fit(x_train_resized, y_train,
                            validation_split=0.1,
                            epochs=EPOCHS, batch_size=32, verbose=1)

# ── Model B: Partial Fine-Tuning (unfreeze last 20 layers) ────────────────────
print("\nTraining with PARTIAL fine-tuning (last 20 layers unfrozen) ...")
model_ft = build_transfer_model(trainable_base=True, fine_tune_at=-20)
print(f"  Trainable params: {model_ft.count_params():,}")
h_ft = model_ft.fit(x_train_resized, y_train,
                    validation_split=0.1,
                    epochs=EPOCHS, batch_size=32, verbose=1)

# ── Evaluate ──────────────────────────────────────────────────────────────────
loss_f, acc_f = model_frozen.evaluate(x_test_resized, y_test, verbose=0)
loss_t, acc_t = model_ft.evaluate(x_test_resized, y_test, verbose=0)

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"  Frozen base (feature extraction) → Accuracy: {acc_f:.4f}, Loss: {loss_f:.4f}")
print(f"  Partial fine-tuning              → Accuracy: {acc_t:.4f}, Loss: {loss_t:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("EXP 3 – Transfer Learning: Frozen vs Fine-Tuned", fontsize=14)

ax = axes[0]
ax.plot(h_frozen.history['val_accuracy'], label='Frozen', linewidth=2)
ax.plot(h_ft.history['val_accuracy'],    label='Fine-Tuned', linewidth=2)
ax.set_title("Validation Accuracy")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(True)

ax = axes[1]
ax.plot(h_frozen.history['val_loss'], label='Frozen', linewidth=2)
ax.plot(h_ft.history['val_loss'],    label='Fine-Tuned', linewidth=2)
ax.set_title("Validation Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig("exp3_transfer_learning.png", dpi=120)
plt.show()
print("\nPlot saved: exp3_transfer_learning.png")
