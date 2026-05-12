"""
EXP 4: CNN vs Transfer Learning using Synthetic Dataset
"""

import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, GlobalAveragePooling2D
)
import matplotlib.pyplot as plt

# Synthetic Dataset
print("Generating Synthetic Dataset...")

x_train_full = np.random.rand(2000, 32, 32, 3)
y_train_full = np.random.randint(0, 2, 2000)

x_test_full = np.random.rand(500, 32, 32, 3)
y_test_full = np.random.randint(0, 2, 500)

DATASET_SIZES = [500, 2000]
EPOCHS = 5

results = {}

for size in DATASET_SIZES:

    x_tr = x_train_full[:size]
    y_tr = y_train_full[:size]

    x_te = x_test_full
    y_te = y_test_full

    # CNN Input
    x_tr_cnn = x_tr
    x_te_cnn = x_te

    # Resize for MobileNetV2
    x_tr_mob = tf.image.resize(x_tr, (96, 96)).numpy()
    x_te_mob = tf.image.resize(x_te, (96, 96)).numpy()

    # CNN from Scratch
    cnn = Sequential([
        Conv2D(32, (3,3), activation='relu',
               input_shape=(32,32,3), padding='same'),

        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu',
               padding='same'),

        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation='relu',
               padding='same'),

        MaxPooling2D((2,2)),

        Flatten(),

        Dense(128, activation='relu'),

        Dropout(0.4),

        Dense(1, activation='sigmoid')
    ])

    cnn.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(f"\nTraining CNN (size={size})...")

    t0 = time.time()

    h_cnn = cnn.fit(
        x_tr_cnn,
        y_tr,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=32,
        verbose=0
    )

    cnn_time = time.time() - t0

    _, cnn_acc = cnn.evaluate(x_te_cnn, y_te, verbose=0)

    print(f"CNN Accuracy: {cnn_acc:.4f}")

    # Transfer Learning
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(96,96,3)
    )

    base.trainable = False

    tl_model = Sequential([
        base,

        GlobalAveragePooling2D(),

        Dense(64, activation='relu'),

        Dropout(0.3),

        Dense(1, activation='sigmoid')
    ])

    tl_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    print(f"Training Transfer Learning (size={size})...")

    t0 = time.time()

    h_tl = tl_model.fit(
        x_tr_mob,
        y_tr,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=32,
        verbose=0
    )

    tl_time = time.time() - t0

    _, tl_acc = tl_model.evaluate(x_te_mob, y_te, verbose=0)

    print(f"TL Accuracy: {tl_acc:.4f}")

    results[size] = {
        'cnn_acc': cnn_acc,
        'cnn_time': cnn_time,
        'tl_acc': tl_acc,
        'tl_time': tl_time,
        'h_cnn': h_cnn,
        'h_tl': h_tl
    }

# Summary
print("\nSUMMARY")
print("=" * 50)

for size, r in results.items():
    print(f"\nDataset Size: {size}")
    print(f"CNN Accuracy: {r['cnn_acc']:.4f}")
    print(f"TL Accuracy : {r['tl_acc']:.4f}")
    print(f"CNN Time    : {r['cnn_time']:.2f}s")
    print(f"TL Time     : {r['tl_time']:.2f}s")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12,5))

# Accuracy Plot
ax = axes[0]

cnn_acc = [results[s]['cnn_acc'] for s in DATASET_SIZES]
tl_acc = [results[s]['tl_acc'] for s in DATASET_SIZES]

x = np.arange(len(DATASET_SIZES))
width = 0.35

ax.bar(x - width/2, cnn_acc, width, label='CNN')
ax.bar(x + width/2, tl_acc, width, label='Transfer Learning')

ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in DATASET_SIZES])

ax.set_title("Accuracy Comparison")
ax.set_ylabel("Accuracy")
ax.legend()

# Time Plot
ax = axes[1]

cnn_times = [results[s]['cnn_time'] for s in DATASET_SIZES]
tl_times = [results[s]['tl_time'] for s in DATASET_SIZES]

ax.bar(x - width/2, cnn_times, width, label='CNN')
ax.bar(x + width/2, tl_times, width, label='Transfer Learning')

ax.set_xticks(x)
ax.set_xticklabels([str(s) for s in DATASET_SIZES])

ax.set_title("Training Time Comparison")
ax.set_ylabel("Seconds")
ax.legend()

plt.tight_layout()

plt.savefig("synthetic_exp4.png", dpi=120)

plt.show()

print("\nPlot saved: synthetic_exp4.png")