"""
EXP 4: CNN vs Transfer Learning
- Trains CNN from scratch on CIFAR-10 (cats vs dogs)
- Compares with MobileNetV2 pretrained model
- Analyses training time, accuracy, dataset size impact
"""

import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, GlobalAveragePooling2D)
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ── Load CIFAR-10 cats vs dogs ────────────────────────────────────────────────
print("Loading data ...")
(x_train_all, y_train_all), (x_test_all, y_test_all) = cifar10.load_data()

def filter_classes(x, y, classes):
    mask = np.isin(y.flatten(), classes)
    x_f  = x[mask]
    y_f  = y[mask].flatten()
    for new_label, orig in enumerate(classes):
        y_f[y_f == orig] = new_label
    return x_f, y_f

x_train_full, y_train_full = filter_classes(x_train_all, y_train_all, [3, 5])
x_test_full,  y_test_full  = filter_classes(x_test_all,  y_test_all,  [3, 5])

# ── Dataset size experiment: small vs full ────────────────────────────────────
DATASET_SIZES = [500, 2000]
EPOCHS = 10

results = {}

for size in DATASET_SIZES:
    x_tr = x_train_full[:size]
    y_tr = y_train_full[:size]
    x_te = x_test_full[:500]
    y_te = y_test_full[:500]

    # Normalise for CNN (32x32)
    x_tr_cnn = x_tr / 255.0
    x_te_cnn = x_te / 255.0

    # Resize for MobileNetV2 (96x96)
    x_tr_mob = tf.image.resize(x_tr, (96, 96)).numpy() / 255.0
    x_te_mob = tf.image.resize(x_te, (96, 96)).numpy() / 255.0

    # ── CNN from scratch ──────────────────────────────────────────────────────
    cnn = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3), padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(128,(3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid'),
    ])
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print(f"\n[Size={size}] Training CNN from scratch ...")
    t0 = time.time()
    h_cnn = cnn.fit(x_tr_cnn, y_tr, validation_split=0.1,
                    epochs=EPOCHS, batch_size=32, verbose=0)
    cnn_time = time.time() - t0
    _, cnn_acc = cnn.evaluate(x_te_cnn, y_te, verbose=0)
    print(f"  CNN Accuracy: {cnn_acc:.4f} | Time: {cnn_time:.1f}s")

    # ── Transfer Learning (MobileNetV2, frozen) ───────────────────────────────
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96,96,3))
    base.trainable = False
    tl_model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid'),
    ])
    tl_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                     loss='binary_crossentropy', metrics=['accuracy'])

    print(f"[Size={size}] Training Transfer Learning model ...")
    t0 = time.time()
    h_tl = tl_model.fit(x_tr_mob, y_tr, validation_split=0.1,
                        epochs=EPOCHS, batch_size=32, verbose=0)
    tl_time = time.time() - t0
    _, tl_acc = tl_model.evaluate(x_te_mob, y_te, verbose=0)
    print(f"  TL  Accuracy: {tl_acc:.4f} | Time: {tl_time:.1f}s")

    results[size] = {
        'cnn_acc': cnn_acc, 'cnn_time': cnn_time, 'h_cnn': h_cnn,
        'tl_acc':  tl_acc,  'tl_time':  tl_time,  'h_tl':  h_tl,
    }

# ── Print Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY: CNN vs Transfer Learning")
print("=" * 60)
print(f"{'Dataset Size':<15} {'CNN Acc':>10} {'TL Acc':>10} {'CNN Time':>12} {'TL Time':>12}")
print("-" * 60)
for size, r in results.items():
    print(f"{size:<15} {r['cnn_acc']:>10.4f} {r['tl_acc']:>10.4f} "
          f"{r['cnn_time']:>10.1f}s {r['tl_time']:>10.1f}s")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EXP 4 – CNN from Scratch vs Transfer Learning", fontsize=14)

for idx, size in enumerate(DATASET_SIZES):
    r = results[size]
    ax = axes[0, idx]
    ax.plot(r['h_cnn'].history['val_accuracy'], label='CNN scratch')
    ax.plot(r['h_tl'].history['val_accuracy'],  label='Transfer Learning')
    ax.set_title(f"Val Accuracy (n={size})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.legend(); ax.grid(True)

# Bar: accuracy vs dataset size
ax = axes[1, 0]
x_pos   = np.arange(len(DATASET_SIZES))
cnn_acc = [results[s]['cnn_acc'] for s in DATASET_SIZES]
tl_acc  = [results[s]['tl_acc']  for s in DATASET_SIZES]
width   = 0.35
ax.bar(x_pos - width/2, cnn_acc, width, label='CNN scratch', color='steelblue')
ax.bar(x_pos + width/2, tl_acc,  width, label='Transfer Learning', color='orange')
ax.set_xticks(x_pos); ax.set_xticklabels([f"n={s}" for s in DATASET_SIZES])
ax.set_title("Test Accuracy vs Dataset Size")
ax.set_ylabel("Accuracy"); ax.legend(); ax.grid(True, axis='y')

# Bar: training time
ax = axes[1, 1]
cnn_times = [results[s]['cnn_time'] for s in DATASET_SIZES]
tl_times  = [results[s]['tl_time']  for s in DATASET_SIZES]
ax.bar(x_pos - width/2, cnn_times, width, label='CNN scratch', color='steelblue')
ax.bar(x_pos + width/2, tl_times,  width, label='Transfer Learning', color='orange')
ax.set_xticks(x_pos); ax.set_xticklabels([f"n={s}" for s in DATASET_SIZES])
ax.set_title("Training Time (seconds)")
ax.set_ylabel("Seconds"); ax.legend(); ax.grid(True, axis='y')

plt.tight_layout()
plt.savefig("exp4_cnn_vs_tl.png", dpi=120)
plt.show()
print("\nPlot saved: exp4_cnn_vs_tl.png")
