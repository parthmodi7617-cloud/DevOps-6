"""
EXP 5: Performance Analysis of CNN
- Studies effect of learning rate on convergence
- Analyses overfitting with increasing epochs
- Plots training vs validation loss curves
- Compares different hyperparameter settings
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ── Load CIFAR-10 (10 classes) ────────────────────────────────────────────────
print("Loading CIFAR-10 ...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, y_train = x_train[:8000], y_train[:8000]
x_test,  y_test  = x_test[:2000],  y_test[:2000]

x_train = x_train / 255.0
x_test  = x_test  / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat  = to_categorical(y_test,  10)

class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# ── Base CNN builder ──────────────────────────────────────────────────────────
def build_cnn(lr=0.001, dropout=0.3):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3), padding='same'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout),
        Dense(10,  activation='softmax'),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A: Effect of Learning Rate
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT A: Learning Rate Effect")
print("=" * 60)

learning_rates  = [0.1, 0.01, 0.001, 0.0001]
lr_histories    = {}
lr_results      = {}

for lr in learning_rates:
    print(f"  Training with lr={lr} ...")
    model = build_cnn(lr=lr)
    hist  = model.fit(x_train, y_train_cat, validation_split=0.1,
                      epochs=15, batch_size=128, verbose=0)
    loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
    lr_histories[lr] = hist
    lr_results[lr]   = (loss, acc)
    print(f"    Accuracy: {acc:.4f}  Loss: {loss:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B: Effect of Number of Epochs (overfitting analysis)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT B: Epochs & Overfitting")
print("=" * 60)

print("  Training for 40 epochs to observe overfitting ...")
overfit_model   = build_cnn(lr=0.001, dropout=0.0)   # No dropout → overfits faster
h_overfit       = overfit_model.fit(x_train, y_train_cat,
                                    validation_split=0.15,
                                    epochs=40, batch_size=128, verbose=0)

print("  Training with Dropout for 40 epochs ...")
regularized_model = build_cnn(lr=0.001, dropout=0.5)
h_reg           = regularized_model.fit(x_train, y_train_cat,
                                        validation_split=0.15,
                                        epochs=40, batch_size=128, verbose=0)

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT C: Hyperparameter Grid
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT C: Hyperparameter Comparison")
print("=" * 60)

configs = {
    "LR=0.001, Drop=0.3": {"lr": 0.001, "dropout": 0.3},
    "LR=0.001, Drop=0.5": {"lr": 0.001, "dropout": 0.5},
    "LR=0.01,  Drop=0.3": {"lr": 0.01,  "dropout": 0.3},
    "LR=0.0001,Drop=0.3": {"lr": 0.0001,"dropout": 0.3},
}
hp_results = {}
for name, cfg in configs.items():
    print(f"  {name} ...")
    m = build_cnn(**cfg)
    m.fit(x_train, y_train_cat, validation_split=0.1,
          epochs=15, batch_size=128, verbose=0)
    _, acc = m.evaluate(x_test, y_test_cat, verbose=0)
    hp_results[name] = acc
    print(f"    Accuracy: {acc:.4f}")

# ── Plotting ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EXP 5 – CNN Performance Analysis", fontsize=15)

# A: Validation accuracy per learning rate
ax = axes[0, 0]
for lr, hist in lr_histories.items():
    ax.plot(hist.history['val_accuracy'], label=f"lr={lr}")
ax.set_title("A: Learning Rate – Val Accuracy")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(True)

# B1: Training vs Validation loss (overfitting)
ax = axes[0, 1]
ax.plot(h_overfit.history['loss'],     label='Train loss (no dropout)', color='blue')
ax.plot(h_overfit.history['val_loss'], label='Val loss  (no dropout)', color='blue', linestyle='--')
ax.plot(h_reg.history['loss'],         label='Train loss (dropout=0.5)', color='orange')
ax.plot(h_reg.history['val_loss'],     label='Val loss  (dropout=0.5)', color='orange', linestyle='--')
ax.set_title("B: Overfitting – Train vs Val Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(fontsize=7); ax.grid(True)

# B2: Train vs Val accuracy gap
ax = axes[1, 0]
ax.plot(h_overfit.history['accuracy'],     label='Train (no dropout)')
ax.plot(h_overfit.history['val_accuracy'], label='Val   (no dropout)')
ax.plot(h_reg.history['accuracy'],         label='Train (dropout=0.5)')
ax.plot(h_reg.history['val_accuracy'],     label='Val   (dropout=0.5)')
ax.set_title("B: Accuracy Gap (Generalization)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(fontsize=7); ax.grid(True)

# C: Hyperparameter bar chart
ax = axes[1, 1]
names = list(hp_results.keys())
accs  = list(hp_results.values())
bars  = ax.barh(names, accs, color='steelblue')
ax.set_title("C: Hyperparameter Accuracy Comparison")
ax.set_xlabel("Test Accuracy")
for bar, acc in zip(bars, accs):
    ax.text(acc + 0.002, bar.get_y() + bar.get_height()/2,
            f"{acc:.3f}", va='center', fontsize=9)
ax.grid(True, axis='x')

plt.tight_layout()
plt.savefig("exp5_cnn_performance.png", dpi=120)
plt.show()

print("\n" + "=" * 60)
print("SUMMARY: Hyperparameter Results")
print("=" * 60)
for name, acc in hp_results.items():
    print(f"  {name:30s} → Accuracy: {acc:.4f}")

print("\nPlot saved: exp5_cnn_performance.png")
