"""
EXP 3: Transfer Learning using Synthetic Dataset
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
import matplotlib.pyplot as plt

# Synthetic dataset: binary classification
# 0 = Cat, 1 = Dog
x_train = np.random.rand(500, 96, 96, 3)
y_train = np.random.randint(0, 2, 500)

x_test = np.random.rand(100, 96, 96, 3)
y_test = np.random.randint(0, 2, 100)

EPOCHS = 5

def build_transfer_model(trainable_base=False, fine_tune_at=None):
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(96, 96, 3)
    )

    if not trainable_base:
        base_model.trainable = False
    elif fine_tune_at is not None:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# Frozen base model
print("Training Frozen Base Model...")
model_frozen = build_transfer_model(trainable_base=False)

h_frozen = model_frozen.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=32,
    verbose=1
)

# Partial fine-tuning model
print("\nTraining Partial Fine-Tuning Model...")
model_ft = build_transfer_model(trainable_base=True, fine_tune_at=-20)

h_ft = model_ft.fit(
    x_train, y_train,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=32,
    verbose=1
)

# Evaluation
loss_f, acc_f = model_frozen.evaluate(x_test, y_test, verbose=0)
loss_t, acc_t = model_ft.evaluate(x_test, y_test, verbose=0)

print("\nRESULTS")
print(f"Frozen Base Accuracy: {acc_f:.4f}, Loss: {loss_f:.4f}")
print(f"Fine-Tuned Accuracy: {acc_t:.4f}, Loss: {loss_t:.4f}")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(h_frozen.history['val_accuracy'], label='Frozen')
plt.plot(h_ft.history['val_accuracy'], label='Fine-Tuned')
plt.title("Transfer Learning on Synthetic Dataset")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("synthetic_transfer_learning.png", dpi=120)
plt.show()

print("Plot saved: synthetic_transfer_learning.png")