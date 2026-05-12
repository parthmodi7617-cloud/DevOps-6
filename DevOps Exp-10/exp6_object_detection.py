import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Synthetic Dataset
x_train = np.random.rand(1000, 96, 96, 3)
y_train = np.random.randint(0, 10, 1000)

x_test = np.random.rand(200, 96, 96, 3)
y_test = np.random.randint(0, 10, 200)

y_train = to_categorical(y_train, 10)

class_names = [
    'class0','class1','class2','class3','class4',
    'class5','class6','class7','class8','class9'
]

# MobileNetV2 Model
base = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(96,96,3)
)

base.trainable = False

model = Sequential([
    base,
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
print("Training Model...")

model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=32,
    verbose=1
)

# Predict
samples = x_test[:5]

preds = model.predict(samples)

# Show Results
fig, axes = plt.subplots(1, 5, figsize=(15,4))

for i in range(5):

    ax = axes[i]

    ax.imshow(samples[i])

    pred = np.argmax(preds[i])

    conf = np.max(preds[i])

    ax.set_title(f"{class_names[pred]}\n{conf:.2f}")

    ax.axis('off')

plt.tight_layout()

plt.show()

# Noise Analysis
noise_levels = [0, 0.1, 0.2, 0.3]

print("\nNoise Analysis")

for noise in noise_levels:

    noisy = np.clip(
        samples[0:1] + np.random.normal(0, noise, samples[0:1].shape),
        0,
        1
    )

    pred = model.predict(noisy, verbose=0)

    conf = np.max(pred)

    print(f"Noise={noise}  Confidence={conf:.4f}")