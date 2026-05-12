import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Synthetic Dataset
x_train = np.random.rand(10000, 28, 28)
y_train = np.random.randint(0, 10, 10000)

x_test = np.random.rand(2000, 28, 28)
y_test = np.random.randint(0, 10, 2000)

# CNN needs channel dimension
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# MLP needs flat input
x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

class_names = [
    'Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4',
    'Class 5', 'Class 6', 'Class 7', 'Class 8', 'Class 9'
]

EPOCHS = 10

# MLP Model
def build_mlp():
    model = Sequential([
        Dense(256, activation='relu', input_shape=(784,)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# CNN with MaxPooling
def build_cnn_max():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# CNN with AveragePooling
def build_cnn_avg():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        AveragePooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        AveragePooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# CNN with 5x5 Kernel
def build_cnn_5x5():
    model = Sequential([
        Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1), padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (5, 5), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Train Models
print("Training MLP...")
mlp = build_mlp()
h_mlp = mlp.fit(
    x_train_flat, y_train_cat,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=128,
    verbose=0
)

print("Training CNN MaxPool...")
cnn_max = build_cnn_max()
h_cnn_max = cnn_max.fit(
    x_train_cnn, y_train_cat,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=128,
    verbose=0
)

print("Training CNN AvgPool...")
cnn_avg = build_cnn_avg()
h_cnn_avg = cnn_avg.fit(
    x_train_cnn, y_train_cat,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=128,
    verbose=0
)

print("Training CNN 5x5...")
cnn_5x5 = build_cnn_5x5()
h_cnn_5x5 = cnn_5x5.fit(
    x_train_cnn, y_train_cat,
    validation_split=0.1,
    epochs=EPOCHS,
    batch_size=128,
    verbose=0
)

# Evaluate
results = {
    "MLP": mlp.evaluate(x_test_flat, y_test_cat, verbose=0),
    "CNN MaxPool": cnn_max.evaluate(x_test_cnn, y_test_cat, verbose=0),
    "CNN AvgPool": cnn_avg.evaluate(x_test_cnn, y_test_cat, verbose=0),
    "CNN 5x5": cnn_5x5.evaluate(x_test_cnn, y_test_cat, verbose=0)
}

print("\nRESULTS")
for name, (loss, acc) in results.items():
    print(f"{name:15s} → Accuracy: {acc:.4f}, Loss: {loss:.4f}")

# Feature Maps
sample_img = x_test_cnn[0:1]

feature_model = Model(
    inputs=cnn_max.input,
    outputs=cnn_max.layers[0].output
)

feature_maps = feature_model.predict(sample_img, verbose=0)

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle("Feature Maps from Conv Layer 1")

for i in range(32):
    ax = axes[i // 8][i % 8]
    ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
    ax.axis('off')
    ax.set_title(f"F{i+1}", fontsize=7)

plt.tight_layout()
plt.savefig("synthetic_feature_maps.png", dpi=120)
plt.show()

# Accuracy Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("CNN vs MLP on Synthetic Dataset")

histories = {
    "MLP": h_mlp,
    "CNN MaxPool": h_cnn_max,
    "CNN AvgPool": h_cnn_avg,
    "CNN 5x5": h_cnn_5x5
}

ax = axes[0]
for name, h in histories.items():
    ax.plot(h.history['val_accuracy'], label=name)

ax.set_title("Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True)

ax = axes[1]
names = list(results.keys())
accs = [results[n][1] for n in names]

bars = ax.bar(names, accs)
ax.set_title("Final Test Accuracy")
ax.set_ylabel("Accuracy")

for bar, acc in zip(bars, accs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{acc:.3f}",
        ha='center',
        va='bottom'
    )

ax.grid(True, axis='y')

plt.tight_layout()
plt.savefig("synthetic_cnn_comparison.png", dpi=120)
plt.show()

print("\nPlots saved: synthetic_feature_maps.png, synthetic_cnn_comparison.png")