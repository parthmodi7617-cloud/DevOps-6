import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# ── Load & Preprocess ─────────────────────────────────────────────
try:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("MNIST dataset loaded successfully.")

except:
    print("MNIST dataset not found. Using synthetic dataset.")

    x_train = np.random.rand(10000, 28, 28)
    y_train = np.random.randint(0, 10, 10000)

    x_test = np.random.rand(2000, 28, 28)
    y_test = np.random.randint(0, 10, 2000)

# Use subset for speed
x_train, y_train = x_train[:10000], y_train[:10000]
x_test, y_test = x_test[:2000], y_test[:2000]

x_train = x_train / 255.0 if x_train.max() > 1 else x_train
x_test = x_test / 255.0 if x_test.max() > 1 else x_test

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# ── Helper Function ─────────────────────────────────────────────
def build_and_train(activation, hidden_layers, epochs=10):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))

    for units in hidden_layers:
        model.add(Dense(units, activation=activation))

    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        x_train, y_train_cat,
        validation_split=0.1,
        epochs=epochs,
        batch_size=128,
        verbose=0
    )

    loss, acc = model.evaluate(x_test, y_test_cat, verbose=0)
    return history, loss, acc

# ── Experiment A ─────────────────────────────────────────────
print("=" * 60)
print("EXPERIMENT A: Effect of Activation Functions")
print("=" * 60)

activations = ['relu', 'sigmoid', 'tanh']
act_histories = {}
act_results = {}

for act in activations:
    print(f"Training with activation: {act} ...")
    hist, loss, acc = build_and_train(act, hidden_layers=[128, 64])
    act_histories[act] = hist
    act_results[act] = (loss, acc)
    print(f"Test Accuracy: {acc:.4f} | Test Loss: {loss:.4f}")

# ── Experiment B ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("EXPERIMENT B: Effect of Number of Hidden Layers")
print("=" * 60)

layer_configs = {
    "1 hidden (128)": [128],
    "2 hidden (128,64)": [128, 64],
    "3 hidden (128,64,32)": [128, 64, 32],
}

layer_histories = {}
layer_results = {}

for name, config in layer_configs.items():
    print(f"Training {name} ...")
    hist, loss, acc = build_and_train('relu', hidden_layers=config)
    layer_histories[name] = hist
    layer_results[name] = (loss, acc)
    print(f"Test Accuracy: {acc:.4f} | Test Loss: {loss:.4f}")

# ── Plot Results ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EXP 1 – MLP Analysis", fontsize=16)

ax = axes[0, 0]
for act, hist in act_histories.items():
    ax.plot(hist.history['val_accuracy'], label=act)
ax.set_title("Activation Functions – Val Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True)

ax = axes[0, 1]
for act, hist in act_histories.items():
    ax.plot(hist.history['val_loss'], label=act)
ax.set_title("Activation Functions – Val Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True)

ax = axes[1, 0]
for name, hist in layer_histories.items():
    ax.plot(hist.history['val_accuracy'], label=name)
ax.set_title("Hidden Layers – Val Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True)

ax = axes[1, 1]
names = list(layer_results.keys())
accs = [layer_results[n][1] for n in names]

bars = ax.bar(names, accs)
ax.set_title("Hidden Layers – Final Test Accuracy")
ax.set_ylabel("Accuracy")

for bar, acc in zip(bars, accs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{acc:.3f}",
        ha='center',
        va='bottom'
    )

ax.tick_params(axis='x', labelrotation=15)
ax.grid(True, axis='y')

plt.tight_layout()
plt.savefig("exp1_mlp_results.png", dpi=120)
plt.show()

print("\nPlot saved as exp1_mlp_results.png")

# ── Summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("\nActivation Function Results:")
for act, (loss, acc) in act_results.items():
    print(f"{act:10s} → Accuracy: {acc:.4f}, Loss: {loss:.4f}")

print("\nHidden Layer Depth Results:")
for name, (loss, acc) in layer_results.items():
    print(f"{name:30s} → Accuracy: {acc:.4f}, Loss: {loss:.4f}")