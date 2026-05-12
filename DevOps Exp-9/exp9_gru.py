"""
EXP 9: GRU (Gated Recurrent Unit)
- Faster training than LSTM
- Compares GRU vs LSTM vs SimpleRNN
- Analyses efficiency and accuracy trade-offs
- Evaluates suitability for real-time applications
"""

import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GRU,
    LSTM,
    SimpleRNN,
    Dense,
    Embedding,
)
import matplotlib.pyplot as plt

# ── Reduce TensorFlow Logs ────────────────────────────────────────────────────
tf.get_logger().setLevel("ERROR")

# ── Text Data ─────────────────────────────────────────────────────────────────
TEXT = """
Deep learning is a subset of machine learning which uses neural networks.
Neural networks are inspired by the structure of the human brain and neurons.
Recurrent neural networks are designed to work with sequential data inputs.
Gated recurrent units are a simplified version of long short term memory.
Long short term memory networks solve the vanishing gradient problem well.
Simple recurrent networks struggle to learn long range dependencies easily.
Training deep neural networks requires large datasets and computing power.
Gradient descent is the optimization algorithm used for learning weights.
Backpropagation through time is used to train recurrent neural networks now.
The gating mechanism in GRU uses reset and update gates for information flow.
LSTM uses three gates namely input forget and output for memory management.
GRU achieves comparable performance to LSTM with fewer parameters overall.
Real time applications benefit from GRU due to its faster inference speed.
""" * 6

TEXT = TEXT.strip()

# ── Character Vocabulary ──────────────────────────────────────────────────────
chars = sorted(set(TEXT))
n_chars = len(chars)

char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}

print(f"Text length : {len(TEXT)}")
print(f"Vocab size  : {n_chars}")

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEQ_LEN = 30
EPOCHS = 20
UNITS = 64
EMBED_DIM = 16
BATCH_SIZE = 128

# ── Prepare Data ──────────────────────────────────────────────────────────────
encoded = np.array([char2idx[c] for c in TEXT])

X = []
y = []

for i in range(len(encoded) - SEQ_LEN):
    X.append(encoded[i:i + SEQ_LEN])
    y.append(encoded[i + SEQ_LEN])

X = np.array(X)
y = np.array(y)

y_oh = tf.keras.utils.to_categorical(y, num_classes=n_chars)

print(f"Samples     : {X.shape}")

# ── Model Builders ────────────────────────────────────────────────────────────
def build_simple_rnn():

    model = Sequential([
        Embedding(input_dim=n_chars, output_dim=EMBED_DIM),
        SimpleRNN(UNITS),
        Dense(n_chars, activation='softmax')
    ])

    model.build(input_shape=(None, SEQ_LEN))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_lstm():

    model = Sequential([
        Embedding(input_dim=n_chars, output_dim=EMBED_DIM),
        LSTM(UNITS, dropout=0.2),
        Dense(n_chars, activation='softmax')
    ])

    model.build(input_shape=(None, SEQ_LEN))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_gru():

    model = Sequential([
        Embedding(input_dim=n_chars, output_dim=EMBED_DIM),
        GRU(UNITS, dropout=0.2),
        Dense(n_chars, activation='softmax')
    ])

    model.build(input_shape=(None, SEQ_LEN))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# ── Count Parameters ──────────────────────────────────────────────────────────
rnn_model = build_simple_rnn()
lstm_model = build_lstm()
gru_model = build_gru()

rnn_params = rnn_model.count_params()
lstm_params = lstm_model.count_params()
gru_params = gru_model.count_params()

print("\nModel Parameters:")
print(f"  SimpleRNN : {rnn_params:,}")
print(f"  LSTM      : {lstm_params:,}")
print(f"  GRU       : {gru_params:,}")

# ── Train Models ──────────────────────────────────────────────────────────────
results = {}

models = [
    ("SimpleRNN", build_simple_rnn),
    ("LSTM", build_lstm),
    ("GRU", build_gru)
]

for name, build_fn in models:

    print("\n" + "=" * 60)
    print(f"Training {name}")
    print("=" * 60)

    model = build_fn()

    start_time = time.time()

    history = model.fit(
        X,
        y_oh,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1
    )

    elapsed = time.time() - start_time

    loss, train_acc = model.evaluate(X, y_oh, verbose=0)

    results[name] = {
        "model": model,
        "history": history,
        "time": elapsed,
        "accuracy": train_acc
    }

    print(f"\n{name} Results")
    print(f"Accuracy : {train_acc:.4f}")
    print(f"Time     : {elapsed:.2f} seconds")

# ── Text Generation ───────────────────────────────────────────────────────────
def generate_text(model, seed_text, n_chars_generate=100, temperature=0.8):

    seed_encoded = [char2idx[c] for c in seed_text[-SEQ_LEN:]]

    generated = seed_text

    for _ in range(n_chars_generate):

        x = np.array([seed_encoded])

        prediction = model.predict(x, verbose=0)[0]

        prediction = prediction.astype("float64")

        prediction = np.log(prediction + 1e-8) / temperature

        exp_preds = np.exp(prediction)

        prediction = exp_preds / np.sum(exp_preds)

        next_index = np.random.choice(len(prediction), p=prediction)

        next_char = idx2char[next_index]

        generated += next_char

        seed_encoded = seed_encoded[1:] + [next_index]

    return generated

# ── Generate Sample Text ──────────────────────────────────────────────────────
seed = TEXT[:SEQ_LEN]

print("\n" + "=" * 60)
print("TEXT GENERATION COMPARISON")
print("=" * 60)

for name, res in results.items():

    generated_text = generate_text(
        res["model"],
        seed
    )

    print(f"\n{name}:")
    print("-" * 60)
    print(generated_text)

# ── Summary Table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(f"{'Model':<12} {'Accuracy':>12} {'Time(s)':>12} {'Parameters':>15}")
print("-" * 60)

param_dict = {
    "SimpleRNN": rnn_params,
    "LSTM": lstm_params,
    "GRU": gru_params
}

for name, res in results.items():

    print(
        f"{name:<12} "
        f"{res['accuracy']:>12.4f} "
        f"{res['time']:>12.2f} "
        f"{param_dict[name]:>15,}"
    )

# ── Plot Results ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

fig.suptitle(
    "EXP 9 – GRU vs LSTM vs SimpleRNN",
    fontsize=16
)

# Validation Loss
ax = axes[0, 0]

for name, res in results.items():
    ax.plot(
        res["history"].history["val_loss"],
        label=name
    )

ax.set_title("Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(True)

# Validation Accuracy
ax = axes[0, 1]

for name, res in results.items():
    ax.plot(
        res["history"].history["val_accuracy"],
        label=name
    )

ax.set_title("Validation Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.legend()
ax.grid(True)

# Training Time
ax = axes[1, 0]

model_names = list(results.keys())

times = [results[m]["time"] for m in model_names]

bars = ax.bar(model_names, times)

for bar, t in zip(bars, times):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{t:.1f}s",
        ha='center'
    )

ax.set_title("Training Time")
ax.set_ylabel("Seconds")
ax.grid(True, axis='y')

# Parameter Count
ax = axes[1, 1]

params = [param_dict[m] for m in model_names]

bars = ax.bar(model_names, params)

for bar, p in zip(bars, params):

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{p:,}",
        ha='center'
    )

ax.set_title("Parameter Count")
ax.set_ylabel("Parameters")
ax.grid(True, axis='y')

plt.tight_layout()

plt.savefig(
    "exp9_gru_results.png",
    dpi=120
)

print("\nPlot saved as: exp9_gru_results.png")

plt.show()
