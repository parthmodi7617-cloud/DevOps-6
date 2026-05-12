"""
EXP 8: LSTM (Long Short-Term Memory)
- Handles long-term dependencies in text data
- Compares with Simple RNN
- Analyses loss convergence and prediction quality
- Shows how memory mechanisms help
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense, Embedding, Dropout
import matplotlib.pyplot as plt

# ── Text Data ─────────────────────────────────────────────────────────────────
TEXT = """
The quick brown fox jumps over the lazy dog.
A journey of a thousand miles begins with a single step.
To be or not to be that is the question whether tis nobler.
All that glitters is not gold often have you heard that told.
The early bird catches the worm while the night owl sees the moon.
Actions speak louder than words in every situation we face.
Knowledge is power and power brings responsibility to all.
Time flies like an arrow fruit flies like a banana in summer.
In the beginning was the word and the word was with god above.
Every cloud has a silver lining patience is a virtue to keep.
The pen is mightier than the sword in every battle ever fought.
Better late than never but better never late than always early.
Ask not what your country can do for you but what you can do.
""" * 6

TEXT = TEXT.strip()
print(f"Text length: {len(TEXT)} characters")

# ── Character vocabulary ──────────────────────────────────────────────────────
chars    = sorted(set(TEXT))
n_chars  = len(chars)
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
print(f"Vocabulary size: {n_chars}")

SEQ_LEN = 25   # Fixed sequence length for fair comparison
EPOCHS  = 25
UNITS   = 128

# ── Build sequences ───────────────────────────────────────────────────────────
encoded = np.array([char2idx[c] for c in TEXT])
X, y   = [], []
for i in range(len(encoded) - SEQ_LEN):
    X.append(encoded[i: i + SEQ_LEN])
    y.append(encoded[i + SEQ_LEN])
X = np.array(X)
y_oh = tf.keras.utils.to_categorical(y, num_classes=n_chars)
print(f"Training samples: {X.shape}")

# ── Model A: Simple RNN ───────────────────────────────────────────────────────
def build_simple_rnn():
    model = Sequential([
        Embedding(n_chars, 32, input_length=SEQ_LEN),
        SimpleRNN(UNITS, return_sequences=True),
        SimpleRNN(UNITS),
        Dense(n_chars, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ── Model B: LSTM ─────────────────────────────────────────────────────────────
def build_lstm():
    model = Sequential([
        Embedding(n_chars, 32, input_length=SEQ_LEN),
        LSTM(UNITS, return_sequences=True),
        Dropout(0.2),
        LSTM(UNITS),
        Dropout(0.2),
        Dense(n_chars, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ── Model C: LSTM – longer sequence (long-term dependency test) ───────────────
SEQ_LONG = 50
X_long, y_long = [], []
for i in range(len(encoded) - SEQ_LONG):
    X_long.append(encoded[i: i + SEQ_LONG])
    y_long.append(encoded[i + SEQ_LONG])
X_long = np.array(X_long)
y_long_oh = tf.keras.utils.to_categorical(y_long, num_classes=n_chars)

def build_lstm_long():
    model = Sequential([
        Embedding(n_chars, 32, input_length=SEQ_LONG),
        LSTM(UNITS, return_sequences=True),
        Dropout(0.2),
        LSTM(UNITS),
        Dense(n_chars, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ── Train ──────────────────────────────────────────────────────────────────────
print("\nTraining Simple RNN ...")
rnn_model = build_simple_rnn()
h_rnn = rnn_model.fit(X, y_oh, epochs=EPOCHS, batch_size=128,
                       validation_split=0.1, verbose=1)

print("\nTraining LSTM (short seq) ...")
lstm_model = build_lstm()
h_lstm = lstm_model.fit(X, y_oh, epochs=EPOCHS, batch_size=128,
                         validation_split=0.1, verbose=1)

print("\nTraining LSTM (long seq, len=50) ...")
lstm_long_model = build_lstm_long()
h_lstm_long = lstm_long_model.fit(X_long, y_long_oh, epochs=EPOCHS, batch_size=128,
                                   validation_split=0.1, verbose=1)

# ── Text Generation ───────────────────────────────────────────────────────────
def generate(model, seed_chars, seq_len, n=100):
    seed_enc = [char2idx[c] for c in seed_chars[-seq_len:]]
    out = seed_chars
    for _ in range(n):
        x    = np.array([seed_enc])
        pred = model.predict(x, verbose=0)[0]
        # temperature sampling
        pred = np.log(pred + 1e-8) / 0.8
        pred = np.exp(pred) / np.sum(np.exp(pred))
        nxt  = np.random.choice(len(pred), p=pred)
        out += idx2char[nxt]
        seed_enc = seed_enc[1:] + [nxt]
    return out

seed = TEXT[:SEQ_LEN]
print("\n" + "=" * 60)
print("TEXT GENERATION SAMPLES")
print("=" * 60)
print(f"\nSeed: {seed!r}")
print(f"\nSimple RNN output:\n  {generate(rnn_model, seed, SEQ_LEN)}")
print(f"\nLSTM output:\n  {generate(lstm_model, seed, SEQ_LEN)}")

# ── Evaluation ────────────────────────────────────────────────────────────────
_, rnn_acc  = rnn_model.evaluate(X, y_oh, verbose=0)
_, lstm_acc = lstm_model.evaluate(X, y_oh, verbose=0)
_, lstm_long_acc = lstm_long_model.evaluate(X_long, y_long_oh, verbose=0)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  Simple RNN (seq={SEQ_LEN})  → Accuracy: {rnn_acc:.4f}")
print(f"  LSTM       (seq={SEQ_LEN})  → Accuracy: {lstm_acc:.4f}")
print(f"  LSTM       (seq={SEQ_LONG}) → Accuracy: {lstm_long_acc:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EXP 8 – LSTM vs Simple RNN", fontsize=14)

# Loss comparison
ax = axes[0, 0]
ax.plot(h_rnn.history['loss'],  label='RNN  – Train')
ax.plot(h_rnn.history['val_loss'], label='RNN  – Val', linestyle='--')
ax.plot(h_lstm.history['loss'], label='LSTM – Train')
ax.plot(h_lstm.history['val_loss'], label='LSTM – Val', linestyle='--')
ax.set_title("Training & Val Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True)

# Accuracy comparison
ax = axes[0, 1]
ax.plot(h_rnn.history['accuracy'],      label='RNN  – Train')
ax.plot(h_rnn.history['val_accuracy'],  label='RNN  – Val', linestyle='--')
ax.plot(h_lstm.history['accuracy'],     label='LSTM – Train')
ax.plot(h_lstm.history['val_accuracy'], label='LSTM – Val', linestyle='--')
ax.set_title("Training & Val Accuracy")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(True)

# LSTM short vs long
ax = axes[1, 0]
ax.plot(h_lstm.history['val_loss'],      label=f'LSTM seq={SEQ_LEN}')
ax.plot(h_lstm_long.history['val_loss'], label=f'LSTM seq={SEQ_LONG}')
ax.set_title("LSTM: Short vs Long Sequence Val Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True)

# Bar chart: final accuracy
ax = axes[1, 1]
labels = ['Simple RNN', f'LSTM (seq={SEQ_LEN})', f'LSTM (seq={SEQ_LONG})']
accs   = [rnn_acc, lstm_acc, lstm_long_acc]
bars   = ax.bar(labels, accs, color=['steelblue', 'orange', 'green'])
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{acc:.3f}", ha='center', fontsize=10)
ax.set_title("Final Training Accuracy Comparison")
ax.set_ylabel("Accuracy"); ax.grid(True, axis='y')
ax.tick_params(axis='x', labelrotation=10)

plt.tight_layout()
plt.savefig("exp8_lstm_results.png", dpi=120)
plt.show()
print("\nPlot saved: exp8_lstm_results.png")
