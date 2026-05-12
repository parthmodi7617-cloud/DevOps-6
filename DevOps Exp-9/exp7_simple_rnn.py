"""
EXP 7: Simple RNN
- Next character/word prediction on text data
- Analyses effect of sequence length
- Studies limitations with long-term dependencies
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding
import matplotlib.pyplot as plt

# ── Text Data ─────────────────────────────────────────────────────────────────
TEXT = """
The quick brown fox jumps over the lazy dog.
A journey of a thousand miles begins with a single step.
To be or not to be that is the question.
All that glitters is not gold.
The early bird catches the worm.
Actions speak louder than words.
Knowledge is power and power is responsibility.
Time flies like an arrow and fruit flies like a banana.
In the beginning was the word and the word was with god.
Every cloud has a silver lining and patience is a virtue.
The pen is mightier than the sword in every situation.
Better late than never but better never late.
Ask not what your country can do for you.
""" * 5   # repeat to get ~1000+ chars

TEXT = TEXT.strip()
print(f"Text length: {len(TEXT)} characters")

# ── Character-level vocabulary ────────────────────────────────────────────────
chars   = sorted(set(TEXT))
n_chars = len(chars)
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}
print(f"Vocabulary size: {n_chars} characters")

# ── Create sequences ──────────────────────────────────────────────────────────
def make_sequences(text, seq_len):
    X, y = [], []
    encoded = [char2idx[c] for c in text]
    for i in range(len(encoded) - seq_len):
        X.append(encoded[i: i + seq_len])
        y.append(encoded[i + seq_len])
    X = np.array(X)
    y = np.array(y)
    # One-hot encode y
    y_oh = tf.keras.utils.to_categorical(y, num_classes=n_chars)
    return X, y_oh

# ── Build Simple RNN ──────────────────────────────────────────────────────────
def build_rnn(seq_len, units=64):
    model = Sequential([
        Embedding(n_chars, 16, input_length=seq_len),
        SimpleRNN(units, return_sequences=False),
        Dense(n_chars, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ── Generate text helper ──────────────────────────────────────────────────────
def generate_text(model, seed, seq_len, num_chars=100):
    result  = seed
    context = [char2idx[c] for c in seed[-seq_len:]]
    for _ in range(num_chars):
        x    = np.array([context])
        pred = model.predict(x, verbose=0)[0]
        next_idx = np.argmax(pred)
        result  += idx2char[next_idx]
        context  = context[1:] + [next_idx]
    return result

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT A: Effect of Sequence Length
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT A: Effect of Sequence Length")
print("=" * 60)

seq_lengths = [5, 15, 30]
EPOCHS      = 20
seq_results = {}

for seq_len in seq_lengths:
    print(f"\n  Sequence length = {seq_len} ...")
    X, y_oh = make_sequences(TEXT, seq_len)
    print(f"    Samples: {X.shape}")
    model = build_rnn(seq_len)
    hist  = model.fit(X, y_oh, epochs=EPOCHS, batch_size=64,
                      validation_split=0.1, verbose=0)
    final_loss = hist.history['val_loss'][-1]
    final_acc  = hist.history['val_accuracy'][-1]
    seq_results[seq_len] = {
        'history': hist, 'loss': final_loss, 'acc': final_acc, 'model': model
    }
    print(f"    Val Accuracy: {final_acc:.4f}, Val Loss: {final_loss:.4f}")

    # Generate sample text
    seed = TEXT[:seq_len]
    generated = generate_text(model, seed, seq_len, num_chars=80)
    print(f"    Generated: ...{generated[seq_len:seq_len+80]!r}...")

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT B: Short vs Long Sequences (long-term dependency)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT B: Short vs Long Sequences (long-term deps)")
print("=" * 60)

# Short sequence
X_short, y_short = make_sequences(TEXT, 5)
# Long sequence
X_long,  y_long  = make_sequences(TEXT, 40)

m_short = build_rnn(5,  units=64)
m_long  = build_rnn(40, units=64)

print("  Training on SHORT sequences (len=5) ...")
h_short = m_short.fit(X_short, y_short, epochs=EPOCHS, batch_size=64,
                       validation_split=0.1, verbose=0)

print("  Training on LONG  sequences (len=40) ...")
h_long  = m_long.fit(X_long,  y_long,  epochs=EPOCHS, batch_size=64,
                      validation_split=0.1, verbose=0)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EXP 7 – Simple RNN Analysis", fontsize=14)

# A: Val accuracy per sequence length
ax = axes[0, 0]
for sl, res in seq_results.items():
    ax.plot(res['history'].history['val_accuracy'], label=f"seq_len={sl}")
ax.set_title("A: Sequence Length – Val Accuracy")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(True)

# A: Val loss per sequence length
ax = axes[0, 1]
for sl, res in seq_results.items():
    ax.plot(res['history'].history['val_loss'], label=f"seq_len={sl}")
ax.set_title("A: Sequence Length – Val Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True)

# B: Short vs Long loss comparison
ax = axes[1, 0]
ax.plot(h_short.history['loss'],     label='Short (5)  – Train')
ax.plot(h_short.history['val_loss'], label='Short (5)  – Val', linestyle='--')
ax.plot(h_long.history['loss'],      label='Long  (40) – Train')
ax.plot(h_long.history['val_loss'],  label='Long  (40) – Val', linestyle='--')
ax.set_title("B: Short vs Long Sequence – Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.legend(fontsize=8); ax.grid(True)

# Bar: final accuracy
ax = axes[1, 1]
labels = [f"seq={sl}" for sl in seq_results]
accs   = [seq_results[sl]['acc'] for sl in seq_results]
bars   = ax.bar(labels, accs, color=['steelblue', 'orange', 'green'])
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{acc:.3f}", ha='center', fontsize=9)
ax.set_title("Final Val Accuracy per Sequence Length")
ax.set_ylabel("Accuracy"); ax.grid(True, axis='y')

plt.tight_layout()
plt.savefig("exp7_rnn_results.png", dpi=120)
plt.show()
print("\nPlot saved: exp7_rnn_results.png")
