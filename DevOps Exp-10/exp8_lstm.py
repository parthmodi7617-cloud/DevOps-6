import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Embedding
import matplotlib.pyplot as plt

# Synthetic Text Dataset
TEXT = "abcde fghij klmno pqrst uvwxy z " * 100

chars = sorted(set(TEXT))
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for c, i in char2idx.items()}

vocab_size = len(chars)
SEQ_LEN = 20

# Create sequences
encoded = [char2idx[c] for c in TEXT]

X = []
y = []

for i in range(len(encoded) - SEQ_LEN):
    X.append(encoded[i:i+SEQ_LEN])
    y.append(encoded[i+SEQ_LEN])

X = np.array(X)
y = tf.keras.utils.to_categorical(y, vocab_size)

# Simple RNN Model
rnn = Sequential([
    Embedding(vocab_size, 16, input_length=SEQ_LEN),
    SimpleRNN(32),
    Dense(vocab_size, activation='softmax')
])

rnn.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# LSTM Model
lstm = Sequential([
    Embedding(vocab_size, 16, input_length=SEQ_LEN),
    LSTM(32),
    Dense(vocab_size, activation='softmax')
])

lstm.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train models
print("Training Simple RNN...")
h_rnn = rnn.fit(X, y, epochs=5, batch_size=32, validation_split=0.1, verbose=0)

print("Training LSTM...")
h_lstm = lstm.fit(X, y, epochs=5, batch_size=32, validation_split=0.1, verbose=0)

# Plot comparison
plt.plot(h_rnn.history['val_accuracy'], label="Simple RNN")
plt.plot(h_lstm.history['val_accuracy'], label="LSTM")

plt.title("Simple RNN vs LSTM on Synthetic Text")
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Prediction example
seed = TEXT[:SEQ_LEN]
context = [char2idx[c] for c in seed]

result = seed

for i in range(50):
    x = np.array([context])
    pred = lstm.predict(x, verbose=0)[0]
    next_idx = np.argmax(pred)
    result += idx2char[next_idx]
    context = context[1:] + [next_idx]

print("\nGenerated Text:")
print(result)