import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import matplotlib.pyplot as plt

# Synthetic Text Dataset
TEXT = "abcdefg hijklmnop qrstuv wxyz " * 100

# Vocabulary
chars = sorted(set(TEXT))

char2idx = {c:i for i,c in enumerate(chars)}

idx2char = {i:c for c,i in char2idx.items()}

vocab_size = len(chars)

print("Vocabulary Size:", vocab_size)

# Create Sequences
def make_sequences(text, seq_len):

    X = []
    y = []

    encoded = [char2idx[c] for c in text]

    for i in range(len(encoded) - seq_len):

        X.append(encoded[i:i+seq_len])

        y.append(encoded[i+seq_len])

    X = np.array(X)

    y = tf.keras.utils.to_categorical(y, vocab_size)

    return X, y

# Build RNN
def build_rnn(seq_len):

    model = Sequential([

        Embedding(vocab_size, 8, input_length=seq_len),

        SimpleRNN(32),

        Dense(vocab_size, activation='softmax')

    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Sequence Lengths
seq_lengths = [5, 10]

results = {}

for seq_len in seq_lengths:

    print("\nTraining with Sequence Length =", seq_len)

    X, y = make_sequences(TEXT, seq_len)

    model = build_rnn(seq_len)

    history = model.fit(
        X,
        y,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    results[seq_len] = history

# Plot
for seq_len, history in results.items():

    plt.plot(
        history.history['val_accuracy'],
        label=f"Seq={seq_len}"
    )

plt.title("Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.show()