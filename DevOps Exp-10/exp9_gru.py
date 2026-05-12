import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Embedding
import matplotlib.pyplot as plt

# Synthetic Text Dataset
TEXT = "abcdef ghijkl mnopqr stuvwx yz " * 100

chars = sorted(set(TEXT))

char2idx = {c:i for i,c in enumerate(chars)}

idx2char = {i:c for c,i in char2idx.items()}

vocab_size = len(chars)

SEQ_LEN = 15

# Create sequences
encoded = [char2idx[c] for c in TEXT]

X = []
y = []

for i in range(len(encoded) - SEQ_LEN):

    X.append(encoded[i:i+SEQ_LEN])

    y.append(encoded[i+SEQ_LEN])

X = np.array(X)

y = tf.keras.utils.to_categorical(y, vocab_size)

# Model Functions
def build_rnn():

    model = Sequential([
        Embedding(vocab_size, 8, input_length=SEQ_LEN),
        SimpleRNN(32),
        Dense(vocab_size, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def build_lstm():

    model = Sequential([
        Embedding(vocab_size, 8, input_length=SEQ_LEN),
        LSTM(32),
        Dense(vocab_size, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def build_gru():

    model = Sequential([
        Embedding(vocab_size, 8, input_length=SEQ_LEN),
        GRU(32),
        Dense(vocab_size, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

models = {
    "RNN": build_rnn(),
    "LSTM": build_lstm(),
    "GRU": build_gru()
}

histories = {}

# Train Models
for name, model in models.items():

    print("\nTraining", name)

    history = model.fit(
        X,
        y,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )

    histories[name] = history

# Plot Accuracy
for name, history in histories.items():

    plt.plot(
        history.history['val_accuracy'],
        label=name
    )

plt.title("RNN vs LSTM vs GRU")

plt.xlabel("Epoch")

plt.ylabel("Validation Accuracy")

plt.legend()

plt.grid(True)

plt.show()