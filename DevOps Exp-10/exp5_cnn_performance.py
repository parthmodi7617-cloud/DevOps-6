import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Synthetic Dataset
x_train = np.random.rand(1000, 32, 32, 3)
y_train = np.random.randint(0, 10, 1000)

x_test = np.random.rand(200, 32, 32, 3)
y_test = np.random.randint(0, 10, 200)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# CNN Model
def create_model(lr):

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Different Learning Rates
rates = [0.01, 0.001]

for lr in rates:

    print("\nTraining with LR =", lr)

    model = create_model(lr)

    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        validation_split=0.1,
        verbose=0
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    print("Accuracy:", acc)

    plt.plot(history.history['val_accuracy'], label=f"LR={lr}")

plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()