import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
import matplotlib.pyplot as plt

# Synthetic Dataset
x_train = np.random.rand(1000, 28 * 28)

LATENT_DIM = 50

# Generator
generator = Sequential([
    Dense(128, activation='relu', input_dim=LATENT_DIM),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28))
])

# Discriminator
discriminator = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

discriminator.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# GAN
discriminator.trainable = False

gan = Sequential([
    generator,
    discriminator
])

gan.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)

# Training
epochs = 500
batch_size = 32

for epoch in range(epochs):

    # Real Images
    idx = np.random.randint(0, x_train.shape[0], batch_size)

    real_imgs = x_train[idx].reshape(-1, 28, 28)

    # Fake Images
    noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))

    fake_imgs = generator.predict(noise, verbose=0)

    # Labels
    real_y = np.ones((batch_size, 1))

    fake_y = np.zeros((batch_size, 1))

    # Train Discriminator
    d_loss_real = discriminator.train_on_batch(real_imgs, real_y)

    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_y)

    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))

    g_loss = gan.train_on_batch(noise, real_y)

    if epoch % 100 == 0:

        print(f"Epoch {epoch}")

# Generate Images
noise = np.random.normal(0, 1, (5, LATENT_DIM))

generated = generator.predict(noise)

# Show Images
fig, axes = plt.subplots(1, 5, figsize=(10,2))

for i in range(5):

    axes[i].imshow(generated[i], cmap='gray')

    axes[i].axis('off')

plt.show()