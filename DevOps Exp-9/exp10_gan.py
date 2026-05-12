"""
EXP 10: GAN (Generative Adversarial Network)
- Generates synthetic MNIST digit images
- Tracks generator and discriminator loss during training
- Visualizes generated samples improving over iterations
- Evaluates quality of generated images
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, LeakyReLU, BatchNormalization,
                                     Reshape, Flatten, Dropout, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM  = 100
IMG_SHAPE   = (28, 28, 1)
IMG_FLAT    = 28 * 28
BATCH_SIZE  = 64
EPOCHS      = 5000       # GAN needs many iterations; each "epoch" = 1 batch step
SAMPLE_EVERY = 1000      # Save sample images every N steps
LR          = 0.0002
BETA        = 0.5

# ── Load MNIST ────────────────────────────────────────────────────────────────
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train[:10000]                    # Subset for speed
x_train = (x_train.astype('float32') - 127.5) / 127.5   # Scale to [-1, 1]
x_train = x_train.reshape(-1, IMG_FLAT)      # Flatten
print(f"Training data shape: {x_train.shape}")

# ── Generator ─────────────────────────────────────────────────────────────────
def build_generator():
    model = Sequential([
        Dense(256, input_dim=LATENT_DIM),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),

        Dense(512),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),

        Dense(1024),
        LeakyReLU(0.2),
        BatchNormalization(momentum=0.8),

        Dense(IMG_FLAT, activation='tanh'),
    ], name='Generator')
    return model

# ── Discriminator ─────────────────────────────────────────────────────────────
def build_discriminator():
    model = Sequential([
        Dense(512, input_dim=IMG_FLAT),
        LeakyReLU(0.2),
        Dropout(0.3),

        Dense(256),
        LeakyReLU(0.2),
        Dropout(0.3),

        Dense(1, activation='sigmoid'),
    ], name='Discriminator')
    return model

# ── Build Models ──────────────────────────────────────────────────────────────
generator     = build_generator()
discriminator = build_discriminator()

discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(LR, BETA),
    metrics=['accuracy']
)

# Combined (GAN) model: train generator through discriminator
discriminator.trainable = False
gan_input  = Input(shape=(LATENT_DIM,))
gan_output = discriminator(generator(gan_input))
gan_model  = Model(gan_input, gan_output, name='GAN')
gan_model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(LR, BETA)
)

generator.summary()
discriminator.summary()

# ── Training Loop ─────────────────────────────────────────────────────────────
d_losses, d_accs  = [], []
g_losses          = []
sample_images     = {}    # step → image grid

def sample_grid(step):
    noise = np.random.normal(0, 1, (25, LATENT_DIM))
    imgs  = generator.predict(noise, verbose=0)
    imgs  = 0.5 * imgs + 0.5          # Rescale to [0, 1]
    imgs  = imgs.reshape(-1, 28, 28)
    return imgs

# Labels
real_labels = np.ones((BATCH_SIZE, 1))
fake_labels = np.zeros((BATCH_SIZE, 1))

# Smoothed labels (helps training stability)
real_smooth = 0.9 * real_labels
fake_smooth = 0.1 * fake_labels

print(f"\nStarting GAN training for {EPOCHS} steps ...")
for step in range(EPOCHS + 1):

    # ── Train Discriminator ──────────────────────────────────────────────────
    idx        = np.random.randint(0, x_train.shape[0], BATCH_SIZE)
    real_imgs  = x_train[idx]

    noise      = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    fake_imgs  = generator.predict(noise, verbose=0)

    d_loss_real = discriminator.train_on_batch(real_imgs, real_smooth)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_smooth)
    d_loss      = 0.5 * (d_loss_real[0] + d_loss_fake[0])
    d_acc       = 0.5 * (d_loss_real[1] + d_loss_fake[1])

    # ── Train Generator ──────────────────────────────────────────────────────
    noise  = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    g_loss = gan_model.train_on_batch(noise, real_labels)

    d_losses.append(d_loss)
    d_accs.append(d_acc)
    g_losses.append(g_loss)

    if step % 500 == 0:
        print(f"  Step {step:5d} | D Loss: {d_loss:.4f} | D Acc: {d_acc:.4f} | G Loss: {g_loss:.4f}")

    if step % SAMPLE_EVERY == 0:
        sample_images[step] = sample_grid(step)

# ── Plot: Generated samples at different steps ────────────────────────────────
steps_to_show = sorted(sample_images.keys())
n_cols  = 5
n_rows  = 5

fig, big_axes = plt.subplots(1, len(steps_to_show),
                             figsize=(4 * len(steps_to_show), 5))
if len(steps_to_show) == 1:
    big_axes = [big_axes]

fig.suptitle("EXP 10 – GAN: Generated Images over Training Steps", fontsize=13)

for ax_big, step in zip(big_axes, steps_to_show):
    imgs  = sample_images[step]
    # Create a 5x5 grid image
    grid  = np.zeros((5 * 28, 5 * 28))
    for r in range(5):
        for c in range(5):
            grid[r*28:(r+1)*28, c*28:(c+1)*28] = imgs[r*5 + c]
    ax_big.imshow(grid, cmap='gray')
    ax_big.set_title(f"Step {step}", fontsize=11)
    ax_big.axis('off')

plt.tight_layout()
plt.savefig("exp10_gan_samples.png", dpi=120)
plt.show()

# ── Plot: Loss curves ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("EXP 10 – GAN Training Dynamics", fontsize=14)

ax = axes[0]
ax.plot(d_losses, label='Discriminator Loss', color='blue',  alpha=0.7)
ax.plot(g_losses, label='Generator Loss',     color='orange',alpha=0.7)
ax.set_title("Generator vs Discriminator Loss")
ax.set_xlabel("Training Step"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True)

# Smooth losses with moving average for clarity
def moving_avg(arr, w=50):
    return np.convolve(arr, np.ones(w)/w, mode='valid')

ax = axes[1]
ax.plot(moving_avg(d_losses), label='D Loss (smoothed)', color='blue')
ax.plot(moving_avg(g_losses), label='G Loss (smoothed)', color='orange')
ax.axhline(y=0.693, color='green', linestyle='--', label='Ideal D loss (ln2≈0.693)')
ax.set_title("Smoothed Loss (convergence analysis)")
ax.set_xlabel("Training Step"); ax.set_ylabel("Loss")
ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig("exp10_gan_loss.png", dpi=120)
plt.show()

# ── Final quality check ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
final_d = np.mean(d_losses[-500:])
final_g = np.mean(g_losses[-500:])
final_acc = np.mean(d_accs[-500:])
print(f"  Avg D Loss (last 500 steps): {final_d:.4f}")
print(f"  Avg G Loss (last 500 steps): {final_g:.4f}")
print(f"  Avg D Accuracy              : {final_acc:.4f}")
print(f"\n  Plots saved: exp10_gan_samples.png, exp10_gan_loss.png")
