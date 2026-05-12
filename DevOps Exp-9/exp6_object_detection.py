"""
EXP 6: Object Detection
- Uses MobileNetV2 + SSD pretrained model from TensorFlow Hub
  (or falls back to a simple CIFAR-10 classifier if Hub unavailable)
- Detects objects and draws bounding boxes
- Analyses confidence scores
- Evaluates image quality effect on detection
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ── COCO class labels (91 classes) ───────────────────────────────────────────
COCO_LABELS = {
    1:'person', 2:'bicycle', 3:'car', 4:'motorcycle', 5:'airplane',
    6:'bus', 7:'train', 8:'truck', 9:'boat', 10:'traffic light',
    11:'fire hydrant', 13:'stop sign', 14:'parking meter', 15:'bench',
    16:'bird', 17:'cat', 18:'dog', 19:'horse', 20:'sheep',
    21:'cow', 22:'elephant', 23:'bear', 24:'zebra', 25:'giraffe',
    27:'backpack', 28:'umbrella', 31:'handbag', 32:'tie', 33:'suitcase',
    34:'frisbee', 35:'skis', 36:'snowboard', 37:'sports ball', 38:'kite',
    39:'baseball bat', 40:'baseball glove', 41:'skateboard', 42:'surfboard',
    44:'bottle', 46:'wine glass', 47:'cup', 48:'fork', 49:'knife',
    50:'spoon', 51:'bowl', 52:'banana', 53:'apple', 54:'sandwich',
    55:'orange', 56:'broccoli', 57:'carrot', 58:'hot dog', 59:'pizza',
    60:'donut', 61:'cake', 62:'chair', 63:'couch', 64:'potted plant',
    65:'bed', 67:'dining table', 70:'toilet', 72:'tv', 73:'laptop',
    74:'mouse', 75:'remote', 76:'keyboard', 77:'cell phone', 78:'microwave',
    79:'oven', 80:'toaster', 81:'sink', 82:'refrigerator', 84:'book',
    85:'clock', 86:'vase', 87:'scissors', 88:'teddy bear', 89:'hair drier',
    90:'toothbrush'
}


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.utils import to_categorical
print("\nBuilding CIFAR-10 classifier as detection proxy ...")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
x_train = x_train[:6000] / 255.0
x_test  = x_test[:1000]  / 255.0
y_train_cat = to_categorical(y_train[:6000], 10)
# Resize for MobileNetV2
x_train_r = tf.image.resize(x_train * 255, (96, 96)).numpy() / 255.0
x_test_r  = tf.image.resize(x_test  * 255, (96, 96)).numpy() / 255.0
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96,96,3))
base.trainable = False
model = Sequential([base, GlobalAveragePooling2D(),
                    Dense(64, activation='relu'), Dropout(0.3),
                    Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("  Training ...")
model.fit(x_train_r, y_train_cat, validation_split=0.1,
          epochs=5, batch_size=64, verbose=1)
# Predict on samples
samples     = x_test[:6]
samples_r   = x_test_r[:6]
preds       = model.predict(samples_r, verbose=0)   # (6, 10)
true_labels = y_test[:6].flatten()
# ── Plot with confidence bars ────────────────────────────────────────────
fig, axes = plt.subplots(2, 6, figsize=(16, 6))
fig.suptitle("EXP 6 – Object Detection Proxy (Classifier + Confidence)", fontsize=14)
for i in range(6):
    # Top row: image
    ax = axes[0, i]
    ax.imshow(samples[i])
    pred_cls   = np.argmax(preds[i])
    confidence = preds[i][pred_cls]
    true_cls   = class_names[true_labels[i]]
    pred_name  = class_names[pred_cls]
    color      = 'green' if pred_cls == true_labels[i] else 'red'
    ax.set_title(f"True: {true_cls}\nPred: {pred_name}\nConf: {confidence:.2f}",
                 fontsize=8, color=color)
    ax.axis('off')
    # Bottom row: confidence bar chart
    ax = axes[1, i]
    ax.barh(class_names, preds[i], color='steelblue')
    ax.barh(class_names[pred_cls], preds[i][pred_cls], color='orange')
    ax.set_xlim(0, 1)
    ax.set_title("Confidence", fontsize=8)
    ax.tick_params(labelsize=6)
plt.tight_layout()
plt.savefig("exp6_detection_results.png", dpi=120)
plt.show()
# ── Image Quality Effect Analysis ────────────────────────────────────────
print("\nAnalysing effect of image quality (noise levels) ...")
test_img   = x_test_r[0:1]   # single image
noise_levels = [0, 0.05, 0.1, 0.2, 0.4]
quality_results = []
for noise in noise_levels:
    noisy = np.clip(test_img + np.random.normal(0, noise, test_img.shape), 0, 1)
    pred  = model.predict(noisy, verbose=0)
    conf  = np.max(pred)
    quality_results.append((noise, conf))
    print(f"  Noise={noise:.2f} → Max Confidence: {conf:.4f}")
fig2, ax2 = plt.subplots(figsize=(8, 4))
noises = [r[0] for r in quality_results]
confs  = [r[1] for r in quality_results]
ax2.plot(noises, confs, 'o-', color='red', linewidth=2, markersize=8)
ax2.set_title("EXP 6 – Image Quality vs Detection Confidence")
ax2.set_xlabel("Noise Level"); ax2.set_ylabel("Max Confidence Score")
ax2.grid(True)
plt.tight_layout()
plt.savefig("exp6_quality_analysis.png", dpi=120)
plt.show()

