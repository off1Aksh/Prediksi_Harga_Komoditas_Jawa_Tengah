"""
Training Model Final MobileNetV2 — Menggunakan Konfigurasi Terbaik dari K-Fold Tuning
=======================================================================================
Dataset: gambar_balanced (100/kelas × 8 kelas = 800 gambar)
Split: 80% Training / 20% Validation (seed=123, konsisten dengan train_vision.py asli)

Konfigurasi terbaik hasil tuning:
  - Learning Rate  : 0.001
  - Batch Size     : 16
  - Dropout Rate   : 0.3
  - Max Epochs     : 50
  - Augmentation   : enhanced (flip + rotation + zoom + brightness + contrast)
  - Early Stopping : val_loss, patience=5

Output:
  - models_vision/vision_model.keras        (model final)
  - models_vision/class_names.json          (mapping kelas)
  - models_vision/training_history.png      (kurva loss & accuracy)
  - models_vision/classification_report.txt (laporan evaluasi per kelas)
  - models_vision/confusion_matrix.png      (confusion matrix)
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout,
    RandomFlip, RandomRotation, RandomZoom,
    RandomBrightness, RandomContrast
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# KONFIGURASI — Hasil dari K-Fold Tuning
# ============================================================
DATASET_DIR = r"d:\KULIAH\SKRIPSI\Dataset\gambar_balanced"
MODEL_SAVE_DIR = r"d:\KULIAH\SKRIPSI\Percobaan 2\backend\models_vision"

# Untuk Kaggle, uncomment baris di bawah:
# DATASET_DIR = "/kaggle/input/gambar-balanced/gambar_balanced"
# MODEL_SAVE_DIR = "/kaggle/working/models_vision"

IMG_SIZE = (224, 224)
BATCH_SIZE = 64           # Hasil tuning final (sebelumnya: 32)
EPOCHS = 50               # Hasil tuning final (sebelumnya: 15)
LEARNING_RATE = 0.001     # Hasil tuning final (tetap sama)
DROPOUT_RATE = 0.3        # Hasil tuning final (sebelumnya: 0.2)
SEED = 123                # Sama dengan train_vision.py asli

# Reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ============================================================
# GPU CHECK
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU ditemukan: {len(gpus)} device(s)")
else:
    print("⚠ Tidak ada GPU — menggunakan CPU")

# ============================================================
# 1. MEMUAT DATASET
# ============================================================
print("\n📂 Memuat dataset gambar...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"✅ Ditemukan {num_classes} kelas: {class_names}")

# Simpan nama kelas
with open(os.path.join(MODEL_SAVE_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f)
print(f"💾 class_names.json disimpan.")

# Optimasi performa loading
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ============================================================
# 2. MEMBANGUN ARSITEKTUR MODEL
# ============================================================
print("\n🏗 Membangun arsitektur MobileNetV2...")

# Data Augmentation — Minimal (hasil tuning final)
data_augmentation = Sequential([
    RandomFlip("horizontal"),
], name='data_augmentation_minimal')

# Base Model (Frozen — tidak ikut dilatih)
base_model = MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Arsitektur Lengkap
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(DROPOUT_RATE)(x)      # 0.3 (hasil tuning)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

model.summary()

# ============================================================
# 3. TRAINING
# ============================================================
print("\n🚀 Memulai Training...")
print(f"   Learning Rate : {LEARNING_RATE}")
print(f"   Batch Size    : {BATCH_SIZE}")
print(f"   Dropout Rate  : {DROPOUT_RATE}")
print(f"   Max Epochs    : {EPOCHS}")
print(f"   Augmentation  : minimal")
print(f"   Early Stopping: val_loss, patience=5")

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ============================================================
# 4. SIMPAN MODEL
# ============================================================
model_path = os.path.join(MODEL_SAVE_DIR, "vision_model.keras")
model.save(model_path)
print(f"\n💾 Model disimpan: {model_path}")
print(f"   Ukuran: {os.path.getsize(model_path) / (1024*1024):.2f} MB")

# ============================================================
# 5. EVALUASI PADA VALIDATION SET
# ============================================================
print("\n📊 Evaluasi pada Validation Set...")

# Kumpulkan semua prediksi
y_true = []
y_pred = []
for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1).tolist())
    y_true.extend(labels.numpy().tolist())

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\n" + report)

# Simpan report ke file
report_path = os.path.join(MODEL_SAVE_DIR, "classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("CLASSIFICATION REPORT — MODEL VISION FINAL\n")
    f.write(f"Konfigurasi: lr={LEARNING_RATE}, bs={BATCH_SIZE}, dropout={DROPOUT_RATE}, ")
    f.write(f"epochs={EPOCHS}, augmentation=enhanced\n")
    f.write(f"Dataset: {DATASET_DIR} (800 gambar, 100/kelas)\n")
    f.write(f"Split: 80/20 (seed={SEED})\n")
    f.write(f"Epochs berjalan: {len(history.history['loss'])}\n")
    f.write("=" * 60 + "\n\n")
    f.write(report)
print(f"💾 Classification report disimpan: {report_path}")

# ============================================================
# 6. CONFUSION MATRIX
# ============================================================
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=class_names, yticklabels=class_names, ax=ax
)
ax.set_xlabel('Prediksi', fontsize=12)
ax.set_ylabel('Aktual', fontsize=12)
ax.set_title('Confusion Matrix — Model Vision Final (MobileNetV2)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()

cm_path = os.path.join(MODEL_SAVE_DIR, "confusion_matrix.png")
fig.savefig(cm_path, dpi=150)
plt.close(fig)
print(f"💾 Confusion matrix disimpan: {cm_path}")

# ============================================================
# 7. GRAFIK TRAINING HISTORY
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

epochs_range = range(1, len(history.history['loss']) + 1)

# Loss
axes[0].plot(epochs_range, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
axes[0].plot(epochs_range, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss', fontsize=11)
axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Accuracy
axes[1].plot(epochs_range, history.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
axes[1].plot(epochs_range, history.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Accuracy', fontsize=11)
axes[1].set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(0, 1.05)

plt.suptitle('MobileNetV2 Training History — Konfigurasi Optimal', fontsize=14, fontweight='bold')
plt.tight_layout()

history_path = os.path.join(MODEL_SAVE_DIR, "training_history.png")
fig.savefig(history_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"💾 Training history disimpan: {history_path}")

# ============================================================
# SELESAI
# ============================================================
final_acc = max(history.history['val_accuracy'])
final_loss = min(history.history['val_loss'])
epochs_run = len(history.history['loss'])

print(f"\n{'='*60}")
print(f"✅ TRAINING SELESAI!")
print(f"{'='*60}")
print(f"   Epochs berjalan   : {epochs_run}")
print(f"   Best Val Accuracy : {final_acc:.4f}")
print(f"   Best Val Loss     : {final_loss:.4f}")
print(f"\n📁 Semua output tersimpan di: {MODEL_SAVE_DIR}")
print(f"   📦 vision_model.keras")
print(f"   📄 class_names.json")
print(f"   📄 classification_report.txt")
print(f"   📊 confusion_matrix.png")
print(f"   📈 training_history.png")
