import os
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping

# Konfigurasi Path
DATASET_DIR = r"d:\KULIAH\SKRIPSI\Dataset\gambar"
MODEL_SAVE_DIR = r"d:\KULIAH\SKRIPSI\Percobaan 2\backend\models_vision"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Parameter Hyperparameters
IMG_SIZE = (224, 224) # Standar MobileNetV2
BATCH_SIZE = 32
EPOCHS = 15

print("Memuat dataset gambar...")
# Membagi data menjadi 80% Training dan 20% Validation
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print(f"Ditemukan {len(class_names)} kelas: {class_names}")

# Simpan nama kelas ke JSON agar Flask tahu urutan ID-nya
with open(os.path.join(MODEL_SAVE_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f)

# Optimalisasi performa loading dataset
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data Augmentation: Mencegah Overfitting dengan cara memutar & membalik gambar secara acak
data_augmentation = Sequential([
  RandomFlip("horizontal"),
  RandomRotation(0.2),
  RandomZoom(0.2),
])

# Memuat model dasar MobileNetV2 (Pre-trained dari Google)
print("Membangun arsitektur MobileNetV2...")
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')

# Kunci (Freeze) lapisan dasar agar tidak ikut dilatih (karena sudah pintar dari Google)
base_model.trainable = False

# Menyusun Kepala Model (Top Layers) khusus untuk 8 Komoditas kita
inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
# MobileNetV2 membutuhkan input dengan rentang [-1, 1]
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x) # Dropout untuk regularisasi
outputs = Dense(len(class_names), activation='softmax')(x)

model = Model(inputs, outputs)

# Kompilasi Model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Pelatihan Model
print("Memulai proses Training...")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# Menyimpan Model
model_path = os.path.join(MODEL_SAVE_DIR, "vision_model.keras")
model.save(model_path)
print(f"\nTraining Selesai! Model disimpan di: {model_path}")
