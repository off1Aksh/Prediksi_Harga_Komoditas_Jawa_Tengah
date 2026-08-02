"""
Hyperparameter Tuning Terstruktur MobileNetV2 — Stratified K-Fold Cross-Validation
====================================================================================
Metode: Control Variable — ubah 1 variabel per tahap, kunci yang terbaik, lanjut.

Tahap 0: Baseline   (lr=1e-3, bs=32, dropout=0.2, epochs=30, aug=baseline)
Tahap 1: Learning Rate  → Uji: 1e-3, 1e-4, 1e-5
Tahap 2: Batch Size     → Uji: 16, 32, 64
Tahap 3: Dropout Rate   → Uji: 0.2, 0.3, 0.5
Tahap 4: Max Epochs     → Uji: 15, 30, 50
Tahap 5: Augmentation   → Uji: minimal, baseline, enhanced

Cara Pakai di Kaggle:
1. Upload folder gambar_balanced/ sebagai Kaggle Dataset
2. Buat Notebook baru → Nyalakan GPU (Settings → Accelerator → GPU)
3. Add dataset yang sudah di-upload
4. Copy-paste seluruh script ini ke satu cell, atau pisah per section
5. Sesuaikan DATASET_DIR di bawah
6. Run All

Estimasi Waktu (GPU): ~2-3 jam total
Support Resume: Ya — jika terputus, jalankan ulang, tahap yang sudah selesai akan di-skip.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, precision_score, recall_score
)
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import time
import json

# ============================================================
# KONFIGURASI — SESUAIKAN SEBELUM MENJALANKAN
# ============================================================
# Path dataset (sesuaikan jika berbeda)
DATASET_DIR = r"d:\KULIAH\SKRIPSI\Dataset\gambar_balanced"

# Path output hasil tuning
OUTPUT_DIR = r"d:\KULIAH\SKRIPSI\Percobaan 2\backend\hasil_tuning_vision"

# Parameter tetap
IMG_SIZE = (224, 224)
K_FOLDS = 5
SEED = 42

# ============================================================
# SETUP ENVIRONMENT
# ============================================================
np.random.seed(SEED)
tf.random.set_seed(SEED)

# GPU memory growth (hindari Out-of-Memory)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✅ GPU ditemukan: {len(gpus)} device(s)")
else:
    print("⚠ Tidak ada GPU terdeteksi — akan menggunakan CPU (jauh lebih lambat)")

# Ekstensi gambar yang valid
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


# ============================================================
# FUNGSI: LOAD DATASET KE MEMORY
# ============================================================
def load_dataset(dataset_dir, img_size=(224, 224)):
    """
    Memuat seluruh gambar dari folder dataset ke numpy array.
    Menggunakan Stratified split, jadi seluruh data harus di-load ke RAM.
    800 gambar × 224×224×3 ≈ 460 MB — aman untuk Kaggle (16GB RAM).
    """
    images = []
    labels = []
    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    print(f"\n📂 Memuat dataset dari: {dataset_dir}")

    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        count = 0
        for img_name in sorted(os.listdir(class_dir)):
            # Skip file non-gambar
            if os.path.splitext(img_name)[1].lower() not in VALID_EXTENSIONS:
                continue
            img_path = os.path.join(class_dir, img_name)
            try:
                img = tf.keras.utils.load_img(img_path, target_size=img_size)
                img_array = tf.keras.utils.img_to_array(img)
                images.append(img_array)
                labels.append(class_to_idx[class_name])
                count += 1
            except Exception as e:
                print(f"  ⚠ Skip {img_path}: {e}")
        print(f"  {class_name}: {count} gambar")

    X = np.array(images, dtype=np.float32)
    y = np.array(labels)
    print(f"\n📊 Total: {len(X)} gambar, {len(class_names)} kelas")
    print(f"   Shape X: {X.shape}")
    print(f"   Shape y: {y.shape}")
    print(f"   Distribusi: {dict(zip(class_names, np.bincount(y)))}")

    return X, y, class_names


# ============================================================
# FUNGSI: BUILD MODEL MOBILENETV2
# ============================================================
def build_model(num_classes, learning_rate=1e-3, dropout_rate=0.2,
                augmentation_type='baseline'):
    """
    Membangun arsitektur MobileNetV2 + Transfer Learning.
    Model dibangun dari scratch untuk setiap fold (menghindari weight leakage).

    Arsitektur:
    Input(224,224,3) → Augmentation → Preprocess → MobileNetV2(frozen)
    → GlobalAveragePooling2D → Dropout → Dense(num_classes, softmax)
    """

    # --- Layer Data Augmentation ---
    if augmentation_type == 'minimal':
        aug_layers = [
            tf.keras.layers.RandomFlip("horizontal"),
        ]
    elif augmentation_type == 'enhanced':
        aug_layers = [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ]
    else:  # baseline
        aug_layers = [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ]

    data_augmentation = tf.keras.Sequential(aug_layers, name='data_augmentation')

    # --- Base Model (Frozen / Tidak ikut di-train) ---
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    # --- Susun Arsitektur ---
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # Rescale ke [-1, 1]
    x = base_model(x, training=False)  # Freeze BatchNorm
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # --- Compile ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return model


# ============================================================
# FUNGSI: JALANKAN K-FOLD UNTUK 1 KONFIGURASI
# ============================================================
def run_kfold(X, y, class_names, params, config_name, stage_dir):
    """
    Menjalankan Stratified K-Fold Cross-Validation untuk satu set parameter.
    Mengembalikan list of dict (hasil per fold) dan confusion matrix gabungan.
    """

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []
    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        fold_num = fold_idx + 1
        print(f"    Fold {fold_num}/{K_FOLDS}...", end=" ", flush=True)
        start_time = time.time()

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Build model baru dari scratch (hindari weight leakage)
        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED + fold_idx)  # Seed berbeda tiap fold
        np.random.seed(SEED + fold_idx)

        model = build_model(
            num_classes=len(class_names),
            learning_rate=params['learning_rate'],
            dropout_rate=params['dropout_rate'],
            augmentation_type=params.get('augmentation', 'baseline')
        )

        # Callback: Early Stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Training
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            callbacks=[early_stop],
            verbose=0
        )

        # Prediksi pada validation set
        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)

        # Hitung metrik
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
        prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_val, y_pred, average='macro', zero_division=0)
        epochs_run = len(history.history['loss'])
        best_val_loss = min(history.history['val_loss'])
        best_val_acc = max(history.history['val_accuracy'])

        elapsed = time.time() - start_time
        print(f"Acc={acc:.4f}  F1={f1:.4f}  Epochs={epochs_run}  ({elapsed:.1f}s)")

        fold_results.append({
            'config': config_name,
            'fold': fold_num,
            'accuracy': round(acc, 4),
            'f1_macro': round(f1, 4),
            'precision_macro': round(prec, 4),
            'recall_macro': round(rec, 4),
            'epochs_run': epochs_run,
            'best_val_loss': round(best_val_loss, 4),
            'best_val_acc': round(best_val_acc, 4),
            'time_seconds': round(elapsed, 1)
        })

        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())

        # Bersihkan memori
        del model, history
        gc.collect()

    # Simpan confusion matrix gabungan (semua fold digabung)
    cm = confusion_matrix(all_y_true, all_y_pred)
    _save_confusion_matrix(cm, class_names, config_name, stage_dir)

    return fold_results


# ============================================================
# FUNGSI: SIMPAN CONFUSION MATRIX
# ============================================================
def _save_confusion_matrix(cm, class_names, config_name, output_dir):
    """Simpan heatmap confusion matrix ke PNG."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel('Prediksi', fontsize=12)
    ax.set_ylabel('Aktual', fontsize=12)
    ax.set_title(f'Confusion Matrix — {config_name}', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    safe_name = config_name.replace(' ', '_').replace('=', '_')
    filepath = os.path.join(output_dir, f"confusion_matrix_{safe_name}.png")
    fig.savefig(filepath, dpi=150)
    plt.close(fig)


# ============================================================
# FUNGSI: GRAFIK PERBANDINGAN ANTAR KONFIGURASI
# ============================================================
def _save_comparison_chart(recap_df, stage_name, output_dir):
    """Buat bar chart perbandingan accuracy & F1-score (dengan error bar)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    configs = recap_df['config'].tolist()
    x = np.arange(len(configs))

    # Tentukan index terbaik untuk highlighting
    best_acc_pos = recap_df['mean_accuracy'].values.argmax()
    best_f1_pos = recap_df['mean_f1_macro'].values.argmax()

    # Warna: terbaik = gelap, sisanya = terang
    colors_acc = ['#2E7D32' if i == best_acc_pos else '#A5D6A7' for i in range(len(configs))]
    colors_f1 = ['#1565C0' if i == best_f1_pos else '#90CAF9' for i in range(len(configs))]

    # --- Chart 1: Accuracy ---
    bars1 = axes[0].bar(
        x, recap_df['mean_accuracy'], yerr=recap_df['std_accuracy'],
        capsize=5, color=colors_acc, edgecolor='black', linewidth=0.5
    )
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_title('Accuracy (Mean ± Std)', fontsize=12, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(configs, rotation=45, ha='right', fontsize=10)
    y_min_acc = max(0, recap_df['mean_accuracy'].min() - 0.15)
    axes[0].set_ylim(y_min_acc, 1.05)

    for i, (v, s) in enumerate(zip(recap_df['mean_accuracy'], recap_df['std_accuracy'])):
        label = f'{v:.4f}'
        if i == best_acc_pos:
            label += ' ★'
        axes[0].text(i, v + s + 0.008, label, ha='center', fontsize=9, fontweight='bold')

    # --- Chart 2: F1-Score Macro ---
    bars2 = axes[1].bar(
        x, recap_df['mean_f1_macro'], yerr=recap_df['std_f1_macro'],
        capsize=5, color=colors_f1, edgecolor='black', linewidth=0.5
    )
    axes[1].set_ylabel('F1-Score Macro', fontsize=11)
    axes[1].set_title('F1-Score Macro (Mean ± Std)', fontsize=12, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(configs, rotation=45, ha='right', fontsize=10)
    y_min_f1 = max(0, recap_df['mean_f1_macro'].min() - 0.15)
    axes[1].set_ylim(y_min_f1, 1.05)

    for i, (v, s) in enumerate(zip(recap_df['mean_f1_macro'], recap_df['std_f1_macro'])):
        label = f'{v:.4f}'
        if i == best_f1_pos:
            label += ' ★'
        axes[1].text(i, v + s + 0.008, label, ha='center', fontsize=9, fontweight='bold')

    plt.suptitle(stage_name, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "grafik_perbandingan.png"),
        dpi=150, bbox_inches='tight'
    )
    plt.close(fig)


# ============================================================
# FUNGSI: JALANKAN 1 TAHAP TUNING (SEMUA VARIASI × K-FOLD)
# ============================================================
def run_stage(X, y, class_names, stage_name, stage_dir,
              param_variations, base_params, value_map):
    """
    Menjalankan satu tahap tuning lengkap.

    Args:
        X, y, class_names: Dataset
        stage_name: Nama tahap (untuk display)
        stage_dir: Folder output tahap ini
        param_variations: Dict {config_name: {param_overrides}}
        base_params: Dict parameter dasar (dari tahap sebelumnya)
        value_map: Dict {config_name: actual_value} untuk extract hasil

    Returns:
        (recap_df, best_value): DataFrame rekapitulasi dan nilai parameter terbaik
    """

    os.makedirs(stage_dir, exist_ok=True)
    recap_path = os.path.join(stage_dir, "rekapitulasi.csv")

    # --- Resume: jika sudah selesai, skip ---
    if os.path.exists(recap_path):
        print(f"\n{'='*60}")
        print(f"⏭ {stage_name} — SUDAH SELESAI SEBELUMNYA (skip)")
        print(f"{'='*60}")
        recap = pd.read_csv(recap_path)
        best_config = recap.loc[recap['mean_f1_macro'].idxmax(), 'config']
        best_val = value_map[best_config]
        print(f"  🏆 Terbaik: {best_config} → F1={recap.loc[recap['mean_f1_macro'].idxmax(), 'mean_f1_macro']:.4f}")
        return recap, best_val

    print(f"\n{'='*60}")
    print(f"🔬 {stage_name}")
    print(f"{'='*60}")

    all_fold_results = []
    stage_start = time.time()

    for var_name, var_overrides in param_variations.items():
        # Gabungkan base params + variasi tahap ini
        params = {**base_params, **var_overrides}

        print(f"\n  📋 Konfigurasi: {var_name}")
        print(f"     lr={params['learning_rate']}, bs={params['batch_size']}, "
              f"dropout={params['dropout_rate']}, epochs={params['epochs']}, "
              f"aug={params.get('augmentation', 'baseline')}")

        fold_results = run_kfold(X, y, class_names, params, var_name, stage_dir)
        all_fold_results.extend(fold_results)

    # --- Simpan hasil per fold ---
    df_folds = pd.DataFrame(all_fold_results)
    df_folds.to_csv(os.path.join(stage_dir, "hasil_per_fold.csv"), index=False)

    # --- Buat rekapitulasi (rata-rata ± std per konfigurasi) ---
    recap = df_folds.groupby('config').agg(
        mean_accuracy=('accuracy', 'mean'),
        std_accuracy=('accuracy', 'std'),
        mean_f1_macro=('f1_macro', 'mean'),
        std_f1_macro=('f1_macro', 'std'),
        mean_precision=('precision_macro', 'mean'),
        std_precision=('precision_macro', 'std'),
        mean_recall=('recall_macro', 'mean'),
        std_recall=('recall_macro', 'std'),
        mean_epochs=('epochs_run', 'mean'),
        mean_val_loss=('best_val_loss', 'mean'),
        total_time_seconds=('time_seconds', 'sum'),
    ).reset_index()

    # Bulatkan
    for col in recap.columns:
        if recap[col].dtype == 'float64':
            recap[col] = recap[col].round(4)

    recap = recap.sort_values('mean_f1_macro', ascending=False)
    recap.to_csv(recap_path, index=False)

    # --- Tentukan terbaik ---
    best_row = recap.iloc[0]
    best_config = best_row['config']
    best_val = value_map[best_config]

    stage_elapsed = time.time() - stage_start

    print(f"\n  {'─'*50}")
    print(f"  🏆 TERBAIK: {best_config}")
    print(f"     Accuracy : {best_row['mean_accuracy']:.4f} ± {best_row['std_accuracy']:.4f}")
    print(f"     F1-Score : {best_row['mean_f1_macro']:.4f} ± {best_row['std_f1_macro']:.4f}")
    print(f"     Precision: {best_row['mean_precision']:.4f} ± {best_row['std_precision']:.4f}")
    print(f"     Recall   : {best_row['mean_recall']:.4f} ± {best_row['std_recall']:.4f}")
    print(f"     Waktu    : {stage_elapsed/60:.1f} menit")

    # --- Print tabel rekapitulasi ---
    print(f"\n  📊 Rekapitulasi {stage_name}:")
    display_cols = ['config', 'mean_accuracy', 'std_accuracy',
                    'mean_f1_macro', 'std_f1_macro', 'mean_epochs']
    print(recap[display_cols].to_string(index=False))

    # --- Buat grafik perbandingan ---
    if len(recap) > 1:
        _save_comparison_chart(recap, stage_name, stage_dir)

    return recap, best_val


# ============================================================
# FUNGSI: REKAPITULASI FINAL SEMUA TAHAP
# ============================================================
def save_final_summary(output_dir, current_best, total_time, stage_recaps):
    """Buat ringkasan akhir: JSON, TXT, dan CSV gabungan semua tahap."""

    # 1. JSON konfigurasi terbaik
    json_path = os.path.join(output_dir, "konfigurasi_terbaik.json")
    with open(json_path, "w") as f:
        json.dump(current_best, f, indent=2, ensure_ascii=False)

    # 2. TXT ringkasan naratif
    txt_path = os.path.join(output_dir, "ringkasan_tuning.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("RINGKASAN HYPERPARAMETER TUNING MOBILENETV2\n")
        f.write("Stratified K-Fold Cross-Validation (Control Variable Method)\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Dataset      : {DATASET_DIR}\n")
        f.write(f"K-Fold       : {K_FOLDS}\n")
        f.write(f"Seed         : {SEED}\n")
        f.write(f"Image Size   : {IMG_SIZE}\n")
        f.write(f"Total Waktu  : {total_time/60:.1f} menit ({total_time/3600:.1f} jam)\n\n")

        f.write("KONFIGURASI TERBAIK (FINAL):\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Learning Rate  : {current_best['learning_rate']}\n")
        f.write(f"  Batch Size     : {current_best['batch_size']}\n")
        f.write(f"  Dropout Rate   : {current_best['dropout_rate']}\n")
        f.write(f"  Max Epochs     : {current_best['epochs']}\n")
        f.write(f"  Augmentation   : {current_best['augmentation']}\n\n")

        f.write("HASIL PER TAHAP:\n")
        f.write("-" * 40 + "\n")
        for stage_name, recap_df in stage_recaps:
            best = recap_df.iloc[0]  # Sudah di-sort descending by F1
            f.write(f"\n  {stage_name}:\n")
            f.write(f"    Terbaik   : {best['config']}\n")
            f.write(f"    Accuracy  : {best['mean_accuracy']:.4f} ± {best['std_accuracy']:.4f}\n")
            f.write(f"    F1 Macro  : {best['mean_f1_macro']:.4f} ± {best['std_f1_macro']:.4f}\n")

    # 3. CSV gabungan semua tahap
    all_recaps = []
    for stage_name, recap_df in stage_recaps:
        recap_copy = recap_df.copy()
        recap_copy.insert(0, 'tahap', stage_name)
        # Tandai konfigurasi terbaik
        recap_copy['terbaik'] = ''
        recap_copy.iloc[0, recap_copy.columns.get_loc('terbaik')] = '★'
        all_recaps.append(recap_copy)

    combined = pd.concat(all_recaps, ignore_index=True)
    combined.to_csv(os.path.join(output_dir, "rekapitulasi_semua_tahap.csv"), index=False)

    print(f"\n📁 File output tersimpan di: {output_dir}")
    print(f"   📄 konfigurasi_terbaik.json")
    print(f"   📄 ringkasan_tuning.txt")
    print(f"   📄 rekapitulasi_semua_tahap.csv")


# ============================================================
# ============================================================
#                        MAIN PIPELINE
# ============================================================
# ============================================================
def main():
    print("=" * 60)
    print("🚀 HYPERPARAMETER TUNING MOBILENETV2")
    print("   Stratified K-Fold Cross-Validation")
    print("   Control Variable Method")
    print("=" * 60)

    # ----------------------------------------------------------
    # LOAD DATASET
    # ----------------------------------------------------------
    X, y, class_names = load_dataset(DATASET_DIR, IMG_SIZE)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Simpan info kelas
    with open(os.path.join(OUTPUT_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2, ensure_ascii=False)

    total_start = time.time()

    # Parameter awal — akan di-update seiring tahap berjalan
    current_best = {
        'learning_rate': 1e-3,
        'batch_size': 32,
        'dropout_rate': 0.2,
        'epochs': 30,
        'augmentation': 'baseline'
    }

    # Kumpulkan rekapitulasi semua tahap
    stage_recaps = []

    # ==========================================================
    # TAHAP 0: BASELINE
    # ==========================================================
    stage_dir = os.path.join(OUTPUT_DIR, "tahap_0_baseline")
    variations = {'baseline': {}}
    value_map = {'baseline': None}

    recap, _ = run_stage(
        X, y, class_names,
        stage_name="Tahap 0: Baseline",
        stage_dir=stage_dir,
        param_variations=variations,
        base_params=current_best,
        value_map=value_map
    )
    stage_recaps.append(("Tahap 0: Baseline", recap))

    # ==========================================================
    # TAHAP 1: LEARNING RATE
    # ==========================================================
    stage_dir = os.path.join(OUTPUT_DIR, "tahap_1_learning_rate")
    variations = {
        'lr=1e-3': {'learning_rate': 1e-3},
        'lr=1e-4': {'learning_rate': 1e-4},
        'lr=1e-5': {'learning_rate': 1e-5},
    }
    value_map = {
        'lr=1e-3': 1e-3,
        'lr=1e-4': 1e-4,
        'lr=1e-5': 1e-5,
    }

    recap, best_lr = run_stage(
        X, y, class_names,
        stage_name="Tahap 1: Learning Rate",
        stage_dir=stage_dir,
        param_variations=variations,
        base_params=current_best,
        value_map=value_map
    )
    current_best['learning_rate'] = best_lr
    stage_recaps.append(("Tahap 1: Learning Rate", recap))
    print(f"\n  🔒 Learning Rate di-lock: {best_lr}")

    # ==========================================================
    # TAHAP 2: BATCH SIZE
    # ==========================================================
    stage_dir = os.path.join(OUTPUT_DIR, "tahap_2_batch_size")
    variations = {
        'bs=16': {'batch_size': 16},
        'bs=32': {'batch_size': 32},
        'bs=64': {'batch_size': 64},
    }
    value_map = {
        'bs=16': 16,
        'bs=32': 32,
        'bs=64': 64,
    }

    recap, best_bs = run_stage(
        X, y, class_names,
        stage_name="Tahap 2: Batch Size",
        stage_dir=stage_dir,
        param_variations=variations,
        base_params=current_best,
        value_map=value_map
    )
    current_best['batch_size'] = best_bs
    stage_recaps.append(("Tahap 2: Batch Size", recap))
    print(f"\n  🔒 Batch Size di-lock: {best_bs}")

    # ==========================================================
    # TAHAP 3: DROPOUT RATE
    # ==========================================================
    stage_dir = os.path.join(OUTPUT_DIR, "tahap_3_dropout")
    variations = {
        'dropout=0.2': {'dropout_rate': 0.2},
        'dropout=0.3': {'dropout_rate': 0.3},
        'dropout=0.5': {'dropout_rate': 0.5},
    }
    value_map = {
        'dropout=0.2': 0.2,
        'dropout=0.3': 0.3,
        'dropout=0.5': 0.5,
    }

    recap, best_dropout = run_stage(
        X, y, class_names,
        stage_name="Tahap 3: Dropout Rate",
        stage_dir=stage_dir,
        param_variations=variations,
        base_params=current_best,
        value_map=value_map
    )
    current_best['dropout_rate'] = best_dropout
    stage_recaps.append(("Tahap 3: Dropout Rate", recap))
    print(f"\n  🔒 Dropout Rate di-lock: {best_dropout}")

    # ==========================================================
    # TAHAP 4: MAX EPOCHS
    # ==========================================================
    stage_dir = os.path.join(OUTPUT_DIR, "tahap_4_epochs")
    variations = {
        'epochs=15': {'epochs': 15},
        'epochs=30': {'epochs': 30},
        'epochs=50': {'epochs': 50},
    }
    value_map = {
        'epochs=15': 15,
        'epochs=30': 30,
        'epochs=50': 50,
    }

    recap, best_epochs = run_stage(
        X, y, class_names,
        stage_name="Tahap 4: Max Epochs",
        stage_dir=stage_dir,
        param_variations=variations,
        base_params=current_best,
        value_map=value_map
    )
    current_best['epochs'] = best_epochs
    stage_recaps.append(("Tahap 4: Max Epochs", recap))
    print(f"\n  🔒 Max Epochs di-lock: {best_epochs}")

    # ==========================================================
    # TAHAP 5: DATA AUGMENTATION
    # ==========================================================
    stage_dir = os.path.join(OUTPUT_DIR, "tahap_5_augmentation")
    variations = {
        'aug=minimal': {'augmentation': 'minimal'},
        'aug=baseline': {'augmentation': 'baseline'},
        'aug=enhanced': {'augmentation': 'enhanced'},
    }
    value_map = {
        'aug=minimal': 'minimal',
        'aug=baseline': 'baseline',
        'aug=enhanced': 'enhanced',
    }

    recap, best_aug = run_stage(
        X, y, class_names,
        stage_name="Tahap 5: Data Augmentation",
        stage_dir=stage_dir,
        param_variations=variations,
        base_params=current_best,
        value_map=value_map
    )
    current_best['augmentation'] = best_aug
    stage_recaps.append(("Tahap 5: Data Augmentation", recap))
    print(f"\n  🔒 Augmentation di-lock: {best_aug}")

    # ==========================================================
    # RINGKASAN FINAL
    # ==========================================================
    total_time = time.time() - total_start

    print(f"\n{'='*60}")
    print(f"✅ SELURUH TUNING SELESAI!")
    print(f"   Total waktu: {total_time/60:.1f} menit ({total_time/3600:.1f} jam)")
    print(f"{'='*60}")
    print(f"\n🏆 KONFIGURASI TERBAIK (FINAL):")
    print(f"   Learning Rate  : {current_best['learning_rate']}")
    print(f"   Batch Size     : {current_best['batch_size']}")
    print(f"   Dropout Rate   : {current_best['dropout_rate']}")
    print(f"   Max Epochs     : {current_best['epochs']}")
    print(f"   Augmentation   : {current_best['augmentation']}")

    # Simpan ringkasan final
    save_final_summary(OUTPUT_DIR, current_best, total_time, stage_recaps)

    print(f"\n🎓 Semua hasil siap untuk Bab 4 skripsi Anda!")
    print(f"   Download folder '{OUTPUT_DIR}' dari Kaggle.")


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    main()
