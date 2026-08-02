import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

# Set paths
base_dir = r"d:\KULIAH\SKRIPSI\Dataset\harga"

files = [
    ("Beras", "beras_2017_2020.xlsx", "beras 2021-2026.xlsx"),
    ("Bawang Merah", "bawang_merah_putih_2017_2020.xlsx", "bawang_merah_putih_2021_2026.xlsx"),
    ("Bawang Putih", "bawang_merah_putih_2017_2020.xlsx", "bawang_merah_putih_2021_2026.xlsx"),
    ("Cabai Merah", "cabai merah dan cabe rawit 2017-2020.xlsx", "cabai merah dan cabe rawit 2021-2026.xlsx"),
    ("Cabai Rawit", "cabai merah dan cabe rawit 2017-2020.xlsx", "cabai merah dan cabe rawit 2021-2026.xlsx"),
    ("Daging Ayam", "daging ayam 2017-2020.xlsx", "daging ayam 2021-2026.xlsx"),
    ("Minyak Goreng", "minyak goreng 2017-2020.xlsx", "minyak goreng 2021-2026.xlsx"),
    ("Telur Ayam", "telur ayam 2017-2020.xlsx", "telur ayam 2021-2026.xlsx")
]

all_data = {}

def clean_value(val):
    if pd.isna(val) or val == '-':
        return np.nan
    if isinstance(val, str):
        val = val.replace(',', '')
        # Jika nilai mengandung titik dan itu bukan desimal (karena harga ribuan), hapus titik
        if val.count('.') == 1 and len(val.split('.')[1]) == 3:
            val = val.replace('.', '')
    try:
        return float(val)
    except:
        return np.nan

for name, f1, f2 in files:
    try:
        df1 = pd.read_excel(os.path.join(base_dir, f1))
        df2 = pd.read_excel(os.path.join(base_dir, f2))
        
        # Identify row for the commodity
        col_name_1 = [c for c in df1.columns if 'Komoditas' in str(c)][0]
        row1_match = df1[df1[col_name_1].astype(str).str.contains(name, case=False, na=False)]
        row1 = row1_match.iloc[0] if len(row1_match) > 0 else df1.iloc[0]
        
        col_name_2 = [c for c in df2.columns if 'Komoditas' in str(c)][0]
        row2_match = df2[df2[col_name_2].astype(str).str.contains(name, case=False, na=False)]
        row2 = row2_match.iloc[0] if len(row2_match) > 0 else df2.iloc[0]
        
        # Extract date columns
        dates1 = [c for c in df1.columns if isinstance(c, (pd.Timestamp, datetime.datetime)) or '/' in str(c) or '-' in str(c)]
        dates2 = [c for c in df2.columns if isinstance(c, (pd.Timestamp, datetime.datetime)) or '/' in str(c) or '-' in str(c)]
        
        ts1 = row1[dates1]
        ts2 = row2[dates2]
        
        def parse_dates(idx):
            cleaned = []
            for i in idx:
                if isinstance(i, str):
                    cleaned.append(i.replace(' ', ''))
                else:
                    cleaned.append(i)
            # Use coerce so unparseable columns become NaT
            return pd.to_datetime(cleaned, dayfirst=True, errors='coerce')

        ts1.index = parse_dates(ts1.index)
        ts2.index = parse_dates(ts2.index)
        
        # Clean values
        ts1 = ts1.apply(clean_value)
        ts2 = ts2.apply(clean_value)
        
        # Combine
        ts = pd.concat([ts1, ts2])
        ts = ts[ts.index.notna()]
        ts = ts.sort_index()
        ts = ts[~ts.index.duplicated(keep='first')]
        
        all_data[name] = ts
        print(f"Berhasil memproses {name}: {len(ts)} titik data")
    except Exception as e:
        print(f"Error pada {name}: {e}")

df_all = pd.DataFrame(all_data)

out_folder = r'd:\KULIAH\SKRIPSI\grafik_komoditas'
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

# 1. Simpan sebagai 8 grafik terpisah
if not df_all.empty:
    for col in df_all.columns:
        plt.figure(figsize=(10, 5))
        plt.plot(df_all.index, df_all[col], color='#1f77b4', linewidth=1.5)
        plt.title(f"Fluktuasi Harga {col} di Jawa Tengah (2017-2026)", fontsize=14)
        plt.xlabel("Tahun", fontsize=12)
        plt.ylabel("Harga (Rp)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        safe_name = col.replace(" ", "_").lower()
        out_path = os.path.join(out_folder, f'fluktuasi_{safe_name}.png')
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Plot tersimpan: {out_path}")

# 2. Simpan sebagai 1 gambar gabungan (grid 4x2)
if not df_all.empty and len(df_all.columns) <= 8:
    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    axes = axes.flatten()
    for i, col in enumerate(df_all.columns):
        ax = axes[i]
        ax.plot(df_all.index, df_all[col], color='teal', linewidth=1.5)
        ax.set_title(f"{col}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Tahun", fontsize=10)
        ax.set_ylabel("Harga (Rp)", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Sembunyikan subplot kosong jika komoditas kurang dari 8
    for j in range(len(df_all.columns), 8):
        fig.delaxes(axes[j])

    plt.suptitle("Fluktuasi Harga 8 Komoditas Pangan di Jawa Tengah (2017-2026)", fontsize=18, y=1.01)
    plt.tight_layout()
    out_path_grid = os.path.join(out_folder, 'fluktuasi_semua_komoditas_grid.png')
    plt.savefig(out_path_grid, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot gabungan (grid) tersimpan: {out_path_grid}")
