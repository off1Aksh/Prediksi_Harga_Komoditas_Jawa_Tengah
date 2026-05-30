import pandas as pd
import os

# konfigurasi: (file_excel_2017_2020, file_excel_2021_2026, [(nama_komoditas, nama_kolom_csv), ...])
configs = [
    ("bawang_merah_putih_2017_2020.xlsx", "bawang_merah_putih_2021_2026.xlsx", [
        ("Bawang Merah", "bawang_merah"),
        ("Bawang Putih", "bawang_putih"),
    ]),
    ("beras_2017_2020.xlsx", "beras 2021-2026.xlsx", [
        ("Beras", "beras"),
    ]),
    ("cabai merah dan cabe rawit 2017-2020.xlsx", "cabai merah dan cabe rawit 2021-2026.xlsx", [
        ("Cabai Merah", "cabai_merah"),
        ("Cabai Rawit", "cabai_rawit"),
    ]),
    ("daging ayam 2017-2020.xlsx", "daging ayam 2021-2026.xlsx", [
        ("Daging Ayam", "daging_ayam"),
    ]),
    ("minyak goreng 2017-2020.xlsx", "minyak goreng 2021-2026.xlsx", [
        ("Minyak Goreng", "minyak_goreng"),
    ]),
    ("telur ayam 2017-2020.xlsx", "telur ayam 2021-2026.xlsx", [
        ("Telur Ayam", "telur_ayam"),
    ]),
]

# direktori sumber data asli
dataset_dir = r"D:\KULIAH\SKRIPSI\Dataset"

# direktori tempat script ini berada (untuk output)
script_dir = os.path.dirname(os.path.abspath(__file__))

all_dfs = []

for file_1, file_2, commodities in configs:
    path_1 = os.path.join(dataset_dir, file_1)
    path_2 = os.path.join(dataset_dir, file_2)
    
    print(f"Membaca {file_1} dan {file_2}...")
    df1 = pd.read_excel(path_1)
    df2 = pd.read_excel(path_2)

    for komoditas, col_name in commodities:
        # Proses df1 (2017-2020)
        row1 = df1[df1["Komoditas (Rp)"] == komoditas]
        row1 = row1.drop(columns=["No", "Komoditas (Rp)"])
        t1 = row1.T
        t1.columns = [col_name]
        t1 = t1.reset_index().rename(columns={"index": "tanggal"})
        
        # Proses df2 (2021-2026)
        row2 = df2[df2["Komoditas (Rp)"] == komoditas]
        row2 = row2.drop(columns=["No", "Komoditas (Rp)"])
        t2 = row2.T
        t2.columns = [col_name]
        t2 = t2.reset_index().rename(columns={"index": "tanggal"})
        
        # Gabungkan secara vertikal
        combined = pd.concat([t1, t2], ignore_index=True)
        all_dfs.append(combined)
        print(f"[OK] {komoditas} tergabung: {len(combined)} baris")

# df pertama akan jadi basis untuk tanggal (karena urutan sama untuk semua komoditas)
tanggal = all_dfs[0][["tanggal"]]
commodity_cols = [df.drop(columns=["tanggal"]) for df in all_dfs]
result = pd.concat([tanggal] + commodity_cols, axis=1)

# hapus baris yang memiliki missing value ("-")
before = len(result)
result = result[~(result == "-").any(axis=1)].reset_index(drop=True)
after = len(result)
print(f"\nMenghapus {before - after} baris dengan missing value")
print(f"Hasil gabungan: {after} baris x {result.shape[1]} kolom")
print(f"Kolom: {result.columns.tolist()}")

output_path = os.path.join(script_dir, "dataset_harga_komoditas_2017_2026.csv")
result.to_csv(output_path, index=False)
print(f"\n[OK] Dataset berhasil disimpan ke: {output_path}")
print(result.head())
print(result.tail())
