import os
import numpy as np
import pandas as pd

# train ve etiket verilerini klasörden okuyalım
train_path = os.path.join("golub-data-set", "data_set_ALL_AML_train.csv")
actual_path = os.path.join("golub-data-set", "actual.csv")

df_train_raw = pd.read_csv(train_path)
df_actual = pd.read_csv(actual_path)

print("--- HAM VERİ BOYUTLARI ---")
print(f"Train Ham Boyut (Gen x Kolon): {df_train_raw.shape}")
print(f"Actual Etiket Boyutu           : {df_actual.shape}")

# call sütunlarını atıp sadece hasta verilerini (sayısal kolonları) ayıklayan fonksiyon
def extract_numeric_patient_data(df):
    gene_accessions = df['Gene Accession Number'].values

    # sadece kolon adı rakam olanları (1, 2, 38 vs.) al. call vb. metin içerenler eleniyor
    patient_cols = [c for c in df.columns if str(c).strip().isdigit()]

    # sayısal formata çevirip transpozesini alalım ki satırlar hasta, sütunlar gen olsun
    df_numeric = df[patient_cols].apply(pd.to_numeric, errors='coerce')
    df_transposed = df_numeric.T

    df_transposed.columns = gene_accessions
    df_transposed.index = df_transposed.index.astype(int)
    df_transposed.index.name = 'patient'
    return df_transposed

X_train = extract_numeric_patient_data(df_train_raw)

# actual dosyasındaki kanser tiplerini hasta id'leri üzerinden eşleştir
df_actual['patient'] = df_actual['patient'].astype(int)
labels_map = dict(zip(df_actual['patient'], df_actual['cancer']))

y_train = X_train.index.map(labels_map)

print("\n--- DÜZENLENMİŞ VERİ MATRİSLERİ ---")
print(f"Eğitim Seti (Hasta x Gen) : {X_train.shape} | Sınıflar: {pd.Series(y_train).value_counts().to_dict()}")

# veride eksik, sonsuz veya negatif değerler
print("\n--- VERİ KALİTESİ VE BOŞLUK RAPORU ---")
nan_count = X_train.isna().sum().sum()
inf_count = np.isinf(X_train.values).sum()
neg_count = (X_train.values < 0).sum()

total_entries = X_train.size

print(f"Toplam Hücre Sayısı        : {total_entries:,}")
print(f"Eksik Değer (NaN/Null)     : {nan_count}")
print(f"Sonsuz Değer (INF/-INF)    : {inf_count}")
print(f"Negatif İfade Değeri       : {neg_count} (Tüm değerlerin %{(100 * neg_count) / total_entries:.2f}'si)")
print(f"Minimum İfade Değeri       : {X_train.values.min()}")
print(f"Maksimum İfade Değeri      : {X_train.values.max()}")

# son olarak verinin bir kısmına göz atalım
print("\n--- İLK 5 HASTA VE İLK 5 GEN GÖRÜNÜMÜ ---")
print(X_train.iloc[:5, :5])

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# grafikleri biraz daha derli toplu göstermek için seaborn stili
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=100)

# çizim çok uzun sürmesin diye sadece eğitim setinden rastgele 100 bin değer çekip dağılıma bakıyoruz
raw_sample = np.random.choice(X_train.values.flatten(), size=100000, replace=False)
sns.histplot(raw_sample, bins=100, kde=True, ax=axes[0], color='#1f77b4')
axes[0].set_title("Ham Gen İfade Dağılımı", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Floresan Işıma Değeri")
axes[0].set_ylabel("Frekans")

# negatif değerleri (arka plan okumaları vs.) 0'a çekip log2(x + 1) ile veriyi normalize edelim
X_clipped = np.clip(X_train.values, a_min=0, a_max=None)
X_log2 = np.log2(X_clipped + 1.0)
log_sample = np.random.choice(X_log2.flatten(), size=100000, replace=False)

sns.histplot(log_sample, bins=100, kde=True, ax=axes[1], color='#2ca02c')
axes[1].set_title("Log2 Dönüşümü Sonrası Dağılım ($\log_2(x+1)$)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Log2 İfade Değeri")
axes[1].set_ylabel("Frekans")

plt.tight_layout()
plt.show()