import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

# 1. Klasör ve dosya yolu ayarları
dataset_dir = "breast-data-set"
input_path = os.path.join(dataset_dir, "Breast_GSE45827.csv")

# 2. Veri setini yükleme
print("Veri seti yükleniyor...")
base_df = pd.read_csv(input_path)

# 3. Train (%80) ve Test (%20) olarak sınıf oranlarını koruyarak (Stratified) ayırma
train_df, test_df = train_test_split(
    base_df, test_size=0.20, random_state=42, stratify=base_df["type"]
)

# 4. Kayıt yolları
train_path = os.path.join(dataset_dir, "train_GSE45827.csv")
test_path = os.path.join(dataset_dir, "test_GSE45827.csv")

# 5. Dosyaları CSV olarak diske yazma
print("Dosyalar diske yazılıyor...")
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

# 6. Diske yazılma doğrulaması
train_saved = os.path.exists(train_path)
test_saved = os.path.exists(test_path)

print(f"\nDosya Kontrolleri:")
print(f"- Train dosyası kaydedildi mi?: {train_saved} ({train_path})")
print(f"- Test dosyası kaydedildi mi? : {test_saved} ({test_path})\n")

# 7. Bilgilendirme ve sınıf dağılımları
print(f"Toplam Veri Sayısı : {len(base_df)}")
print(
    f"Eğitim Seti Boyutu : {train_df.shape} (Örnek sayısı: {len(train_df)})"
)
print(f"Test Seti Boyutu   : {test_df.shape} (Örnek sayısı: {len(test_df)})\n")

print("--- Eğitim Seti Sınıf Dağılımı ---")
print(train_df["type"].value_counts())

print("\n--- Test Seti Sınıf Dağılımı ---")
print(test_df["type"].value_counts())