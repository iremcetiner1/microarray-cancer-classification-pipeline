import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# İLK ADIM = Eksik/INF değerler, negatif ve uç değerler, call (P/A/M) sütunlarının anlamı, 
# etiket–sütun eşleştirmesinin doğrulanması.

dataset_dir = "golub-data-set"
train_df = pd.read_csv(os.path.join(dataset_dir, "data_set_ALL_AML_train.csv"))
actual_df = pd.read_csv(os.path.join(dataset_dir, "actual.csv"))
test_df = pd.read_csv(os.path.join(dataset_dir, "data_set_ALL_AML_independent.csv"))

print("--- 1. HAM VERİ BOYUTLARI ---")
print(f"Train Ham Boyut (Gen x Kolon) : {train_df.shape}")
print(f"Test Ham Boyut (Gen x Kolon)  : {test_df.shape}")
print(f"Actual Etiket Boyutu          : {actual_df.shape}")

# 2. Hem test hem train verisindeki call (P/A/M) sütunlarını ayrıştırıyoruz
train_call_cols = [c for c in train_df.columns if 'call' in c.lower()]
test_call_cols = [c for c in test_df.columns if 'call' in c.lower()]

train_calls = pd.Series(train_df[train_call_cols].values.flatten()).value_counts(normalize=True) * 100
test_calls = pd.Series(test_df[test_call_cols].values.flatten()).value_counts(normalize=True) * 100

print("\n--- 2. AFFYMETRIX CALL (P/A/M) DAĞILIM ORANLARI (%) ---")
print("Eğitim Call (%) Dağılımı:\n", train_calls.round(2).to_dict())
print("Test Call (%) Dağılımı  :\n", test_calls.round(2).to_dict())

# 3. Hasta verilerini ayıklama, sayısallaştırma ve N x p (Hasta x Gen) formatına çevirme
def extract_numeric_patient_data(df):
    gene_accessions = df['Gene Accession Number'].values
    
    # Sadece sütun adı sayı (hasta ID) olanları güvenli şekilde seç (2. koddaki yöntem)
    patient_cols = [c for c in df.columns if str(c).strip().isdigit()]
    
    # Sayısal formata çevirip transpoz alalım (Satırlar: Hasta, Sütunlar: Gen)
    df_numeric = df[patient_cols].apply(pd.to_numeric, errors='coerce')
    df_transposed = df_numeric.T
    
    df_transposed.columns = gene_accessions
    df_transposed.index = df_transposed.index.astype(int)
    df_transposed.index.name = 'patient'
    
    # Hasta ID'lerini küçükten büyüğe sırala (1. koddaki yöntem)
    return df_transposed.sort_index()

X_train_raw = extract_numeric_patient_data(train_df)
X_test_raw = extract_numeric_patient_data(test_df)

# 4. Etiket-Sütun (actual.csv) eşleştirmesi ve doğrulaması
actual_df['patient'] = actual_df['patient'].astype(int)
y_train = actual_df.set_index('patient').loc[X_train_raw.index, 'cancer']
y_test = actual_df.set_index('patient').loc[X_test_raw.index, 'cancer']

assert (X_train_raw.index == y_train.index).all(), "Eğitim indeks eşleşme hatası!"
assert (X_test_raw.index == y_test.index).all(), "Test indeks eşleşme hatası!"

print("\n--- 3. DÜZENLENMİŞ VERİ MATRİSLERİ VE SINIF DAĞILIMI ---")
print(f"Eğitim Seti (Hasta x Gen) : {X_train_raw.shape} | Sınıflar: {y_train.value_counts().to_dict()}")
print(f"Test Seti (Hasta x Gen)   : {X_test_raw.shape} | Sınıflar: {y_test.value_counts().to_dict()}")

# 5. Eksik, INF, Negatif ve Uç Değer Kontrolleri (Alttaki kodun detaylı rapor formatıyla)
def print_quality_report(X, dataset_name):
    nan_count = X.isna().sum().sum()
    inf_count = np.isinf(X.values).sum()
    neg_count = (X.values < 0).sum()
    total_entries = X.size
    
    print(f"\n--- {dataset_name} VERİ KALİTESİ VE BOŞLUK RAPORU ---")
    print(f"Toplam Hücre Sayısı        : {total_entries:,}")
    print(f"Eksik Değer (NaN/Null)     : {nan_count}")
    print(f"Sonsuz Değer (INF/-INF)    : {inf_count}")
    print(f"Negatif İfade Değeri       : {neg_count:,} (Tüm değerlerin %{(100 * neg_count) / total_entries:.2f}'si)")
    print(f"Minimum İfade Değeri       : {X.values.min()}")
    print(f"Maksimum İfade Değeri      : {X.values.max()}")

print_quality_report(X_train_raw, "EĞİTİM SETİ (TRAIN)")
print_quality_report(X_test_raw, "BAĞIMSIZ TEST SETİ (TEST)")

#CLASS DAĞILIMI 
# Sınıf dağılımı tablosu
class_dist = pd.DataFrame({
    'Eğitim (n=38)': pd.Series(y_train).value_counts(),
    'Test (n=34)': pd.Series(y_test).value_counts()
})
print("\nSınıf Dağılımı:\n", class_dist)

# Kırpma (20 - 16.000) ve Log10 Dönüşümü Fonksiyonu
def clip_and_log(df, lower=20, upper=16000):
    df_clipped = df.clip(lower=lower, upper=upper)
    return np.log10(df_clipped)

X_train_log = clip_and_log(X_train_raw)
X_test_log = clip_and_log(X_test_raw)

# Görselleştirme: Dönüşüm Öncesi vs Sonrası İfade Dağılımları
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(X_train_raw.values.ravel(), bins=100, ax=axes[0], color='crimson', kde=True)
axes[0].set_title("Ham Gen İfade Dağılımı (Çarpık & Negatif Değerli)")
axes[0].set_xlabel("Floresan Yoğunluğu")

sns.histplot(X_train_log.values.ravel(), bins=60, ax=axes[1], color='teal', kde=True)
axes[1].set_title("Kırpma [20-16.000] ve Log10 Dönüşümü Sonrası Dağılım")
axes[1].set_xlabel("Log10(İfade Değeri)")

plt.tight_layout()
plt.show()


#BATCH ETKİSİ
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 1. Genel Yoğunluk Karşılaştırması (KDE)
sns.kdeplot(X_train_log.values.ravel(), label='Eğitim Seti (38)', ax=axes[0], fill=True, alpha=0.4)
sns.kdeplot(X_test_log.values.ravel(), label='Bağımsız Test Seti (34)', ax=axes[0], fill=True, alpha=0.4)
axes[0].set_title("Eğitim vs Test Seti: Genel Dağılım Karşılaştırması (Batch Etkisi)")
axes[0].set_xlabel("Log10(İfade)")
axes[0].legend()

# 2. Örnek Bazlı Medyan/IQR Karşılaştırması (İlk 15'er örnek)
combined_samples = pd.concat([
    X_train_log.iloc[:15].T.assign(Set='Train'),
    X_test_log.iloc[:15].T.assign(Set='Test')
], axis=1)

# Kutu grafiği için veriyi eritme (melt)
melt_train = X_train_log.iloc[:15].T.melt(var_name='Patient', value_name='Log_Expr')
melt_train['Set'] = 'Eğitim'
melt_test = X_test_log.iloc[:15].T.melt(var_name='Patient', value_name='Log_Expr')
melt_test['Set'] = 'Test'
melt_df = pd.concat([melt_train, melt_test])

sns.boxplot(data=melt_df, x='Patient', y='Log_Expr', hue='Set', ax=axes[1], fliersize=1)
axes[1].set_title("Hasta Bazlı İfade Dağılımları (Örneklem Karşılaştırması)")
axes[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.show()

# İstatistiksel Batch Farkı Testi (Kolmogorov-Smirnov)
ks_stat, ks_pval = stats.ks_2samp(X_train_log.values.ravel(), X_test_log.values.ravel())
print(f"Eğitim vs Test Dağılım Farkı (KS Testi): İstatistik={ks_stat:.4f}, p-değeri={ks_pval:.4e}")

