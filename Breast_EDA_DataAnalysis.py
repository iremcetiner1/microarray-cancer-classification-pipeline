import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
dataset_dir = "breast-data-set"
train_df = pd.read_csv(os.path.join(dataset_dir, "train_GSE45827.csv"))
test_df = pd.read_csv(os.path.join(dataset_dir, "test_GSE45827.csv"))
meta_cols = ["samples", "type"]
gene_cols = [c for c in train_df.columns if c not in meta_cols]

X_train = train_df[gene_cols].copy()
y_train = train_df["type"].copy()

X_test = test_df[gene_cols].copy()
y_test = test_df["type"].copy()
print("=" * 60)
print(f"Train Gen Matrisi Boyutu: {X_train.shape}")
print(f"Test Gen Matrisi Boyutu : {X_test.shape}")
print("=" * 60)
# 1. NaN / INF check
print("\n[1] EKSİK VE INF DEĞER KONTROLÜ:")
null_train = X_train.isnull().sum().sum()
null_test = X_test.isnull().sum().sum()
inf_train = np.isinf(X_train.values).sum()
inf_test = np.isinf(X_test.values).sum()
print(f" - Train Eksik Değer (NaN): {null_train} | INF: {inf_train}")
print(f" - Test Eksik Değer (NaN) : {null_test} | INF: {inf_test}")
# 2. negative and IQR
print("\n[2] NEGATİF VE UÇ DEĞER ANALİZİ:")
neg_train_count = (X_train.values < 0).sum()
print(
    f" - Negatif Değer Sayısı (Train): {neg_train_count} ({neg_train_count / X_train.size * 100:.4f}%)"
)
q1 = np.percentile(X_train.values, 25)
q3 = np.percentile(X_train.values, 75)
iqr = q3 - q1
alt_esik = q1 - 1.5 * iqr
ust_esik = q3 + 1.5 * iqr

outliers_low = (X_train.values < alt_esik).sum()
outliers_high = (X_train.values > ust_esik).sum()
print(
    f" - Alt Uç Eşik ({alt_esik:.3f}) Altındaki Değerler: {outliers_low} ({outliers_low / X_train.size * 100:.2f}%)"
)
print(
    f" - Üst Uç Eşik ({ust_esik:.3f}) Üstündeki Değerler: {outliers_high} ({outliers_high / X_train.size * 100:.2f}%)"
)
#check column
print("\n[3 & 4] SÜTUN VE ETİKET KONTROLÜ:")
pam_cols = [c for c in train_df.columns if c.endswith("_call") or c in ["P", "A", "M"]]
print(
    f" - Golub benzeri P/A/M kalite sütunları var mı?: {'EVET' if pam_cols else 'HAYIR (Sadece gen ekspresyon değerleri mevcut)'}"
)
print(f" - Örnek Kimlik Sütunu: 'samples'")
print(f" - Hedef Sınıf Sütunu : 'type'")
print(
    f" - Gen Sütun Formatı Örnekleri: {gene_cols[:3]} ... (Toplam {len(gene_cols)} gen probu)"
)
# 5. class dist and statistics

print("\n[5] SINIF DAĞILIMI (TRAIN vs TEST):")
class_dist = pd.DataFrame(
    {
        "Train Sayı": y_train.value_counts(),
        "Train Oran (%)": (y_train.value_counts(normalize=True) * 100).round(
            2
        ),
        "Test Sayı": y_test.value_counts(),
        "Test Oran (%)": (y_test.value_counts(normalize=True) * 100).round(2),
    }
)
print(class_dist)

# Gen ekspresyon matrisi genel betimsel istatistikler
val_min = np.min(X_train.values)
val_max = np.max(X_train.values)
val_mean = np.mean(X_train.values)
val_median = np.median(X_train.values)
val_std = np.std(X_train.values)

print(f"\nGen Ekspresyon Değerleri Özeti (Train):")
print(
    f" - Min: {val_min:.4f} | Max: {val_max:.4f} | Medyan: {val_median:.4f} | Ortalama: {val_mean:.4f} | Std: {val_std:.4f}"
)


# 6. LOG2 transform decision and visualization
print("\n[6] LOG DÖNÜŞÜMÜ DEĞERLENDİRMESİ:")

already_log = val_max <= 20 and val_min >= 0 # Karar kuralı: Max değer ~16 civarındaysa zaten log2 ölçeğindedir.

if already_log:
  print(
      f" >> Karar: Maksimum değer ({val_max:.2f}) <= 20 olduğundan veri seti ZATEN log2 ölçeğindedir."
  )
  print(" >> Tekrar log dönüşümü UYGULANMAMALIDIR.")
else:
  print(
      f" >> Karar: Maksimum değer ({val_max:.2f}) > 20 olduğu için log2 dönüşümü GEREKLİDİR."
  )

# Karşılaştırma grafiği (Mevcut Dağılım vs Log2 Uygulanmış Hipotetik Dağılım)
sample_values = X_train.values.flatten()
sample_subset = np.random.choice(sample_values, size=100000, replace=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(sample_subset, bins=80, kde=True, ax=axes[0], color="royalblue")
axes[0].set_title(
    f"Mevcut Ekspresyon Dağılımı\n(Min: {val_min:.2f}, Max: {val_max:.2f})"
)
axes[0].set_xlabel("Ekspresyon Değeri")
# Hipotetik log2(x + 1)
log_transformed = np.log2(np.maximum(sample_subset, 0) + 1)
sns.histplot(
    log_transformed, bins=80, kde=True, ax=axes[1], color="forestgreen"
)
axes[1].set_title(
    f"Hipotetik log2(x + 1) Dağılımı\n(Min: {log_transformed.min():.2f}, Max: {log_transformed.max():.2f})"
)
axes[1].set_xlabel("log2(Ekspresyon + 1)")

plt.tight_layout()
plt.savefig(
    os.path.join("Breast_log_donusumu_karsilastirma.png"), dpi=300
)
plt.show()

# ----------------------------------------------------
# 7. BOYUT İNDİRGEME: PCA, t-SNE, UMAP
# ----------------------------------------------------
print("\n[7] BOYUT İNDİRGEME ÇALIŞTIRILIYOR...")

# 7.1 PCA (Açıklanan Varyans Hesabı)
# Maksimum bileşen sayısı min(ornek_sayisi, gen_sayisi) = len(X_train)
n_components = min(X_train.shape[0], 50)
pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train)

exp_var = pca.explained_variance_ratio_ * 100
cum_var = np.cumsum(exp_var)

print(
    f" - PCA 1. Bileşen Varyansı : %{exp_var[0]:.2f}"
)
print(
    f" - PCA 2. Bileşen Varyansı : %{exp_var[1]:.2f}"
)
print(
    f" - İlk 10 Bileşen Kümülatif Varyans: %{cum_var[9]:.2f}"
)
print(
    f" - İlk 50 Bileşen Kümülatif Varyans: %{cum_var[-1]:.2f}"
)

# 7.2 t-SNE (Gürültüyü azaltmak için ilk 50 PCA bileşeni üzerinden)
tsne = TSNE(
    n_components=2, perplexity=15, max_iter=1000, random_state=42, init="pca"
)
X_train_tsne = tsne.fit_transform(X_train_pca)

# 7.3 UMAP
try:
  import umap
except ImportError:
  umap = None
if umap is not None:
  reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
  X_train_umap = reducer.fit_transform(X_train_pca)
else:
  X_train_umap = None
  print(
      " - UMAP kütüphanesi bulunamadı (`pip install umap-learn` ile kurabilirsiniz)."
  )

# --- Görselleştirme: Scree Plot & 2D İndirgemeler ---
fig = plt.figure(figsize=(18, 10))

# 1. Scree Plot (Kümülatif Varyans)
ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(range(1, n_components + 1), cum_var, marker="o", color="darkred")
ax1.axhline(
    y=80, color="gray", linestyle="--", label="%80 Açıklanan Varyans Eşiği"
)
ax1.set_title("PCA Açıklanan Kümülatif Varyans (Scree Plot)")
ax1.set_xlabel("Bileşen Sayısı")
ax1.set_ylabel("Kümülatif Varyans (%)")
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. PCA 2D Dağılımı
ax2 = fig.add_subplot(2, 2, 2)
sns.scatterplot(
    x=X_train_pca[:, 0],
    y=X_train_pca[:, 1],
    hue=y_train,
    palette="Set1",
    s=70,
    ax=ax2,
)
ax2.set_title(
    f"PCA 2D İzdüşümü (PC1: %{exp_var[0]:.1f}, PC2: %{exp_var[1]:.1f})"
)
ax2.set_xlabel("PC 1")
ax2.set_ylabel("PC 2")
ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# 3. t-SNE 2D Dağılımı
ax3 = fig.add_subplot(2, 2, 3)
sns.scatterplot(
    x=X_train_tsne[:, 0],
    y=X_train_tsne[:, 1],
    hue=y_train,
    palette="Set1",
    s=70,
    ax=ax3,
)
ax3.set_title("t-SNE 2D İzdüşümü (Perplexity=15)")
ax3.set_xlabel("t-SNE 1")
ax3.set_ylabel("t-SNE 2")
ax3.legend().remove()

# 4. UMAP 2D Dağılımı
if X_train_umap is not None:
  ax4 = fig.add_subplot(2, 2, 4)
  sns.scatterplot(
      x=X_train_umap[:, 0],
      y=X_train_umap[:, 1],
      hue=y_train,
      palette="Set1",
      s=70,
      ax=ax4,
  )
  ax4.set_title("UMAP 2D İzdüşümü (n_neighbors=15)")
  ax4.set_xlabel("UMAP 1")
  ax4.set_ylabel("UMAP 2")
  ax4.legend().remove()

plt.tight_layout()
plt.savefig(os.path.join("Breast_boyut_indirgeme_analizi.png"), dpi=300)
plt.show()

print(
    f"\nAnaliz tamamlandı. Grafikler '{dataset_dir}' klasörüne kaydedildi."
)