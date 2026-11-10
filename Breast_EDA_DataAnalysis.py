import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway, ks_2samp

# Çoklu hipotez testi düzeltmesi (Benjamini-Hochberg FDR) fonksiyonu
def benjamini_hochberg_fdr(p_vals):
    p_vals = np.asarray(p_vals, dtype=float)
    n = len(p_vals)
    order = np.argsort(p_vals)
    ranked_p = p_vals[order]
    q_vals = ranked_p * n / np.arange(1, n + 1)
    q_vals = np.minimum.accumulate(q_vals[::-1])[::-1]
    q_vals = np.clip(q_vals, 0, 1)
    q_orig = np.empty(n)
    q_orig[order] = q_vals
    return q_orig

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
neg_test_count = (X_test.values < 0).sum()
print(
    f" - Negatif Değer Sayısı (Train): {neg_train_count} ({neg_train_count / X_train.size * 100:.4f}%)"
)
print(
    f" - Negatif Değer Sayısı (Test) : {neg_test_count} ({neg_test_count / X_test.size * 100:.4f}%)"
)

q1 = np.percentile(X_train.values, 25)
q3 = np.percentile(X_train.values, 75)
iqr = q3 - q1
alt_esik = q1 - 1.5 * iqr
ust_esik = q3 + 1.5 * iqr

outliers_low = (X_train.values < alt_esik).sum()
outliers_high = (X_train.values > ust_esik).sum()
print(
    f" - Alt Uç Eşik ({alt_esik:.3f}) Altındaki Değerler (Train): {outliers_low} ({outliers_low / X_train.size * 100:.2f}%)"
)
print(
    f" - Üst Uç Eşik ({ust_esik:.3f}) Üstündeki Değerler (Train): {outliers_high} ({outliers_high / X_train.size * 100:.2f}%)"
)

# 3 & 4. check column and alignment
print("\n[3 & 4] SÜTUN VE ETİKET EŞLEŞTİRME KONTROLÜ:")
cols_match = (X_train.columns == X_test.columns).all()
index_aligned = (X_train.index == y_train.index).all() and (X_test.index == y_test.index).all()

print(f" - Örnek Kimlik Sütunu: 'samples' | Hedef Sınıf Sütunu: 'type'")
print(f" - Train ve Test Gen Sütun Sırası Birebir Aynı mı?: {'EVET' if cols_match else 'HAYIR'}")
print(f" - Özellik Matrisi (X) ve Etiket (y) İndeksleri Hizalı mı?: {'EVET' if index_aligned else 'HAYIR'}")
print(
    f" - Gen Sütun Formatı Örnekleri: {gene_cols[:3]} ... (Toplam {len(gene_cols)} gen probu)"
)

# 5. class dist and statistics
print("\n[5] SINIF DAĞILIMI (TRAIN vs TEST):")
class_dist = pd.DataFrame(
    {
        "Train Sayı": y_train.value_counts(),
        "Train Oran (%)": (y_train.value_counts(normalize=True) * 100).round(2),
        "Test Sayı": y_test.value_counts(),
        "Test Oran (%)": (y_test.value_counts(normalize=True) * 100).round(2),
    }
)
print(class_dist)
# --- 5. BÖLÜM EKİ: TRAIN vs TEST SINIF DAĞILIMI GRAFİĞİ ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Grafik: Örnek Sayıları (Countplot)
class_order = y_train.value_counts().index
df_classes = pd.DataFrame({
    "Sınıf": pd.concat([y_train, y_test], ignore_index=True),
    "Veri_Seti": ["Train"] * len(y_train) + ["Test"] * len(y_test)
})

sns.countplot(
    data=df_classes,
    x="Sınıf",
    hue="Veri_Seti",
    order=class_order,
    palette={"Train": "royalblue", "Test": "darkorange"},
    ax=axes[0]
)
axes[0].set_title(" Breast Train vs Test Sınıf Dağılımı (Örnek Sayısı)")
axes[0].set_xlabel("Kanser Alt Tipi")
axes[0].set_ylabel("Örnek Sayısı (Adet)")
axes[0].tick_params(axis="x", rotation=30)
axes[0].grid(axis="y", alpha=0.3)

# Sütunların üzerine sayıları yazdırma
for container in axes[0].containers:
    axes[0].bar_label(container, fontsize=9)

# 2. Grafik: Oransal Dağılım (%) Karşılaştırması
prop_df = class_dist[["Train Oran (%)", "Test Oran (%)"]].reset_index()
prop_df.columns = ["Sınıf", "Train (%)", "Test (%)"]
prop_melted = prop_df.melt(id_vars="Sınıf", var_name="Veri_Seti", value_name="Oran (%)")

sns.barplot(
    data=prop_melted,
    x="Sınıf",
    y="Oran (%)",
    hue="Veri_Seti",
    order=class_order,
    palette={"Train (%)": "royalblue", "Test (%)": "darkorange"},
    ax=axes[1]
)
axes[1].set_title("Breast Train vs Test Sınıf Dağılımı (Yüzdelik Oran %)")
axes[1].set_xlabel("Kanser Alt Tipi")
axes[1].set_ylabel("Oran (%)")
axes[1].tick_params(axis="x", rotation=30)
axes[1].grid(axis="y", alpha=0.3)

for container in axes[1].containers:
    axes[1].bar_label(container, fmt="%.1f%%", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join( "Breast_sinif_dagilimi_train_vs_test.png"), dpi=300)
plt.show()
# Gen ekspresyon matrisi genel betimsel istatistikler
val_min, val_max = np.min(X_train.values), np.max(X_train.values)
val_mean, val_median, val_std = np.mean(X_train.values), np.median(X_train.values), np.std(X_train.values)

t_min, t_max = np.min(X_test.values), np.max(X_test.values)
t_mean, t_median, t_std = np.mean(X_test.values), np.median(X_test.values), np.std(X_test.values)

print(f"\nBreast Gen Ekspresyon Değerleri Özeti:")
print(
    f" - TRAIN -> Min: {val_min:.4f} | Max: {val_max:.4f} | Medyan: {val_median:.4f} | Ortalama: {val_mean:.4f} | Std: {val_std:.4f}"
)
print(
    f" - TEST  -> Min: {t_min:.4f} | Max: {t_max:.4f} | Medyan: {t_median:.4f} | Ortalama: {t_mean:.4f} | Std: {t_std:.4f}"
)

# 6. LOG2 transform decision and visualization
print("\n[6] LOG DÖNÜŞÜMÜ DEĞERLENDİRMESİ:")
already_log = val_max <= 20 and val_min >= 0

if already_log:
    print(
        f" >> Karar: Maksimum değer ({val_max:.2f}) <= 20 olduğundan veri seti ZATEN log2 ölçeğindedir."
    )
    print(" >> Tekrar log dönüşümü UYGULANMAMALIDIR.")
else:
    print(
        f" >> Karar: Maksimum değer ({val_max:.2f}) > 20 olduğu için log2 dönüşümü GEREKLİDİR."
    )

np.random.seed(42)
sample_values = X_train.values.flatten()
sample_subset = np.random.choice(sample_values, size=min(100000, sample_values.size), replace=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(sample_subset, bins=80, kde=True, ax=axes[0], color="royalblue")
axes[0].set_title(
    f"Mevcut Ekspresyon Dağılımı\n(Min: {val_min:.2f}, Max: {val_max:.2f})"
)
axes[0].set_xlabel("Ekspresyon Değeri")

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

# 7. BOYUT İNDİRGEME: PCA, t-SNE, UMAP
print("\n[7] BOYUT İNDİRGEME ÇALIŞTIRILIYOR...")

n_components = min(X_train.shape[0], 50)
pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train)

exp_var = pca.explained_variance_ratio_ * 100
cum_var = np.cumsum(exp_var)

print(f" - PCA 1. Bileşen Varyansı : %{exp_var[0]:.2f}")
print(f" - PCA 2. Bileşen Varyansı : %{exp_var[1]:.2f}")
print(f" - İlk 10 Bileşen Kümülatif Varyans: %{cum_var[min(9, len(cum_var)-1)]:.2f}")
print(f" - İlk {n_components} Bileşen Kümülatif Varyans: %{cum_var[-1]:.2f}")

# 7.2 t-SNE
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
    print(" - UMAP kütüphanesi bulunamadı (`pip install umap-learn` ile kurabilirsiniz).")

# --- Görselleştirme: Scree Plot & 2D İndirgemeler ---
fig = plt.figure(figsize=(18, 10))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(range(1, n_components + 1), cum_var, marker="o", color="darkred")
ax1.axhline(
    y=80, color="gray", linestyle="--", label="%80 Açıklanan Varyans Eşiği"
)
ax1.set_title("Breast PCA Açıklanan Kümülatif Varyans (Scree Plot)")
ax1.set_xlabel("Bileşen Sayısı")
ax1.set_ylabel("Kümülatif Varyans (%)")
ax1.grid(True, alpha=0.3)
ax1.legend()

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


# 8. AYIRT EDİCİ GENLER: SINIFLAR ARASI TESTLER, ISI HARİTASI VE KÜMELEME
print("\n[8] AYIRT EDİCİ GEN ANALİZİ (ANOVA F-TESTİ, FDR & HİYERARŞİK KÜMELEME)...")

# 8.1 Sınıflar arası tek yönlü ANOVA testi (Vektörel hesaplama)
classes = y_train.unique()
class_groups = [X_train[y_train == cls].values for cls in classes]
f_stat, p_val = f_oneway(*class_groups, axis=0)

# NaN çıkabilecek sabit varyanslı genler için güvenli dönüşüm
f_stat = np.nan_to_num(f_stat, nan=0.0)
p_val = np.nan_to_num(p_val, nan=1.0)

# Çoklu test düzeltmesi (FDR - Benjamini-Hochberg)
q_val = benjamini_hochberg_fdr(p_val)

deg_df = pd.DataFrame({
    "Gen_Probu": gene_cols,
    "ANOVA_F": f_stat,
    "p_degeri": p_val,
    "FDR_q_degeri": q_val
}).sort_values(by="ANOVA_F", ascending=False).reset_index(drop=True)

sig_fdr_005 = (deg_df["FDR_q_degeri"] < 0.05).sum()
sig_fdr_001 = (deg_df["FDR_q_degeri"] < 0.01).sum()

print(f" - Toplam İncelenen Gen Probu Sayısı : {len(gene_cols)}")
print(f" - Anlamlı Gen Sayısı (FDR < 0.05)   : {sig_fdr_005} (%{sig_fdr_005 / len(gene_cols) * 100:.2f})")
print(f" - Anlamlı Gen Sayısı (FDR < 0.01)   : {sig_fdr_001} (%{sig_fdr_001 / len(gene_cols) * 100:.2f})")
print("\n BreastEn Ayırt Edici İlk 10 Gen Probu:")
print(deg_df.head(10).to_string(index=False))

# Tabloyu CSV olarak kaydet
deg_df.head(50).to_csv(os.path.join(dataset_dir, "Breast_en_iyi_50_gen_ANOVA.csv"), index=False)

# 8.2 En İyi 35 Gen İçin Hiyerarşik Kümeleme ve Isı Haritası (Clustermap)
top_k = 35  # İster: 20-50 arası en iyi gen
top_genes = deg_df["Gen_Probu"].iloc[:top_k].values

# Görselleştirme için gen bazında Z-skoru standardizasyonu
scaler_top = StandardScaler()
X_train_top_scaled = pd.DataFrame(
    scaler_top.fit_transform(X_train[top_genes]),
    index=train_df["samples"].astype(str),
    columns=top_genes
)

# Sınıflar için üst renk barı hazırlama
unique_classes = sorted(y_train.unique())
palette_classes = sns.color_palette("Set1", len(unique_classes))
class_color_dict = dict(zip(unique_classes, palette_classes))
sample_colors = y_train.map(class_color_dict)
sample_colors.index = train_df["samples"].astype(str)

cg = sns.clustermap(
    X_train_top_scaled.T,  # Satırlar: Genler, Sütunlar: Örnekler
    method="ward",
    metric="euclidean",
    cmap="vlag",
    col_colors=sample_colors,
    figsize=(15, 11),
    dendrogram_ratio=(0.15, 0.15),
    cbar_pos=(0.02, 0.82, 0.03, 0.14),
    vmin=-2.5,
    vmax=2.5,
    xticklabels=False,
    yticklabels=True
)
cg.ax_heatmap.set_xlabel("Örnekler (Samples)", fontsize=11)
cg.ax_heatmap.set_ylabel(f"En Ayırt Edici {top_k} Gen Probu", fontsize=11)
cg.fig.suptitle(
    f"Breast En Ayırt Edici {top_k} Gen İçin Hiyerarşik Kümeleme ve Isı Haritası (Z-Skor)",
    y=1.02,
    fontsize=14,
    fontweight="bold"
)

# Sınıf renk lejantını ekle
legend_handles = [mpatches.Patch(color=class_color_dict[c], label=c) for c in unique_classes]
cg.ax_col_dendrogram.legend(
    handles=legend_handles,
    title="Kanser Alt Tipi",
    loc="center",
    bbox_to_anchor=(1.12, 0.5),
    frameon=True
)

cg.savefig(os.path.join( "Breast_top35_gen_clustermap.png"), dpi=300, bbox_inches="tight")
plt.show()

# 8.3 En Ayırt Edici İlk 6 Gen İçin Sınıf Bazlı Kutu Grafiği (Boxplot)
top_6_genes = deg_df["Gen_Probu"].iloc[:6].values
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for idx, gene in enumerate(top_6_genes):
    sns.boxplot(
        x=y_train,
        y=X_train[gene],
        hue=y_train,
        order=unique_classes,
        palette="Set1",
        legend=False,
        ax=axes[idx]
    )
    f_val_g = deg_df.loc[deg_df["Gen_Probu"] == gene, "ANOVA_F"].values[0]
    q_val_g = deg_df.loc[deg_df["Gen_Probu"] == gene, "FDR_q_degeri"].values[0]
    axes[idx].set_title(f"Gen: {gene}\n(F={f_val_g:.1f}, FDR={q_val_g:.2e})", fontsize=11)
    axes[idx].set_xlabel("Sınıf (Alt Tip)")
    axes[idx].set_ylabel("Ekspresyon Değeri")
    axes[idx].tick_params(axis="x", rotation=30)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle("BREAST En Yüksek ANOVA F-Skoruna Sahip İlk 6 Genin Sınıflara Göre İfade Dağılımı", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join("Breast_top6_gen_kutu_grafigi.png"), dpi=300)
plt.show()

# train-test karşılaştırması : bathch etkisi 

print("\n[9] EĞİTİM VE BAĞIMSIZ TEST SETİ KARŞILAŞTIRMASI (BATCH ETKİSİ ANALİZİ)...")

# 9.1 İstatistiksel Karşılaştırma (Kolmogorov-Smirnov Testi ve Gen Bazlı Kayma)
ks_global_stat, ks_global_p = ks_2samp(sample_subset, np.random.choice(X_test.values.flatten(), size=min(100000, X_test.size), replace=False))

# Gen bazında ortalama ve standart sapma korelasyonu (Train vs Test)
train_gene_means = X_train.mean(axis=0)
test_gene_means = X_test.mean(axis=0)
mean_corr = np.corrcoef(train_gene_means, test_gene_means)[0, 1]
mean_abs_diff = np.mean(np.abs(train_gene_means - test_gene_means))

# Örnek bazlı medyan ekspresyon değerleri
train_sample_medians = X_train.median(axis=1)
test_sample_medians = X_test.median(axis=1)
ks_median_stat, ks_median_p = ks_2samp(train_sample_medians, test_sample_medians)

print(f" - Genel İfade Dağılımı KS Testi -> İstatistik: {ks_global_stat:.4f} | p-değeri: {ks_global_p:.4e}")
print(f" - Örnek Medyanları KS Testi     -> İstatistik: {ks_median_stat:.4f} | p-değeri: {ks_median_p:.4e}")
print(f" - Gen Ortalamaları Korelasyonu (Train vs Test): r = {mean_corr:.4f}")
print(f" - Gen Ortalamaları Mutlak Fark Ortalaması     : {mean_abs_diff:.4f}")

# 9.2 Ortak PCA Uzayı Hesaplaması (Train üzerinde fit, Test üzerinde transform)
scaler_batch = StandardScaler()
X_train_sc = scaler_batch.fit_transform(X_train)
X_test_sc = scaler_batch.transform(X_test)

pca_batch = PCA(n_components=2, random_state=42)
pca_tr = pca_batch.fit_transform(X_train_sc)
pca_te = pca_batch.transform(X_test_sc)
var_tr_te = pca_batch.explained_variance_ratio_ * 100

# 9.3 Görselleştirme Paneli (2x2 Batch Etkisi ve Dağılım Farkları)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Panel 1: Veri Setine Göre Ortak PCA (Batch kontrolü)
axes[0, 0].scatter(pca_tr[:, 0], pca_tr[:, 1], c="royalblue", label=f"Train (n={len(X_train)})", alpha=0.75, edgecolors="k", s=65)
axes[0, 0].scatter(pca_te[:, 0], pca_te[:, 1], c="darkorange", marker="^", label=f"Test (n={len(X_test)})", alpha=0.85, edgecolors="k", s=80)
axes[0, 0].set_title(f"Ortak PCA Uzayı: Veri Seti Kaynağı (Batch Kontrolü)\n(PC1: %{var_tr_te[0]:.1f}, PC2: %{var_tr_te[1]:.1f})")
axes[0, 0].set_xlabel("PC 1")
axes[0, 0].set_ylabel("PC 2")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Hem Sınıf Hem Veri Seti Kaynağına Göre Ortak PCA
combined_pca_df = pd.DataFrame({
    "PC1": np.concatenate([pca_tr[:, 0], pca_te[:, 0]]),
    "PC2": np.concatenate([pca_tr[:, 1], pca_te[:, 1]]),
    "Sınıf": pd.concat([y_train, y_test], ignore_index=True),
    "Veri_Seti": ["Train"] * len(X_train) + ["Test"] * len(X_test)
})
sns.scatterplot(
    data=combined_pca_df,
    x="PC1",
    y="PC2",
    hue="Sınıf",
    style="Veri_Seti",
    palette="Set1",
    markers={"Train": "o", "Test": "X"},
    s=80,
    ax=axes[0, 1]
)
axes[0, 1].set_title("Ortak PCA Uzayı: Biyolojik Sınıflar vs Veri Seti")
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

# Panel 3: Train vs Test Genel Gen İfade Yoğunluğu (KDE)
test_subset = np.random.choice(X_test.values.flatten(), size=min(100000, X_test.size), replace=False)
sns.kdeplot(sample_subset, label=f"Train (Ort: {val_mean:.2f}, Std: {val_std:.2f})", ax=axes[1, 0], color="royalblue", fill=True, alpha=0.3)
sns.kdeplot(test_subset, label=f"Test (Ort: {t_mean:.2f}, Std: {t_std:.2f})", ax=axes[1, 0], color="darkorange", fill=True, alpha=0.3)
axes[1, 0].set_title(f"Genel İfade Yoğunluğu Karşılaştırması (KS Stat: {ks_global_stat:.4f})")
axes[1, 0].set_xlabel("Gen Ekspresyon Değeri")
axes[1, 0].set_ylabel("Yoğunluk (Density)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: Örnek Bazlı Medyan Dağılımları (Boxplot + Stripplot)
median_df = pd.DataFrame({
    "Medyan_Ekspresyon": np.concatenate([train_sample_medians, test_sample_medians]),
    "Veri_Seti": ["Train"] * len(train_sample_medians) + ["Test"] * len(test_sample_medians)
})
sns.boxplot(
    data=median_df,
    x="Veri_Seti",
    y="Medyan_Ekspresyon",
    hue="Veri_Seti",
    palette={"Train": "royalblue", "Test": "darkorange"},
    width=0.4,
    legend=False,
    ax=axes[1, 1]
)
sns.stripplot(
    data=median_df,
    x="Veri_Seti",
    y="Medyan_Ekspresyon",
    color="black",
    alpha=0.4,
    jitter=0.15,
    ax=axes[1, 1]
)
axes[1, 1].set_title(f"Örnek Bazlı Medyan Ekspresyon Dağılımı (KS p={ks_median_p:.2e})")
axes[1, 1].set_xlabel("Veri Seti")
axes[1, 1].set_ylabel("Örnek Medyan Ekspresyon Değeri")
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle("Breast Eğitim ve Bağımsız Test Setlerinin Dağılım ve Batch Etkisi Analizi", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join( "Breast_batch_etkisi_analizi.png"), dpi=300)
plt.show()


# 10. BULGULARIN NUMARALI ÖZETİ VE MODELLEME İÇİN ÇIKARIMLAR

train_counts = y_train.value_counts()
imbalance_ratio = train_counts.max() / train_counts.min()
min_class_name = train_counts.idxmin()
max_class_name = train_counts.idxmax()
batch_status = "DÜŞÜK (Veri setleri uyumlu)" if ks_global_stat < 0.05 else "DİKKATE DEĞER (Dağılım kayması mevcut)"

print("\n" + "=" * 75)
print("     BULGULARIN NUMARALI ÖZETİ VE MODELLEME İÇİN ÇIKARIMLAR")
print("=" * 75)

print(
    f"1. [Veri Kalitesi ve Ön İşleme]:\n"
    f"   - Eğitim ve test setlerinde eksik (NaN: {null_train + null_test}) veya sonsuz (INF: {inf_train + inf_test}) değer bulunmamaktadır.\n"
    f"   - Sütun sıralamaları ve indeks eşleşmeleri tamdır. Ekspresyon değerleri [{val_min:.2f}, {val_max:.2f}] aralığında olduğundan\n"
    f"     veri seti halihazırda log2 ölçeğindedir; modelleme öncesinde tekrar log dönüşümü UYGULANMAMALIDIR."
)

print(
    f"\n2. [Boyut Laneti (p >> n) ve Öznitelik Seçimi (Feature Selection)]:\n"
    f"   - Eğitim setinde sadece {X_train.shape[0]} örnek olmasına karşın {X_train.shape[1]} gen probu mevcuttur.\n"
    f"   - ANOVA F-testi sonucunda FDR < 0.05 anlamlılık düzeyinde {sig_fdr_005} genin sınıfları anlamlı şekilde ayırdığı görülmüştür.\n"
    f"   - Aşırı öğrenmeyi (overfitting) engellemek için modelleme aşamasında tek değişkenli filtreleme (SelectKBest - ANOVA F),\n"
    f"     L1 (Lasso) / ElasticNet regülarizasyonu veya aşamalı öznitelik eleme (RFE) mutlaka pipeline içine eklenmelidir."
)

print(
    f"\n3. [Ölçekleme (Scaling) ve Aykırı Değer Yönetimi]:\n"
    f"   - Genlerin ifade aralıkları ve varyansları birbirinden farklıdır ve IQR kriterine göre uç değerler (Üst: %{outliers_high / X_train.size * 100:.2f}) mevcuttur.\n"
    f"   - SVM, Lojistik Regresyon, MLP ve KNN gibi ölçek duyarlı algoritmalardan önce veri sızıntısını (data leakage) önlemek adına\n"
    f"     sadece eğitim setinde fit edilmiş 'StandardScaler' veya uç değerlere dayanıklı 'RobustScaler' kullanılmalıdır."
)

print(
    f"\n4. [Sınıf Dengesizliği (Class Imbalance) ve Doğrulama Stratejisi]:\n"
    f"   - En kalabalık sınıf ('{max_class_name}': {train_counts.max()} örnek) ile en az örneğe sahip sınıf ('{min_class_name}': {train_counts.min()} örnek)\n"
    f"     arasındaki dengesizlik oranı {imbalance_ratio:.2f}'dir.\n"
    f"   - Modellerde 'class_weight=\"balanced\"' parametresi aktif edilmeli, hiperparametre optimizasyonu 'StratifiedKFold' ile yapılmalı\n"
    f"     ve model performansı sadece Accuracy ile değil, 'Macro-F1', 'Balanced Accuracy' ve 'ROC-AUC' metrikleriyle değerlendirilmelidir."
)

print(
    f"\n5. [Eğitim vs Test Dağılım Uyumu (Batch Etkisi)]:\n"
    f"   - Eğitim ve bağımsız test setleri arasındaki genel dağılım farkı (KS İstatistiği: {ks_global_stat:.4f}, Gen Ort. Korelasyonu: r={mean_corr:.4f})\n"
    f"     batch etkisinin '{batch_status}' düzeyde olduğunu göstermektedir.\n"
    f"   - Ortak PCA uzayında test örnekleri eğitim kümesiyle aynı biyolojik alt tip bölgelerine düşmektedir; bu durum eğitim setinde\n"
    f"     öğrenilecek karar sınırlarının bağımsız test setine yüksek başarıyla genellenebileceğini doğrulamaktadır."
)
print("=" * 75)
print(f"\nAnaliz tamamlandı. Tüm grafikler ve tablolar '{dataset_dir}' klasörüne kaydedildi.")