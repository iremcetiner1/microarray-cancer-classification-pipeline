import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
# DATASET İCİN REHBERDE İSTERLER 
# 1. = Eksik/INF değerler, negatif ve uç değerler, call (P/A/M) sütunlarının anlamı, etiket–sütun eşleştirmesinin doğrulanması.

dataset_dir = "golub-data-set"
train_df = pd.read_csv(os.path.join(dataset_dir, "data_set_ALL_AML_train.csv"))
actual_df = pd.read_csv(os.path.join(dataset_dir, "actual.csv"))
test_df = pd.read_csv(os.path.join(dataset_dir, "data_set_ALL_AML_independent.csv"))

print("--- 1. HAM VERİ BOYUTLARI ---")
print(f"Train Ham Boyut (Gen x Kolon) : {train_df.shape}")
print(f"Test Ham Boyut (Gen x Kolon)  : {test_df.shape}")
print(f"Actual Etiket Boyutu          : {actual_df.shape}")

# 2. Call (P/A/M) Sütunlarının Genel ve Gen Bazlı Kalite Analizi
train_call_cols = [c for c in train_df.columns if 'call' in c.lower()]
test_call_cols = [c for c in test_df.columns if 'call' in c.lower()]

# String boşluklarını temizleyerek düzleştirme (Örn: 'P ' -> 'P')
train_calls_clean = train_df[train_call_cols].apply(lambda col: col.astype(str).str.strip())
test_calls_clean = test_df[test_call_cols].apply(lambda col: col.astype(str).str.strip())

train_call_dist = pd.Series(train_calls_clean.values.flatten()).value_counts(normalize=True) * 100
test_call_dist = pd.Series(test_calls_clean.values.flatten()).value_counts(normalize=True) * 100

# Gen bazında güvenilirlik: Eğitim setinde hiçbir hastada 'P' (Present) olmayan (tamamen gürültü) gen sayısı
all_absent_genes = (train_calls_clean == 'A').all(axis=1).sum()
never_present_genes = (train_calls_clean != 'P').all(axis=1).sum()

print("\n--- 2. AFFYMETRIX CALL (P/A/M) KALİTE VE GÜRÜLTÜ ANALİZİ ---")
print("Eğitim Call (%) Dağılımı :", train_call_dist.round(2).to_dict())
print("Test Call (%) Dağılımı   :", test_call_dist.round(2).to_dict())
print(f"Tüm Eğitim Hastalarında 'A' (Absent) Olan Gen Sayısı   : {all_absent_genes} / {len(train_df)} (%{100*all_absent_genes/len(train_df):.2f})")
print(f"Hiçbir Eğitim Hastasında 'P' (Present) Olmayan Genler  : {never_present_genes} / {len(train_df)} (%{100*never_present_genes/len(train_df):.2f})")

# 3. Hasta verilerini ayıklama, sayısallaştırma ve N x p (Hasta x Gen) formatına çevirme
def extract_numeric_patient_data(df):
    gene_accessions = df['Gene Accession Number'].astype(str).str.strip().values
    patient_cols = [c for c in df.columns if str(c).strip().isdigit()]
    
    df_numeric = df[patient_cols].apply(pd.to_numeric, errors='coerce')
    df_transposed = df_numeric.T
    
    df_transposed.columns = gene_accessions
    df_transposed.index = df_transposed.index.str.strip().astype(int)
    df_transposed.index.name = 'patient'
    
    return df_transposed.sort_index()

X_train_raw = extract_numeric_patient_data(train_df)
X_test_raw = extract_numeric_patient_data(test_df)

# 4. Etiket-Sütun ve Gen Hizalaması Doğrulaması (Assertions)
actual_df['patient'] = actual_df['patient'].astype(int)
y_train = actual_df.set_index('patient').loc[X_train_raw.index, 'cancer']
y_test = actual_df.set_index('patient').loc[X_test_raw.index, 'cancer']

# Doğrulama Kontrolleri
assert (X_train_raw.index == y_train.index).all(), "Eğitim hasta ID eşleşme hatası!"
assert (X_test_raw.index == y_test.index).all(), "Test hasta ID eşleşme hatası!"
assert (X_train_raw.columns == X_test_raw.columns).all(), "Train ve Test gen sütun sıralaması uyuşmuyor!"
assert set(X_train_raw.index).isdisjoint(set(X_test_raw.index)), "Train ve Test arasında hasta sızıntısı (overlap) var!"

dup_genes = X_train_raw.columns.duplicated().sum()

print("\n--- 3. DÜZENLENMİŞ VERİ MATRİSLERİ, HİZALAMA VE SINIF DAĞILIMI ---")
print(f"Eğitim Seti (Hasta x Gen) : {X_train_raw.shape} | Sınıflar: {y_train.value_counts().to_dict()}")
print(f"Test Seti (Hasta x Gen)   : {X_test_raw.shape} | Sınıflar: {y_test.value_counts().to_dict()}")
print(f"Tekrarlayan (Duplicate) Gen ID Sayısı : {dup_genes}")
print("Doğrulama: Hasta ID, Etiket ve Gen sıralamaları %100 eşleşti.")

# 5. Eksik, INF, Negatif/Sıfır ve IQR Uç Değer (Outlier) Kontrolleri
def print_quality_report(X, dataset_name):
    vals = X.values
    total_entries = vals.size
    nan_count = np.isnan(vals).sum()
    inf_count = np.isinf(vals).sum()
    neg_count = (vals < 0).sum()
    zero_or_neg_count = (vals <= 0).sum() # Log2 dönüşümü icin
    
    # Küresel (Global) IQR Uç Değer Hesabı
    q25, q75 = np.nanpercentile(vals, [25, 75])
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    iqr_outliers = ((vals < lower_bound) | (vals > upper_bound)).sum()
    
    # Gen bazında (Sütun bazlı) IQR Uç Değer Hesabı
    g_q25 = X.quantile(0.25, axis=0)
    g_q75 = X.quantile(0.75, axis=0)
    g_iqr = g_q75 - g_q25
    gene_outliers = ((X < (g_q25 - 1.5 * g_iqr)) | (X > (g_q75 + 1.5 * g_iqr))).sum().sum()
    
    # Sıfır varyanslı (sabit) gen sayısı
    zero_var_genes = (X.var(axis=0) == 0).sum()

    print(f"\n--- {dataset_name} VERİ KALİTESİ VE UÇ DEĞER RAPORU ---")
    print(f"Toplam Hücre Sayısı               : {total_entries:,}")
    print(f"Eksik (NaN) / Sonsuz (INF) Değer  : {nan_count} / {inf_count}")
    print(f"Negatif (< 0) İfade Değeri        : {neg_count:,} (%{100 * neg_count / total_entries:.2f})")
    print(f"Sıfır veya Negatif (<= 0) Değer   : {zero_or_neg_count:,} (%{100 * zero_or_neg_count / total_entries:.2f}) -> Log2 öncesi düzeltilmeli")
    print(f"Min / Medyan / Maks İfade Değeri  : { np.nanmin(vals):.1f} / {np.nanmedian(vals):.1f} / {np.nanmax(vals):.1f}")
    print(f"Genel IQR Sınırları               : [{lower_bound:.1f}, {upper_bound:.1f}]")
    print(f"Genel IQR Uç Değer Sayısı         : {iqr_outliers:,} (%{100 * iqr_outliers / total_entries:.2f})")
    print(f"Gen Bazlı IQR Uç Değer Sayısı     : {gene_outliers:,} (%{100 * gene_outliers / total_entries:.2f})")
    print(f"Sıfır Varyanslı (Sabit) Gen Sayısı: {zero_var_genes}")

print_quality_report(X_train_raw, "EĞİTİM SETİ (TRAIN)")
print_quality_report(X_test_raw, "BAĞIMSIZ TEST SETİ (TEST)")

#2.Betimsel İstatistikler ve Sınıf Dağılımı + log dönüsümü öncesi ve sonrası
# 1. CLASS DAĞILIMI (Sayı, Yüzde ve Dengesizlik Oranı)
# 1. SINIF DAĞILIMI (Sayı, Yüzde ve Dengesizlik Oranı)
train_counts = y_train.value_counts()
test_counts = y_test.value_counts()
train_pct = y_train.value_counts(normalize=True) * 100
test_pct = y_test.value_counts(normalize=True) * 100

class_dist = pd.DataFrame({
    'Eğitim Sayı (n=38)': train_counts,
    'Eğitim (%)': train_pct.round(2),
    'Test Sayı (n=34)': test_counts,
    'Test (%)': test_pct.round(2)
})
class_dist.loc['Toplam'] = [train_counts.sum(), 100.0, test_counts.sum(), 100.0]

print("\n--- 1. SINIF DAĞILIMI VE DENGESİZLİK ORANI ---")
print(class_dist)
print(f"Eğitim ALL/AML Dengesizlik Oranı : {train_counts['ALL'] / train_counts['AML']:.2f}:1")
print(f"Test ALL/AML Dengesizlik Oranı   : {test_counts['ALL'] / test_counts['AML']:.2f}:1")

# 2. KIRPMA (CLIPPING - LOG2 ÖNCESİ) VE LOG2 DÖNÜŞÜMÜ (Ayrı ayrı kaydediyoruz)
X_train_clipped = X_train_raw.clip(lower=100, upper=16000)
X_test_clipped = X_test_raw.clip(lower=100, upper=16000)

X_train_log2 = np.log2(X_train_clipped)
X_test_log2 = np.log2(X_test_clipped)

# 3. BETİMSEL İSTATİSTİKLER TABLOSU (Ham vs Kırpılmış vs Log2)
def get_descriptive_summary(df, label):
    vals = df.values.ravel()
    q25, q50, q75 = np.percentile(vals, [25, 50, 75])
    return pd.Series({
        'Minimum': np.min(vals),
        'Q1 (%25)': q25,
        'Medyan (Q2)': q50,
        'Ortalama (Mean)': np.mean(vals),
        'Q3 (%75)': q75,
        'Maksimum': np.max(vals),
        'Standart Sapma': np.std(vals),
        'IQR (Q3 - Q1)': q75 - q25,
        'Genel Çarpıklık (Skewness)': stats.skew(vals),
        'Gen Bazlı Ort. Skewness': df.skew(axis=0).mean(),
        'Genel Basıklık (Kurtosis)': stats.kurtosis(vals)
    }, name=label)

desc_table = pd.concat([
    get_descriptive_summary(X_train_raw, '1. Train - Ham (Raw)'),
    get_descriptive_summary(X_train_clipped, '2. Train - Kırpılmış (Log2 Öncesi)'),
    get_descriptive_summary(X_train_log2, '3. Train - Log2 Sonrası'),
    get_descriptive_summary(X_test_raw, '4. Test - Ham (Raw)'),
    get_descriptive_summary(X_test_log2, '5. Test - Log2 Sonrası')
], axis=1).round(2)

print("\n--- 2. BETİMSEL İSTATİSTİKLER (HAM vs KIRPILMIŞ vs LOG2) ---")
print(desc_table)

# 4. GÖRSELLEŞTİRME (2x2 Panel, 300 DPI):
# Panel 1: Sınıf Dağılımı | Panel 2: Ham Veri | Panel 3: Log2 Öncesi (Kırpılmış) | Panel 4: Log2 Sonrası
fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

# Grafik 1 (Sol Üst): Sınıf Dağılımı
class_dist.drop('Toplam')[['Eğitim Sayı (n=38)', 'Test Sayı (n=34)']].T.plot(
    kind='bar', stacked=False, ax=axes[0, 0], color=['#2b5c8f', '#d95f02'], rot=0
)
axes[0, 0].set_title("1. Eğitim ve Test Seti Sınıf Dağılımı (ALL vs AML)", fontweight='bold')
axes[0, 0].set_ylabel("Hasta Sayısı")
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)

# Grafik 2 (Sağ Üst): Ham Veri Dağılımı (Negatif ve Aşırı Uç Değerli)
raw_vals = X_train_raw.values.ravel()
sns.histplot(raw_vals, bins=100, ax=axes[0, 1], color='crimson', stat='density')
axes[0, 1].axvline(np.mean(raw_vals), color='black', linestyle='--', label=f"Ortalama: {np.mean(raw_vals):.0f}")
axes[0, 1].axvline(np.median(raw_vals), color='gold', linestyle='-', label=f"Medyan: {np.median(raw_vals):.0f}")
axes[0, 1].set_title("2. Ham Gen İfade Dağılımı (Negatif ve Uç Değerler Dahil)", fontweight='bold')
axes[0, 1].set_xlabel("Ham Floresan Yoğunluğu")
axes[0, 1].legend()

# Grafik 3 (Sol Alt): Kırpılmış [100-16.000] ama Log2 ÖNCESİ Dağılım
clipped_vals = X_train_clipped.values.ravel()
sns.histplot(clipped_vals, bins=80, ax=axes[1, 0], color='darkorange', stat='density')
axes[1, 0].axvline(np.mean(clipped_vals), color='black', linestyle='--', label=f"Ortalama: {np.mean(clipped_vals):.0f}")
axes[1, 0].axvline(np.median(clipped_vals), color='blue', linestyle='-', label=f"Medyan: {np.median(clipped_vals):.0f}")
axes[1, 0].set_title("3. Log2 Öncesi: Kırpılmış [100 - 16.000] Doğrusal Dağılım", fontweight='bold')
axes[1, 0].set_xlabel("Kırpılmış Floresan Yoğunluğu (Doğrusal Ölçek)")
axes[1, 0].legend()

# Grafik 4 (Sağ Alt): Log2 Dönüşümü SONRASI Dağılım
log_vals = X_train_log2.values.ravel()
sns.histplot(log_vals, bins=60, ax=axes[1, 1], color='teal', stat='density')
axes[1, 1].axvline(np.mean(log_vals), color='black', linestyle='--', label=f"Ortalama: {np.mean(log_vals):.2f}")
axes[1, 1].axvline(np.median(log_vals), color='gold', linestyle='-', label=f"Medyan: {np.median(log_vals):.2f}")
axes[1, 1].set_title("4. Log2 Sonrası: Kırpma [100 - 16.000] + Log2 Dağılımı", fontweight='bold')
axes[1, 1].set_xlabel("Log2(İfade Değeri)")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig("golub_on_isleme_ve_log2_karsilastirma_300dpi.png", dpi=300, bbox_inches='tight')
plt.show()

#3.Boyut indirgeme ile görselleştirme: PCA (açıklanan varyans dahil), t-SNE veya UMAP. 
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import os
import warnings

# TensorFlow oneDNN ve Protobuf uyarılarını tamamen kapatma
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# UMAP'i TensorFlow tetiklemeden doğrudan alt modülden çekme
try:
    from umap.umap_ import UMAP
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False
    print("Bilgi: UMAP kullanılamıyor, 4. panelde PC1 Gen Ağırlıkları (Loadings) çizdirilecek.")

# 1. Z-Skor Normalizasyonu (SADECE eğitim setinde fit edilir!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_log2)
X_test_scaled = scaler.transform(X_test_log2)

# 2. PCA Uygulaması (Eğitimde fit, Testte transform -> %100 Veri Sızıntısız)
pca_full = PCA(random_state=42)
X_train_pca = pca_full.fit_transform(X_train_scaled)
X_test_pca = pca_full.transform(X_test_scaled)

exp_var = pca_full.explained_variance_ratio_ * 100
cum_var = np.cumsum(exp_var)

n_80 = np.argmax(cum_var >= 80) + 1
n_90 = np.argmax(cum_var >= 90) + 1

print("\n--- 1. PCA AÇIKLANAN VARYANS ÖZETİ ---")
print(f"PC1 Açıklanan Varyans : %{exp_var[0]:.2f}")
print(f"PC2 Açıklanan Varyans : %{exp_var[1]:.2f}")
print(f"PC3 Açıklanan Varyans : %{exp_var[2]:.2f}")
print(f"İlk 2 Bileşen (2D)    : %{cum_var[1]:.2f}")
print(f"İlk 10 Bileşen Toplamı: %{cum_var[9]:.2f}")
print(f"Varyansın %80'i için gereken bileşen sayısı: {n_80}")
print(f"Varyansın %90'ı için gereken bileşen sayısı: {n_90}")

# PC1 ve PC2'ye En Çok Katkı Veren İlk 5 Gen (PCA Loadings)
loadings = pd.DataFrame(
    pca_full.components_[:2].T,
    index=X_train_log2.columns,
    columns=['PC1_Loading', 'PC2_Loading']
)
top_pc1_genes = loadings['PC1_Loading'].abs().nlargest(5).index.tolist()
top_pc2_genes = loadings['PC2_Loading'].abs().nlargest(5).index.tolist()
print(f"PC1'i En Çok Etkileyen 5 Gen (Accession): {top_pc1_genes}")
print(f"PC2'yi En Çok Etkileyen 5 Gen (Accession): {top_pc2_genes}")

# 3. t-SNE Uygulaması (%100 İzole: Train ve Test ayrı ayrı fit edilir)
tsne_train = TSNE(n_components=2, perplexity=12, random_state=42, init='pca', learning_rate='auto')
X_train_tsne = tsne_train.fit_transform(X_train_pca[:, :25])

tsne_test = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', learning_rate='auto')
X_test_tsne = tsne_test.fit_transform(X_test_pca[:, :25])

# 4. UMAP Uygulaması (Eğitimde fit, Testte transform -> %100 Veri Sızıntısız!)
if HAS_UMAP:
    umap_reducer = UMAP(n_neighbors=12, min_dist=0.1, n_components=2, random_state=42)
    X_train_umap = umap_reducer.fit_transform(X_train_scaled)
    X_test_umap = umap_reducer.transform(X_test_scaled)

# 5. KÜME AYRIM KALİTESİ (Train ve Test için Ayrı Ayrı Silhouette Skorları)
y_tr_bin = (y_train == 'AML').astype(int)
y_te_bin = (y_test == 'AML').astype(int)

sil_pca_tr = silhouette_score(X_train_pca[:, :2], y_tr_bin)
sil_pca_te = silhouette_score(X_test_pca[:, :2], y_te_bin)
sil_tsne_tr = silhouette_score(X_train_tsne, y_tr_bin)
sil_tsne_te = silhouette_score(X_test_tsne, y_te_bin)

print("\n--- 2. 2D UZAYDA SINIF AYRIM KALİTESİ (SILHOUETTE SKORU - SIZINTISIZ) ---")
print(f"2D PCA Silhouette   -> Train: {sil_pca_tr:.4f} | Test: {sil_pca_te:.4f}")
print(f"2D t-SNE Silhouette -> Train: {sil_tsne_tr:.4f} | Test: {sil_tsne_te:.4f}")
if HAS_UMAP:
    sil_umap_tr = silhouette_score(X_train_umap, y_tr_bin)
    sil_umap_te = silhouette_score(X_test_umap, y_te_bin)
    print(f"2D UMAP Silhouette  -> Train: {sil_umap_tr:.4f} | Test: {sil_umap_te:.4f}")

# 6. GÖRSELLEŞTİRME 
fig, axes = plt.subplots(2, 2, figsize=(12,10 ), dpi=300)
palette = {'ALL': '#1f77b4', 'AML': '#d62728'}

# Panel 1 (Sol Üst): Scree Plot
n_comp = len(exp_var)
axes[0, 0].bar(range(1, n_comp + 1), exp_var, alpha=0.6, color='steelblue', label='Bireysel Varyans (%)')
axes[0, 0].plot(range(1, n_comp + 1), cum_var, color='crimson', marker='o', markersize=4, label='Kümülatif Varyans (%)')
axes[0, 0].axhline(80, color='darkgreen', linestyle='--', linewidth=1.2, label=f'%80 Eşiği ({n_80} PC)')
axes[0, 0].axhline(90, color='darkorange', linestyle=':', linewidth=1.2, label=f'%90 Eşiği ({n_90} PC)')
axes[0, 0].set_title("1. PCA Açıklanan Varyans Oranı\n(Scree Plot)", fontsize=12, fontweight='bold', pad=10)
axes[0, 0].set_xlabel("Temel Bileşen Sayısı (PC)", fontsize=10)
axes[0, 0].set_ylabel("Açıklanan Varyans (%)", fontsize=10)
axes[0, 0].legend(loc='center right', fontsize=9)
axes[0, 0].grid(alpha=0.3)

# Yardımcı çizim fonksiyonu (Train: Daire, Test: Üçgen)
def plot_2d_embedding(ax, X_tr, X_te, title, xlabel, ylabel):
    for label in ['ALL', 'AML']:
        idx_tr = (y_train == label).values
        ax.scatter(X_tr[idx_tr, 0], X_tr[idx_tr, 1], c=palette[label],
                   marker='o', label=f"{label} (Train)", s=70, alpha=0.85, edgecolors='k')
        idx_te = (y_test == label).values
        ax.scatter(X_te[idx_te, 0], X_te[idx_te, 1], c=palette[label],
                   marker='^', label=f"{label} (Test)", s=75, alpha=0.85, edgecolors='k')
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

# Panel 2 (Sağ Üst): 2D PCA Projeksiyonu
plot_2d_embedding(
    axes[0, 1], X_train_pca[:, :2], X_test_pca[:, :2],
    title=f"2. 2D PCA Projeksiyonu (Toplam Varyans: %{cum_var[1]:.1f})\n[Silhouette -> Train: {sil_pca_tr:.2f} | Test: {sil_pca_te:.2f}]",
    xlabel=f"PC1 (%{exp_var[0]:.1f})", ylabel=f"PC2 (%{exp_var[1]:.1f})"
)

# Panel 3 (Sol Alt): 2D İzole t-SNE Projeksiyonu
plot_2d_embedding(
    axes[1, 0], X_train_tsne, X_test_tsne,
    title=f"3. 2D İzole t-SNE Projeksiyonu\n[Silhouette -> Train: {sil_tsne_tr:.2f} | Test: {sil_tsne_te:.2f}]",
    xlabel="t-SNE Boyut 1", ylabel="t-SNE Boyut 2"
)

# Panel 4 (Sağ Alt): UMAP Projeksiyonu veya PC1 Gen Ağırlıkları
if HAS_UMAP:
    plot_2d_embedding(
        axes[1, 1], X_train_umap, X_test_umap,
        title=f"4. 2D UMAP Projeksiyonu (Sızıntısız Train -> Test)\n[Silhouette -> Train: {sil_umap_tr:.2f} | Test: {sil_umap_te:.2f}]",
        xlabel="UMAP Boyut 1", ylabel="UMAP Boyut 2"
    )
else:
    top10_pc1 = loadings['PC1_Loading'].abs().nlargest(10).sort_values()
    top10_pc1.plot(kind='barh', color='teal', ax=axes[1, 1], edgecolor='k')
    axes[1, 1].set_title("4. PC1 Eksenini Oluşturan En Önemli 10 Gen\n(|PCA Loadings|)", fontsize=12, fontweight='bold', pad=10)
    axes[1, 1].set_xlabel("Mutlak Ağırlık (|PC1 Loading|)", fontsize=10)
    axes[1, 1].set_ylabel("Gene Accession Number", fontsize=10)
    axes[1, 1].grid(axis='x', alpha=0.3)

# Alt grafikler arasındaki yatay (wspace) ve dikey (hspace) boşlukları açıkça sabitleme
plt.tight_layout(pad=2.5)
plt.subplots_adjust(hspace=0.32, wspace=0.25)

plt.savefig("golub_pca_tsne_umap_300dpi.png", dpi=300, bbox_inches='tight')
plt.show()
#4 Ayırt edici genler: sınıflar arası testler, en iyi 20–50 gen için ısı haritası ve hiyerarşik kümeleme gibi veri analitiği plot örnekleri.
# 1. SADECE EĞİTİM SETİ ÜZERİNDEN ÖZELLİK SEÇİMİ (Data Leakage Önleme)
train_all_mask = (y_train == 'ALL').values
train_aml_mask = (y_train == 'AML').values

# Signal-to-Noise Ratio (SNR) veya Welch's t-test ile genlerin ayırt edicilik skoru
mean_all = X_train_log2.loc[train_all_mask].mean(axis=0)
mean_aml = X_train_log2.loc[train_aml_mask].mean(axis=0)
std_all = X_train_log2.loc[train_all_mask].std(axis=0)
std_aml = X_train_log2.loc[train_aml_mask].std(axis=0)

# SNR Formülü: (Ort_ALL - Ort_AML) / (Std_ALL + Std_AML)
snr_scores = (mean_all - mean_aml) / (std_all + std_aml + 1e-8)

# En yüksek mutlak SNR skoruna sahip ilk 40 geni seçiyoruz
top_40_genes = snr_scores.abs().nlargest(40).index

print(f"\n--- 1. AYIRT EDİCİ GEN SEÇİMİ ---")
print(f"Seçilen En İyi 40 Genin Accession Numaraları (İlk 40): {top_40_genes[:40].tolist()}")

# Seçilen 40 geni eğitim ve test matrislerinden filtreleme
X_train_top40 = X_train_log2[top_40_genes]
X_test_top40 = X_test_log2[top_40_genes]

# 2. ISI HARİTASI İÇİN ÖLÇEKLENDİRME (Z-Score: Her genin ortalaması=0, std=1)
# KRİTİK: Scaler SADECE eğitim setinde fit edilir!
heatmap_scaler = StandardScaler()
X_train_top40_scaled_vals = heatmap_scaler.fit_transform(X_train_top40)

X_train_top40_scaled = pd.DataFrame(
    X_train_top40_scaled_vals,
    index=X_train_top40.index,
    columns=X_train_top40.columns
)

# 3. HASTA ETİKETLERİ VE RENK PALETİ
# ALL hastaları Mavi, AML hastaları Kırmızı renk ile etiketlenir
row_colors = y_train.map({'ALL': '#1f77b4', 'AML': '#d62728'})

# 4. HİYERARŞİK KÜMELEME VE ISI HARİTASI (Clustermap)
# Ward Metodu ve Euclidean Uzaklığı kullanılarak hiyerarşik ağaç (dendrogram) oluşturulur
g = sns.clustermap(
    X_train_top40_scaled,
    method='ward',
    metric='euclidean',
    row_colors=row_colors,
    cmap='vlag',              # Mavi (Düşük ifade) - Beyaz - Kırmızı (Yüksek ifade)
    figsize=(14, 12),
    dendrogram_ratio=(0.15, 0.15),
    cbar_kws={'label': 'Z-Score (Log2 Expression)'},
    linewidths=0.2,
    col_cluster=True          # Genlerin kendi arasındaki hiyerarşik ilişkisini de kümele
)

# Grafik Başlığı ve Düzenleme (300 DPI)
g.fig.suptitle(
    "Eğitim Seti: En İyi 40 Ayırt Edici Gen - Hiyerarşik Kümeleme ve Isı Haritası\n"
    "(Mavi: ALL Sınıfı | Kırmızı: AML Sınıfı)",
    fontsize=14, fontweight='bold', y=1.03
)

# Lejant (Legend) Ekleme
import matplotlib.patches as mpatches
all_patch = mpatches.Patch(color='#1f77b4', label='ALL (Akut Lenfoblastik)')
aml_patch = mpatches.Patch(color='#d62728', label='AML (Akut Miyeloid)')
g.ax_heatmap.legend(handles=[all_patch, aml_patch], title="Gerçek Sınıflar", loc="upper right", bbox_to_anchor=(1.30, 1.25))

plt.savefig("golub_top40_genes_clustermap_300dpi.png", dpi=300, bbox_inches='tight')
plt.show()

#5 Eğitim ve bağımsız test setlerinin karşılaştırılması (batch etkisi) ve iki veri seti arasındaki dağılım farkları. 
from sklearn.decomposition import PCA

# 1. ÖZELLİK MATRİSLERİNİN ORTAK GENLER ÜZERİNDEN HİZALANMASI
# Train ve Test setlerinin aynı gen sütunlarına sahip olduğundan emin oluyoruz
common_genes = X_train_log2.columns.intersection(X_test_log2.columns)
X_tr_aligned = X_train_log2[common_genes]
X_te_aligned = X_test_log2[common_genes]

print(f"--- 1. VERİ SETİ BOYUTLARI ---")
print(f"Eğitim Seti Hizalanmış Boyut : {X_tr_aligned.shape}")
print(f"Bağımsız Test Seti Hizalanmış: {X_te_aligned.shape}")

# 2. İSTATİSTİKSEL DAĞILIM VE BATCH ETKİSİ TESTLERİ (Kolmogorov-Smirnov Testi)
# Her bir gen için Train ve Test dağılımlarının aynı popülasyondan gelip gelmediğini test ediyoruz
ks_p_values = []
for gene in common_genes:
    stat, p_val = stats.ks_2samp(X_tr_aligned[gene], X_te_aligned[gene])
    ks_p_values.append(p_val)

ks_p_values = np.array(ks_p_values)
# Bonferroni düzeltmesi veya FDR ile anlamlı kayma gösteren gen oranı
significantly_shifted_genes = (ks_p_values < 0.05).sum()

print(f"\n--- 2. DAĞILIM KAYMASI (KOLMOGOROV-SMIRNOV) ANALİZİ ---")
print(f"Train ve Test Arasında Dağılımı Anlamlı Düzeyde (p < 0.05) Farklı Olan Gen Sayısı: {significantly_shifted_genes} / {len(common_genes)} (%{100 * significantly_shifted_genes/len(common_genes):.2f})")

# 3. ORTAK PCA PROJEKSİYONU (Batch Etkisi Görsel Kontrolü için)
# KRİTİK: Scaler SADECE eğitim setinde fit edilir, test seti bu scaler ile dönüştürülür!
scaler_batch = StandardScaler()
X_tr_scaled = scaler_batch.fit_transform(X_tr_aligned)
X_te_scaled = scaler_batch.transform(X_te_aligned)

# PCA SADECE eğitim setinde fit edilir
pca_batch = PCA(n_components=2, random_state=42)
X_tr_pca_proj = pca_batch.fit_transform(X_tr_scaled)
X_te_pca_proj = pca_batch.transform(X_te_scaled)

# 4. GÖRSELLEŞTİRME (2x2 Panel - 300 DPI)
# Panel 1: Gen Ekspresyon Ortalama Değerlerinin Karşılaştırması (Train vs Test)
# Panel 2: Ortak PCA Uzayı (Train vs Test Batch Etkisi Göstergesi)
# Panel 3: Kolmogorov-Smirnov p-değerleri Dağılım Histogramı
# Panel 4: Hasta Bazlı Ortalama Ekspresyon Dağılımı (Boxplot / Violinplot)

fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

# Grafik 1: Ortalama Gen Ekspresyonu Karşılaştırması
mean_tr_genes = X_tr_aligned.mean(axis=0)
mean_te_genes = X_te_aligned.mean(axis=0)
axes[0, 0].scatter(mean_tr_genes, mean_te_genes, alpha=0.5, color='purple', edgecolors='k', s=30)
axes[0, 0].plot([mean_tr_genes.min(), mean_tr_genes.max()], [mean_tr_genes.min(), mean_tr_genes.max()], 'r--', label='İdeal Uyum (y=x)')
axes[0, 0].set_title("1. Gen Bazında Ortalama İfade Karşılaştırması\n(Train vs Test)", fontsize=12, fontweight='bold', pad=10)
axes[0, 0].set_xlabel("Eğitim Seti Ortalama Log2 İfadesi", fontsize=10)
axes[0, 0].set_ylabel("Test Seti Ortalama Log2 İfadesi", fontsize=10)
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Grafik 2: Ortak PCA Uzayı (Train ve Test Örneklerinin Dağılımı)
axes[0, 1].scatter(X_tr_pca_proj[:, 0], X_tr_pca_proj[:, 1], color='#1f77b4', marker='o', s=70, label='Eğitim Seti (Train)', alpha=0.85, edgecolors='k')
axes[0, 1].scatter(X_te_pca_proj[:, 0], X_te_pca_proj[:, 1], color='#ff7f0e', marker='^', s=75, label='Bağımsız Test Seti (Test)', alpha=0.85, edgecolors='k')
axes[0, 1].set_title("2. Ortak PCA Uzayında Batch Etkisi Kontrolü\n(Eğitim Eksenlerine Göre Test İzdüşümü)", fontsize=12, fontweight='bold', pad=10)
axes[0, 1].set_xlabel(f"Temel Bileşen 1 (PC1 - %{pca_batch.explained_variance_ratio_[0]*100:.1f})", fontsize=10)
axes[0, 1].set_ylabel(f"Temel Bileşen 2 (PC2 - %{pca_batch.explained_variance_ratio_[1]*100:.1f})", fontsize=10)
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Grafik 3: Kolmogorov-Smirnov p-değerleri Dağılımı
sns.histplot(ks_p_values, bins=50, ax=axes[1, 0], color='teal', kde=True)
axes[1, 0].axvline(0.05, color='crimson', linestyle='--', linewidth=1.5, label='Anlamlılık Eşiği (p = 0.05)')
axes[1, 0].set_title("3. Gen Dağılım Farklılıkları (KS Test p-değerleri)\n(0.05 altı dağılımı kaymış genlerdir)", fontsize=12, fontweight='bold', pad=10)
axes[1, 0].set_xlabel("Kolmogorov-Smirnov p-değeri", fontsize=10)
axes[1, 0].set_ylabel("Gen Sayısı", fontsize=10)
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Grafik 4: Hasta Bazlı Global İfade Dağılımı (Boxplot / Violin)
sample_summary_df = pd.DataFrame({
    'Ortalama_Ifade': pd.concat([X_tr_aligned.mean(axis=1), X_te_aligned.mean(axis=1)]),
    'Set': ['Train (n=38)'] * len(X_tr_aligned) + ['Test (n=34)'] * len(X_te_aligned)
})
sns.boxplot(x='Set', y='Ortalama_Ifade', data=sample_summary_df, ax=axes[1, 1], palette=['#aec7e8', '#ffbb78'])
sns.stripplot(x='Set', y='Ortalama_Ifade', data=sample_summary_df, ax=axes[1, 1], color='black', alpha=0.6, jitter=0.2, size=6)
axes[1, 1].set_title("4. Hasta Bazlı Global İfade Seviyeleri Dağılımı\n(Train vs Test Medyan ve Çeyrekler)", fontsize=12, fontweight='bold', pad=10)
axes[1, 1].set_xlabel("Veri Seti Bölümü", fontsize=10)
axes[1, 1].set_ylabel("Hasta Ortalama Log2 İfadesi", fontsize=10)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout(pad=2.5)
plt.subplots_adjust(hspace=0.35, wspace=0.25)
plt.savefig("golub_train_test_batch_effect_analysis_300dpi.png", dpi=300, bbox_inches='tight')
plt.show()