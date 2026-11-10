# 🧬 Microarray Cancer Classification Pipeline (ALL vs. AML)

End-to-end exploratory data analysis (EDA), statistical bioinformatics, and leakage-free machine learning pipeline for high-dimensional microarray gene expression subtyping (**Acute Lymphoblastic Leukemia [ALL]** vs. **Acute Myeloid Leukemia [AML]**) based on the landmark **Golub et al. (1999)** dataset.

---

## 📌 Proje Özeti ve Problem Tanımı
Mikroarray gen ifade verileri, örneklem sayısının (n) öznitelik/gen sayısından (p) çok daha küçük olduğu (p>> n, *"Curse of Dimensionality"*) yüksek boyutlu biyomedikal verilerdir. 
Bu projede:
* **7.129 gen (öznitelik)** ve **72 hasta örneği** (38 Eğitim, 34 Bağımsız Test) incelenmiştir.
* Ham floresan yoğunluklarındaki arka plan gürültüsü (negatif değerler), çarpık dağılımlar ve **Batch Etkisi (Eğitim vs. Test dağılım farkları)** analiz edilmiştir.
---

## 📂 Klasör Yapısı
# 🧬 Microarray Cancer Classification Pipeline

End-to-end exploratory data analysis (EDA), statistical bioinformatics, and leakage-free machine learning pipeline for high-dimensional microarray gene expression subtyping across multiple landmark cancer datasets:
- **Leukemia (ALL vs. AML)** — *Golub et al. (1999)*
- **Breast Cancer (Subtyping / GSE45827)**
- **Colon Cancer**

---

## 📌 Proje Özeti ve Problem Tanımı
Mikroarray gen ifade verileri, örneklem sayısının ($n$) öznitelik/gen sayısından ($p$) çok daha küçük olduğu ($p \gg n$, *"Curse of Dimensionality"*) yüksek boyutlu biyomedikal verilerdir. Bu depoda farklı kanser türleri üzerinde:
- Ham floresan yoğunluklarındaki arka plan gürültüsü, çarpık dağılımlar ve **Batch Etkisi (Eğitim vs. Test dağılım farkları)** analiz edilmiştir.
- Boyut indirgeme (**PCA, t-SNE, UMAP**), istatistiksel diferansiyel gen ekspresyonu (**Welch t-testi, ANOVA, Benjamini-Hochberg FDR**) ve makine öğrenmesi modelleme adımları uygulanmıştır.
- Analiz sonuçları yüksek çözünürlüklü (300 DPI) grafikler ve kümeleme ısı haritaları (**Clustermap**) ile görselleştirilmiştir.

---

## 📂 Klasör Yapısı
```text
microarray-cancer-classification-pipeline/
│
├── breast-data-set/
│   ├── Breast_en_iyi_50_gen_ANOVA.csv         # ANOVA ile belirlenen en ayırt edici 50 gen
│   ├── Breast_GSE45827.csv                    # Birleştirilmiş GSE45827 veri seti
│   ├── train_GSE45827.csv                     # Meme kanseri eğitim seti
│   └── test_GSE45827.csv                      # Meme kanseri test seti
│
├── golub-data-set/
│   ├── data_set_ALL_AML_train.csv             # Golub eğitim seti (38 hasta x 7129 gen)
│   ├── data_set_ALL_AML_independent.csv       # Golub bağımsız test seti (34 hasta x 7129 gen)
│   └── actual.csv                             # Golub hasta ID ve kanser alt tipi etiketleri (ALL / AML)
│
├── .gitignore
├── about_GolubDataSet.py                      # Golub veri seti yapısal özet ve keşif betiği
├── Golub_EDA_DataAnalysis.py                  # Golub (Lösemi) uçtan uca EDA ve modelleme betiği
├── Breast_EDA_DataAnalysis.py                 # Meme Kanseri (GSE45827) EDA ve istatistik analiz betiği
├── Breast_test.py                             # Meme Kanseri modelleme/test değerlendirme betiği
├── Colon_EDA_DataAnalysis.py                  # Kolon Kanseri EDA ve diferansiyel analiz betiği
│
├── Breast_batch_etkisi_analizi.png            # Meme kanseri batch etkisi dağılım grafikleri
├── Breast_boyut_indirgeme_analizi.png         # Meme kanseri PCA / t-SNE / UMAP projeksiyonları
├── Breast_log_donusumu_karsilastirma.png      # Meme kanseri log dönüşüm öncesi/sonrası dağılımı
├── Breast_sinif_dagilimi_train_vs_test.png    # Eğitim ve test setleri sınıf dağılımları
├── Breast_top6_gen_kutu_grafigi.png           # En belirteç 6 genin alt tiplere göre kutu grafikleri
├── Breast_top35_gen_clustermap.png            # En ayırt edici 35 genin hiyerarşik kümeleme ısı haritası
│
├── golub_on_isleme_ve_log2_karsilastirma_300dpi.png # Golub kırpma ve log dönüşümü karşılaştırması
├── golub_pca_tsne_umap_300dpi.png                  # Golub boyut indirgeme (PCA, t-SNE, UMAP)
├── golub_top40_genes_cluster_map_300dpi.png        # Golub en ayırt edici 40 gen clustermap
├── golub_train_test_batch_effect_analysis_300dpi.png # Golub KS-testi ve batch etkisi analizi
│
└── README.md                                  # Proje dokümantasyonu ve çalıştırma rehberi
```

---

## 🔬 Metodoloji ve Analitik Adımlar (Pipeline Architecture)

### 1. Veri Kalitesi ve Ön İşleme (Data Quality & Preprocessing)
* **Matris Standardizasyonu:** Ham veri setindeki Affymetrix `call` (`P`: Present, `A`: Absent, `M`: Marginal) bayrakları ayrıştırılarak veri n\p (Hasta\Gen) formatına dönüştürülmüştür.
* **Etiket Doğrulaması:** `actual.csv` dosyasındaki hasta ID'leri ile gen ifade matrislerinin indeks eşleşmesi `assert` kontrolleri ile güvence altına alınmıştır.
* **Gürültü Kırpma ve Dönüşüm (Clipping & Log Transformation):** Arka plan ışıma gürültüsü kaynaklı negatif ve aşırı uç değerler Golub protokolüne uygun olarak **[20, 16.000]** aralığına kırpılmış ve log_{10} dönüşümü ile simetrikleştirilmiştir.

### 2. Keşifsel Veri Analizi (EDA) ve Biyoistatistik
* **Batch Etkisi Analizi:** Eğitim (38) ve bağımsız test (34) setleri arasındaki dağılım farkları Kernel Density Estimation (KDE), hasta bazlı kutu grafikleri (Boxplot) ve **Kolmogorov-Smirnov (KS) testi** ile incelenmiştir.
* **Boyut İndirgeme:** Yüksek boyutlu gen uzayı **PCA** (Açıklanan varyans / Scree plot dahil), **t-SNE** ve **UMAP** ile 2 boyuta indirgenerek sınıfların ayrışabilirliği görselleştirilmiştir.
* **Ayırt Edici Gen Analizi (Differential Expression):** Yalnızca eğitim seti üzerinde **Welch t-testi** ve **Benjamini-Hochberg FDR** çoklu test düzeltmesi uygulanmış; en ayırt edici 30 gen **Hiyerarşik Kümelemeli Isı Haritası (Clustermap)** ile görselleştirilmiştir.


---

## 🚀 Kodları Çalıştırma Yol Haritası (Execution Roadmap)

Projeyi kendi bilgisayarınızda sıfırdan çalıştırmak için aşağıdaki adımları sırasıyla uygulayın:

### Adım 1: Depoyu Klonlayın (Clone Repository)
Terminali (veya Git Bash'i) açın ve projeyi bilgisayarınıza indirin:
```bash
git clone [https://github.com/](https://github.com/)<iremcetiner1">/microarray-cancer-classification-pipeline.git
cd microarray-cancer-classification-pipeline
```

### Adım 2: Sanal Ortam Oluşturun (Önerilen)
Paket çakışmalarını önlemek için bir Python sanal ortamı (virtual environment) kurun ve aktif edin:
```bash
# Windows (CMD / PowerShell)
python -m venv venv
venv\Scripts\activate

# Windows (Git Bash)
python -m venv venv
source venv/Scripts/activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Adım 3: Gerekli Kütüphaneleri Yükleyin (Install Dependencies)
Analiz, istatistiksel testler ve makine öğrenmesi pipeline'ı için gerekli paketleri yükleyin:
```bash
pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn imbalanced-learn umap-learn
```

### Adım 4: Veri Seti Klasörünü Doğrulayın
Ana dizinde `golub-data-set` klasörünün bulunduğundan ve içinde şu 3 dosyanın tam adıyla yer aldığından emin olun:
* `golub-data-set/data_set_ALL_AML_train.csv`
* `golub-data-set/data_set_ALL_AML_independent.csv`
* `golub-data-set/actual.csv`

### Adım 5: Analiz ve Modelleme Kodunu Çalıştırın
Ana Python betiğini çalıştırın:
```bash
python Golub_EDA_DataAnalysis.py

Aynı adımları Breast-data-set için de uygulayıp
Önce Brest_test.py ardından Golub_EDA_DataAnalysis.py çalıştırılmalıdır.