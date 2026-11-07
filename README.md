# 🧬 Microarray Cancer Classification Pipeline (ALL vs. AML)

End-to-end exploratory data analysis (EDA), statistical bioinformatics, and leakage-free machine learning pipeline for high-dimensional microarray gene expression subtyping (**Acute Lymphoblastic Leukemia [ALL]** vs. **Acute Myeloid Leukemia [AML]**) based on the landmark **Golub et al. (1999)** dataset.

---

## 📌 Proje Özeti ve Problem Tanımı
Mikroarray gen ifade verileri, örneklem sayısının (n) öznitelik/gen sayısından (p) çok daha küçük olduğu (p>> n, *"Curse of Dimensionality"*) yüksek boyutlu biyomedikal verilerdir. Bu projede:
* **7.129 gen (öznitelik)** ve **72 hasta örneği** (38 Eğitim, 34 Bağımsız Test) incelenmiştir.
* Ham floresan yoğunluklarındaki arka plan gürültüsü (negatif değerler), çarpık dağılımlar ve **Batch Etkisi (Eğitim vs. Test dağılım farkları)** analiz edilmiştir.
---

## 📂 Klasör Yapısı
microarray-cancer-classification-pipeline/
│
├── golub-data-set/
│   ├── data_set_ALL_AML_train.csv         # Eğitim seti (38 hasta x 7129 gen + call sütunları)
│   ├── data_set_ALL_AML_independent.csv   # Bağımsız test seti (34 hasta x 7129 gen + call sütunları)
│   └── actual.csv                         # Hasta ID (1-72) ve gerçek kanser alt tipi etiketleri (ALL / AML)
│
├── Golub_EDA_DataAnalysis.py              # Uçtan uca EDA, İstatistiksel Testler ve Modelleme betiği
├── requirements.txt                       # Gerekli Python kütüphaneleri (opsiyonel)
└── README.md                              # Proje dokümantasyonu ve çalıştırma rehberi
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