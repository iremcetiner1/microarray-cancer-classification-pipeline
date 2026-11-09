import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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