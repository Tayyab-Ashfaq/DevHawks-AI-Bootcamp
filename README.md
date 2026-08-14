# 🌿 Plant Disease Dataset

A structured tabular dataset for **binary classification** of plant disease presence, based on environmental and soil conditions. This dataset is suitable for machine learning experiments, data preprocessing practice, and agricultural AI research.

---

## 📋 Dataset Overview

| Property | Value |
|---|---|
| **File** | `plant_disease_dataset.csv` |
| **Format** | CSV (Comma-Separated Values) |
| **Rows** | 10,000 |
| **Columns** | 5 |
| **Task Type** | Binary Classification |
| **Missing Values** | None |
| **Target Column** | `disease_present` |

---

## 📁 File Structure

```
plant_disease_dataset.csv
├── temperature       → float64  (°C)
├── humidity          → float64  (%)
├── rainfall          → float64  (mm)
├── soil_pH           → float64  (pH scale)
└── disease_present   → int64    (0 = No Disease, 1 = Disease)
```

---

## 🔢 Feature Descriptions

| Feature | Type | Unit | Description |
|---|---|---|---|
| `temperature` | Continuous | °C | Ambient temperature at the time of observation |
| `humidity` | Continuous | % | Relative humidity in the environment |
| `rainfall` | Continuous | mm | Rainfall recorded over the observation period |
| `soil_pH` | Continuous | pH (4.0–8.5) | Acidity/alkalinity level of the soil |
| `disease_present` | Binary | 0 / 1 | Target label — 1 = disease present, 0 = no disease |

---

## 📊 Statistical Summary

| Feature | Min | Max | Mean | Std Dev | 25th % | Median | 75th % |
|---|---|---|---|---|---|---|---|
| `temperature` | 5.39 | 56.69 | 25.61 | 5.81 | 21.82 | 25.29 | 28.92 |
| `humidity` | 6.24 | 102.40 | 62.12 | 22.68 | 40.77 | 72.21 | 81.03 |
| `rainfall` | 0.00 | 84.65 | 9.81 | 9.85 | 2.81 | 6.87 | 13.44 |
| `soil_pH` | 4.00 | 8.50 | 6.25 | 1.30 | 5.12 | 6.23 | 7.39 |

---

## 🎯 Target Class Distribution

| Class | Label | Count | Percentage |
|---|---|---|---|
| `0` | No Disease | 7,590 | 75.9% |
| `1` | Disease Present | 2,410 | 24.1% |

> ⚠️ **Class Imbalance Note:** The dataset is imbalanced (~76% negative, ~24% positive). When training classification models, consider techniques such as:
> - **SMOTE** (Synthetic Minority Oversampling Technique)
> - **Class weighting** (`class_weight='balanced'` in scikit-learn)
> - Using **Precision, Recall, F1-Score, and AUC-ROC** as evaluation metrics (not just Accuracy)

---

## 🔍 Data Quality

| Check | Status |
|---|---|
| Missing Values | ✅ None |
| Duplicate Rows | ✅ Not detected |
| Negative Rainfall | ✅ None (min = 0.00) |
| pH Out of Range | ✅ All within 4.0–8.5 |
| Outliers | ⚠️ Possible — `temperature` max (56.69°C) and `humidity` max (102.40%) warrant investigation |

---

## 🚀 Suggested Use Cases

**Machine Learning Tasks**
- Binary classification (disease vs. no disease)
- Feature importance analysis (which environmental factor most predicts disease?)
- Threshold tuning for precision/recall trade-off

**Data Preprocessing Practice**
- Outlier detection and removal (IQR method, Z-score)
- Feature scaling (StandardScaler, MinMaxScaler)
- Handling class imbalance (SMOTE, undersampling)
- Exploratory Data Analysis (EDA) and visualization

**AI-900 / ML Education**
- Demonstrates supervised learning with labeled data
- Binary classification problem with a real-world context
- Good for practicing evaluation metrics: Accuracy, Precision, Recall, F1, AUC-ROC

---

## 💻 Quick Start (Python)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('plant_disease_dataset.csv')

# Separate features and target
X = df.drop('disease_present', axis=1)
y = df['disease_present']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Training set : {X_train.shape}")
print(f"Test set     : {X_test.shape}")
print(f"Disease rate : {y.mean():.1%}")
```

---

## 📈 Recommended Evaluation Metrics

Since the dataset is imbalanced, prioritize these metrics over plain accuracy:

| Metric | Why It Matters Here |
|---|---|
| **Precision** | Avoid false disease alarms — unnecessary treatment costs |
| **Recall** | Avoid missing actual disease — crop loss risk |
| **F1 Score** | Balance between Precision and Recall |
| **AUC-ROC** | Overall model discrimination ability across thresholds |
| **Confusion Matrix** | Full breakdown of TP, TN, FP, FN |

---

## 🌱 Domain Context

Plant disease detection is a critical application of AI in **precision agriculture**. Environmental factors like temperature, humidity, rainfall, and soil pH are well-established predictors of fungal, bacterial, and viral plant diseases. Machine learning models trained on such features can help farmers:

- Detect disease risk **before** visible symptoms appear
- Optimize resource use (pesticides, irrigation)
- Reduce crop loss through early intervention

---

## 📝 Notes

- The dataset is **synthetically generated** for educational and experimental purposes
- `humidity` values above 100% in the dataset may represent sensor noise or dew point anomalies — consider capping at 100% during preprocessing
- `temperature` values above 45°C are extreme but not impossible in certain climates — treat as potential outliers and investigate before removing
- For reproducible experiments, set `random_state=42` in train/test splits

---

## 📂 Suggested Project Structure

```
plant-disease-ml/
├── data/
│   └── plant_disease_dataset.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocess.py
│   └── train.py
├── outputs/
│   └── figures/
└── README.md
```

---

*Dataset intended for educational and research use.*
