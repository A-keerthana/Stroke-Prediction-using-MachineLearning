# 🧠 Stroke Risk Prediction using Machine Learning

## 📌 Overview
Stroke is one of the leading causes of death and long-term disability worldwide.  
This project builds a **machine learning–based stroke risk prediction system** using clinical and demographic data to assist in early risk identification.

The goal is to analyze patient attributes such as age, hypertension, heart disease, BMI, glucose levels, and lifestyle factors to predict the likelihood of a stroke.

---

## 📂 Dataset
- **Source:** Public healthcare dataset (Kaggle)
- **Target Variable:** `stroke` (0 = No Stroke, 1 = Stroke)
- **Records:** ~5,000+
- **Features include:**
  - Age
  - Gender
  - Hypertension
  - Heart disease
  - BMI
  - Average glucose level
  - Smoking status

> ⚠️ Dataset is not included in the repository to avoid large file uploads.

---

## 🔍 Exploratory Data Analysis (EDA)
- Checked for missing values and data imbalance
- Visualized stroke vs non-stroke distribution
- Analyzed correlations between risk factors and stroke occurrence
- Applied appropriate preprocessing techniques

---

## ⚙️ Preprocessing Steps
- Missing value handling (median imputation for BMI)
- Label encoding for categorical features
- Feature scaling using StandardScaler
- Train-test split (80/20)

---

## 🤖 Models Implemented
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

---

## 📊 Model Evaluation
Evaluation was performed using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC score

| Model              | Accuracy | ROC-AUC |
|--------------------|----------|---------|
| Logistic Regression| 78%      | 0.81    |
| Random Forest      | 85%      | 0.88    |
| SVM                | 82%      | 0.84    |

✅ **Random Forest performed best overall**

---

## 📈 Results Visualization
- Confusion Matrix
- ROC Curve
- Feature Importance Plot

All plots are generated programmatically using Matplotlib and Seaborn.

---

## 🔁 How to Reproduce This Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/AK-stroke-risk-prediction.git
cd AK-stroke-risk-prediction
