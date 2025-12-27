import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("data/stroke_data.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)

# ---------------------------
# 2. Basic Info & Missing Values
# ---------------------------
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

# ---------------------------
# 3. Class Imbalance Check
# ---------------------------
print("\n--- Stroke Class Distribution ---")
print(df['stroke'].value_counts())
print("\n--- Stroke Class Ratio ---")
print(df['stroke'].value_counts(normalize=True))

plt.figure(figsize=(5,4))
sns.countplot(x='stroke', data=df)
plt.title("Original Stroke Class Distribution (Highly Imbalanced)")
plt.show()

# ---------------------------
# 4. Numerical Feature Distributions
# ---------------------------
num_cols = ['age', 'avg_glucose_level', 'bmi']
df[num_cols].hist(figsize=(10,5))
plt.suptitle("Numerical Feature Distributions")
plt.show()

# ---------------------------
# 5. Feature Type Separation
# ---------------------------
target = 'stroke'

categorical_cols = df.select_dtypes(include='object').columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove(target)

print("\nCategorical Columns:", categorical_cols)
print("Numerical Columns:", numerical_cols)

# ---------------------------
# 6. Missing Value Imputation
# ---------------------------
for col in numerical_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after imputation:")
print(df.isnull().sum())

# ---------------------------
# 7. Encode Categorical Features
# ---------------------------
from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("\nCategorical encoding completed.")

# ---------------------------
# 8. Scale Numerical Features
# ---------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

print("\nNumerical feature scaling completed.")


# ---------------------------
# 9. Train / Validation / Test Split
# ---------------------------
from sklearn.model_selection import train_test_split

X = df.drop(columns=['stroke'])
y = df['stroke']

# First split: Train (70%) + Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# Second split: Validation (15%) + Test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=42
)

print("\n--- Dataset Split Sizes ---")
print("Training set:", X_train.shape, y_train.value_counts())
print("Validation set:", X_val.shape, y_val.value_counts())
print("Test set:", X_test.shape, y_test.value_counts())


# ---------------------------
# 10. Apply SMOTE (Training Set ONLY)
# ---------------------------
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("\n--- After SMOTE (Training Set Only) ---")
print("X_train_smote shape:", X_train_smote.shape)
print(y_train_smote.value_counts())

# ---------------------------
# 11. Logistic Regression (Baseline Model)
# ---------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')

log_reg.fit(X_train_smote, y_train_smote)

# Validation predictions
y_val_pred = log_reg.predict(X_val)
y_val_prob = log_reg.predict_proba(X_val)[:, 1]

print("\n--- Logistic Regression (Validation Results) ---")
print(classification_report(y_val, y_val_pred))
print("ROC-AUC:", roc_auc_score(y_val, y_val_prob))

# ---------------------------
# 12. Random Forest Model
# ---------------------------
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight='balanced',
    random_state=42
)

rf.fit(X_train_smote, y_train_smote)

y_val_pred_rf = rf.predict(X_val)
y_val_prob_rf = rf.predict_proba(X_val)[:, 1]

print("\n--- Random Forest (Validation Results) ---")
print(classification_report(y_val, y_val_pred_rf))
print("ROC-AUC:", roc_auc_score(y_val, y_val_prob_rf))

# ---------------------------
# 13. Threshold Tuning (Random Forest)
# ---------------------------
import numpy as np

threshold = 0.30  # lower threshold for medical recall
y_val_pred_rf_tuned = (y_val_prob_rf >= threshold).astype(int)

print("\n--- Random Forest (Threshold Tuned @ 0.30) ---")
print(classification_report(y_val, y_val_pred_rf_tuned))
print("ROC-AUC:", roc_auc_score(y_val, y_val_prob_rf))

# ---------------------------
# 14. XGBoost Model
# ---------------------------
from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train_smote.value_counts()[0] / y_train_smote.value_counts()[1]),
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train_smote, y_train_smote)

y_val_pred_xgb = xgb_model.predict(X_val)
y_val_prob_xgb = xgb_model.predict_proba(X_val)[:, 1]

print("\n--- XGBoost (Validation Results) ---")
print(classification_report(y_val, y_val_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_val, y_val_prob_xgb))

# ---------------------------
# 15. Threshold Tuning (XGBoost)
# ---------------------------
threshold_xgb = 0.25  # aggressive threshold for medical recall

y_val_pred_xgb_tuned = (y_val_prob_xgb >= threshold_xgb).astype(int)

print("\n--- XGBoost (Threshold Tuned @ 0.25) ---")
print(classification_report(y_val, y_val_pred_xgb_tuned))
print("ROC-AUC:", roc_auc_score(y_val, y_val_prob_xgb))

# ---------------------------
# 16. Soft Voting Ensemble
# ---------------------------

# Get validation probabilities
prob_lr = log_reg.predict_proba(X_val)[:, 1]
prob_rf = rf.predict_proba(X_val)[:, 1]
prob_xgb = xgb_model.predict_proba(X_val)[:, 1]

# Weighted average (favor LR + RF)
ensemble_prob = (
    0.4 * prob_lr +
    0.4 * prob_rf +
    0.2 * prob_xgb
)

ensemble_threshold = 0.30
ensemble_pred = (ensemble_prob >= ensemble_threshold).astype(int)

print("\n--- Ensemble Model (Validation Results) ---")
print(classification_report(y_val, ensemble_pred))
print("ROC-AUC:", roc_auc_score(y_val, ensemble_prob))


# ---------------------------
# 17. Final Test Set Evaluation (Ensemble)
# ---------------------------

# Get test probabilities
prob_lr_test = log_reg.predict_proba(X_test)[:, 1]
prob_rf_test = rf.predict_proba(X_test)[:, 1]
prob_xgb_test = xgb_model.predict_proba(X_test)[:, 1]

# Ensemble probability
ensemble_prob_test = (
    0.4 * prob_lr_test +
    0.4 * prob_rf_test +
    0.2 * prob_xgb_test
)

ensemble_test_pred = (ensemble_prob_test >= ensemble_threshold).astype(int)

print("\n--- FINAL TEST SET RESULTS (Ensemble Model) ---")
print(classification_report(y_test, ensemble_test_pred))
print("ROC-AUC:", roc_auc_score(y_test, ensemble_prob_test))
