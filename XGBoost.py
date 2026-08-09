import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Step 1: Load and Clean Data --------------------

# Load the data, treating -1E+31 as missing
df = pd.read_csv("merged_data.csv", na_values='-1E+31')

# Replace blank CME values with 0 and ensure int type
df['CME'] = df['CME'].fillna(0).astype(int)

# Drop the timestamp column (not useful for prediction)
if 'epoch_for_cdf_mod' in df.columns:
    df.drop(columns=['epoch_for_cdf_mod'], inplace=True)

# Drop columns with more than 95% missing values
missing_ratios = df.isna().mean()
cols_to_drop = missing_ratios[missing_ratios > 0.95].index
df.drop(columns=cols_to_drop, inplace=True)

print(f"\nDropped columns due to >95% missing:\n{list(cols_to_drop)}\n")

# Drop rows with any remaining NaN values
df_clean = df.dropna()

print(f"Cleaned data shape: {df_clean.shape}")

# -------------------- Step 2: Feature Scaling --------------------

# Separate features and target
X = df_clean.drop(columns=['CME', 'HALO'])
y = df_clean['CME']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------- Step 3: Train-Validation-Test Split --------------------

# First split: train+val vs test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Second split: train vs val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42  # 0.25 * 0.8 = 0.2
)

# -------------------- Step 4: Train XGBoost Classifier --------------------

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# -------------------- Step 5: Validation Set Evaluation --------------------

y_val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, y_val_pred)
val_cm = confusion_matrix(y_val, y_val_pred)

print(f"\nValidation Accuracy: {val_acc * 100:.2f}%")
print("Validation Confusion Matrix:\n", val_cm)

plt.figure(figsize=(6, 5))
sns.heatmap(val_cm, annot=True, fmt='d', cmap='Greens', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title('Confusion Matrix (Validation Set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# -------------------- Step 6: Test Set Evaluation --------------------

y_test_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_test_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print("Test Confusion Matrix:\n", test_cm)

plt.figure(figsize=(6, 5))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# -------------------- Step 7: Feature Importance --------------------

importances = model.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='plasma')
plt.title("Feature Importance (XGBoost) for CME Prediction")
plt.tight_layout()
plt.show()
