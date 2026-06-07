# train_model.py

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================
# CREATE REQUIRED FOLDERS
# =====================================================

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("images", exist_ok=True)

# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("data/mobile_price.csv")

print(f"\nDataset Shape: {df.shape}")

# =====================================================
# BASIC DATASET CHECKS
# =====================================================

print("\n" + "=" * 60)
print("Dataset Information")
print("=" * 60)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nClass Distribution:")
print(df["price_range"].value_counts())

# =====================================================
# FEATURE SELECTION
# Removed low-importance features:
# blue, dual_sim, four_g, three_g,
# touch_screen, wifi
# =====================================================

selected_features = [
    "battery_power",
    "clock_speed",
    "fc",
    "int_memory",
    "m_dep",
    "mobile_wt",
    "n_cores",
    "pc",
    "px_height",
    "px_width",
    "ram",
    "sc_h",
    "sc_w",
    "talk_time"
]

X = df[selected_features]
y = df["price_range"]

print("\nSelected Features:")
print(selected_features)

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# Save feature list for Streamlit

joblib.dump(
    selected_features,
    "models/selected_features.pkl"
)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# =====================================================
# DECISION TREE MODEL
# =====================================================

print("\n" + "=" * 60)
print("Training Decision Tree...")
print("=" * 60)

dt_model = DecisionTreeClassifier(
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

dt_model.fit(X_train, y_train)

# Predictions

dt_pred = dt_model.predict(X_test)

# Accuracy

dt_accuracy = accuracy_score(
    y_test,
    dt_pred
)

print(f"\nDecision Tree Accuracy: {dt_accuracy:.4f}")

# Classification Report

dt_report = classification_report(
    y_test,
    dt_pred,
    output_dict=True
)

dt_report_df = pd.DataFrame(dt_report).transpose()

dt_report_df.to_csv(
    "reports/classification_report_dt.csv"
)

# Confusion Matrix

dt_cm = confusion_matrix(
    y_test,
    dt_pred
)

pd.DataFrame(dt_cm).to_csv(
    "reports/confusion_matrix_dt.csv",
    index=False
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=dt_cm
)

disp.plot()

plt.title("Decision Tree Confusion Matrix")

plt.savefig(
    "images/confusion_matrix_dt.png",
    bbox_inches="tight"
)

plt.close()

# Save Model

joblib.dump(
    dt_model,
    "models/decision_tree_model.pkl"
)

# =====================================================
# RANDOM FOREST MODEL
# =====================================================

print("\n" + "=" * 60)
print("Training Random Forest...")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Predictions

rf_pred = rf_model.predict(X_test)

# Accuracy

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print(f"\nRandom Forest Accuracy: {rf_accuracy:.4f}")

# Classification Report

rf_report = classification_report(
    y_test,
    rf_pred,
    output_dict=True
)

rf_report_df = pd.DataFrame(rf_report).transpose()

rf_report_df.to_csv(
    "reports/classification_report_rf.csv"
)

# Confusion Matrix

rf_cm = confusion_matrix(
    y_test,
    rf_pred
)

pd.DataFrame(rf_cm).to_csv(
    "reports/confusion_matrix_rf.csv",
    index=False
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm
)

disp.plot()

plt.title("Random Forest Confusion Matrix")

plt.savefig(
    "images/confusion_matrix_rf.png",
    bbox_inches="tight"
)

plt.close()

# Save Model

joblib.dump(
    rf_model,
    "models/random_forest_model.pkl"
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

print("\n" + "=" * 60)
print("Generating Feature Importance...")
print("=" * 60)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

feature_importance.to_csv(
    "reports/feature_importance.csv",
    index=False
)

# Plot Feature Importance

plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "images/feature_importance.png",
    bbox_inches="tight"
)

plt.close()

# =====================================================
# MODEL COMPARISON
# =====================================================

comparison_df = pd.DataFrame({
    "Model": [
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy": [
        dt_accuracy,
        rf_accuracy
    ]
})

comparison_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nModel Comparison")
print(comparison_df)

best_model = (
    "Random Forest"
    if rf_accuracy > dt_accuracy
    else "Decision Tree"
)

print(f"\nBest Model: {best_model}")

print("\nSaved Files:")
print("✔ models/decision_tree_model.pkl")
print("✔ models/random_forest_model.pkl")
print("✔ models/selected_features.pkl")
print("✔ reports/")
print("✔ images/")