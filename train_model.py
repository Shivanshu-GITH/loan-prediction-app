"""
Loan Approval Prediction - Model Training Script
Author : Shivanshu Tiwari (Roll No: 2400270130173)

Steps demonstrated:
1. Data Loading
2. Data Preprocessing (missing values, duplicates, outlier check, scaling)
3. Model Training (Decision Tree Classifier)
4. Model Evaluation (accuracy, precision, recall, F1, confusion matrix)
5. Saving the trained model for deployment
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. DATA LOADING
# ------------------------------------------------------------------
print("=" * 60)
print("STEP 1: DATA LOADING")
print("=" * 60)

df = pd.read_csv("loan_data.csv")
print(f"Dataset shape : {df.shape}")
print(f"Columns       : {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

# ------------------------------------------------------------------
# 2. DATA PREPROCESSING
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: DATA PREPROCESSING")
print("=" * 60)

# 2a. Missing values
print("\nMissing values per column:")
print(df.isnull().sum())
if df.isnull().sum().sum() > 0:
    df = df.dropna()
    print(f"Rows after dropping missing values: {len(df)}")
else:
    print("No missing values found.")

# 2b. Duplicates
dupes = df.duplicated().sum()
print(f"\nDuplicate rows found: {dupes}")
if dupes > 0:
    df = df.drop_duplicates()
    print(f"Rows after removing duplicates: {len(df)}")

# 2c. Basic statistics / outlier awareness
print("\nStatistical summary:")
print(df.describe().round(2))

# 2d. Class balance
print("\nTarget class distribution (Approved):")
print(df["Approved"].value_counts())
print((df["Approved"].value_counts(normalize=True) * 100).round(2).astype(str) + " %")

# 2e. Feature / target split
FEATURES = ["Age", "Income", "LoanAmount", "CreditScore"]
TARGET = "Approved"
X = df[FEATURES]
y = df[TARGET]

# 2f. Train-test split (stratified to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# 2g. Feature scaling
# (Decision Trees do not strictly require scaling, but it is included
#  here to demonstrate the preprocessing step and keep the pipeline
#  reusable for other models.)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nFeatures scaled with StandardScaler (fit on train set only).")

# ------------------------------------------------------------------
# 3. MODEL TRAINING - DECISION TREE CLASSIFIER
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: MODEL TRAINING (Decision Tree Classifier)")
print("=" * 60)

# Hyperparameter tuning with GridSearchCV to avoid overfitting
param_grid = {
    "max_depth": [3, 5, 7, 10, 15],
    "min_samples_split": [2, 10, 50],
    "min_samples_leaf": [1, 5, 20],
    "criterion": ["gini", "entropy"],
    # class_weight handles the imbalanced target (~22% approved vs ~78% rejected)
    "class_weight": [None, "balanced"],
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)
grid.fit(X_train_scaled, y_train)

model = grid.best_estimator_
print(f"Best parameters      : {grid.best_params_}")
print(f"Best CV F1 score     : {grid.best_score_:.4f}")

# ------------------------------------------------------------------
# 4. MODEL EVALUATION
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: MODEL EVALUATION")
print("=" * 60)

y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save confusion matrix plot
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["Rejected", "Approved"]).plot(ax=ax, cmap="Blues")
ax.set_title("Confusion Matrix - Decision Tree")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120)
plt.close()
print("\nSaved: confusion_matrix.png")

# Feature importance
print("\nFeature Importances:")
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(importances.round(4))

fig, ax = plt.subplots(figsize=(7, 4))
importances.plot(kind="barh", ax=ax, color="#2b7bba")
ax.set_title("Feature Importance - Decision Tree")
ax.set_xlabel("Importance")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.close()
print("Saved: feature_importance.png")

# Tree visualization (top levels only, for readability)
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    model,
    feature_names=FEATURES,
    class_names=["Rejected", "Approved"],
    filled=True,
    max_depth=3,
    fontsize=8,
    ax=ax,
)
ax.set_title("Decision Tree (top 3 levels)")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=120)
plt.close()
print("Saved: decision_tree.png")

# ------------------------------------------------------------------
# 5. SAVE MODEL FOR DEPLOYMENT
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 5: SAVING MODEL FOR DEPLOYMENT")
print("=" * 60)

joblib.dump(model, "loan_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Saved: loan_model.pkl (trained Decision Tree)")
print("Saved: scaler.pkl (fitted StandardScaler)")
print("\nTraining complete. Run the app with:  streamlit run app.py")
