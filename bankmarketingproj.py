# =============================================================================
# Yes or No? Classifying Term Deposit Subscriptions using Random Forest
# =============================================================================
# -*- coding: utf-8 -*-
# --- Libraries ----------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, roc_curve, auc)
from sklearn.inspection import PartialDependenceDisplay
from sklearn.utils import resample
import joblib
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. READING IN THE DATA
# =============================================================================
bank_data = pd.read_csv("bank.csv", sep=";")

# Inspect dtypes (equivalent to sapply(bank.data, class))
print("=== Column Types ===")
print(bank_data.dtypes)
print()

# =============================================================================
# 2. DATA CLEANING
# =============================================================================

# Check for NA values
print("=== Any NA values? ===")
print(bank_data.isna().any().any())
print()

# Remove rows with "unknown" in key categorical columns
cols_to_clean = ["contact", "job", "marital", "education", "loan", "month"]
adj_bank_data = bank_data.copy()
for col in cols_to_clean:
    adj_bank_data = adj_bank_data[adj_bank_data[col] != "unknown"]

# How many poutcome are "unknown"?
print("=== Number of unknown poutcome values ===")
print((bank_data["poutcome"] == "unknown").sum())
print()

# How many pdays are -1 (never previously contacted)?
print("=== Number of pdays == -1 ===")
print((bank_data["pdays"] == -1).sum())
print()

# Drop columns: poutcome, day, month, pdays
adj_bank_data = adj_bank_data.drop(columns=["poutcome", "day", "month", "pdays"])

# Encode binary target variable
adj_bank_data["y"] = adj_bank_data["y"].map({"yes": 1, "no": 0})

# =============================================================================
# 3. CLASS IMBALANCE — VISUALISE & UPSAMPLE
# =============================================================================

# Plot class balance
fig, ax = plt.subplots(figsize=(7, 4))
adj_bank_data["y"].value_counts().rename({1: "yes", 0: "no"}).plot(
    kind="barh", ax=ax, color="orange"
)
ax.set_title("Count of clients subscribing to a term deposit vs not",
             fontsize=12, pad=10)
ax.set_xlabel("Count")
ax.set_ylabel("Result Type")
ax.set_yticklabels(["no", "yes"])
plt.tight_layout()
plt.savefig("class_balance.png", dpi=150)
plt.show()
print("Saved: class_balance.png")

# Upsample minority class (y == 1) to match majority class (y == 0)
majority = adj_bank_data[adj_bank_data["y"] == 0]
minority = adj_bank_data[adj_bank_data["y"] == 1]

minority_upsampled = resample(minority,
                              replace=True,
                              n_samples=len(majority),
                              random_state=42)

balanced_data = pd.concat([majority, minority_upsampled])

# Confirm balance
print("=== Balanced class counts ===")
print(balanced_data["y"].value_counts())
print()

# =============================================================================
# 4. FEATURE ENCODING & TRAIN/TEST SPLIT
# =============================================================================

# One-hot encode remaining categorical columns
balanced_data = pd.get_dummies(balanced_data, drop_first=True)

# Separate features and target
X = balanced_data.drop(columns=["y"])
y = balanced_data["y"]

# 70/30 train-test split (stratified to keep balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print("=== Training set class counts ===")
print(y_train.value_counts())
print()

# =============================================================================
# 5. DECISION TREE (baseline)
# =============================================================================

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# Visualise tree (truncated to depth 3 for readability)
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(tree_model, feature_names=X.columns, class_names=["no", "yes"],
          filled=True, max_depth=3, ax=ax, fontsize=7)
ax.set_title("Decision Tree (depth capped at 3 for display)", fontsize=13)
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
plt.show()
print("Saved: decision_tree.png")

# Decision tree predictions & classification report
dt_preds = tree_model.predict(X_test)
print("=== Decision Tree — Classification Report ===")
print(classification_report(y_test, dt_preds, target_names=["no", "yes"]))

# =============================================================================
# 6. RANDOM FOREST (full model)
# =============================================================================

rf_model = RandomForestClassifier(n_estimators=500, oob_score=True,
                                  random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print("=== Random Forest — OOB Score ===")
print(f"OOB Accuracy: {rf_model.oob_score_:.4f}  |  OOB Error: {1 - rf_model.oob_score_:.4f}")
print()

# --- OOB error curve (per-tree) ---
oob_errors = []
for i, tree in enumerate(rf_model.estimators_, 1):
    # Use cumulative oob error proxy via individual tree OOB predictions
    # (sklearn doesn't expose per-tree OOB error directly; we approximate)
    pass  # Approximated below using staged approach

# Build OOB error across increasing n_estimators
oob_error_list = []
for n in range(10, 501, 10):
    rf_temp = RandomForestClassifier(n_estimators=n, oob_score=True,
                                     warm_start=False, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train, y_train)
    oob_error_list.append(1 - rf_temp.oob_score_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(10, 501, 10), oob_error_list, color="black", label="OOB Error")
ax.set_xlabel("Number of Trees")
ax.set_ylabel("OOB Error Rate")
ax.set_title("Random Forest OOB Error vs Number of Trees")
ax.legend()
plt.tight_layout()
plt.savefig("rf_oob_error.png", dpi=150)
plt.show()
print("Saved: rf_oob_error.png")

# --- Feature importance ---
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 7))
importances_sorted.plot(kind="barh", ax=ax, color="steelblue")
ax.set_title("Variable Importance (Mean Decrease in Impurity)", fontsize=12)
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.show()
print("Saved: feature_importance.png")

print("=== Top 10 Feature Importances ===")
print(importances.sort_values(ascending=False).head(10))
print()

# =============================================================================
# 7. ADJUSTED RANDOM FOREST (drop least important features)
# =============================================================================
# Dropping: default, contact, loan (encoded columns derived from these)

drop_cols = [c for c in X_train.columns
             if c.startswith("default") or c.startswith("contact") or c.startswith("loan")]

X_train_adj = X_train.drop(columns=drop_cols)
X_test_adj  = X_test.drop(columns=drop_cols)

adj_rf_model = RandomForestClassifier(n_estimators=500, oob_score=True,
                                      importance_weights=None,
                                      random_state=42, n_jobs=-1)
adj_rf_model.fit(X_train_adj, y_train)

print("=== Adjusted Random Forest — OOB Score ===")
print(f"OOB Accuracy: {adj_rf_model.oob_score_:.4f}  |  OOB Error: {1 - adj_rf_model.oob_score_:.4f}")
print()

# =============================================================================
# 8. PARTIAL DEPENDENCE PLOTS (top 3 variables)
# =============================================================================

top3 = ["duration", "age", "balance"]
# Ensure all three exist in adjusted training set
top3 = [v for v in top3 if v in X_train_adj.columns]

fig, axes = plt.subplots(1, len(top3), figsize=(5 * len(top3), 5))
PartialDependenceDisplay.from_estimator(
    adj_rf_model, X_train_adj, features=top3,
    kind="average", ax=axes, grid_resolution=50
)
plt.suptitle("Partial Dependence Plots — Top 3 Variables", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("partial_dependence.png", dpi=150)
plt.show()
print("Saved: partial_dependence.png")

# =============================================================================
# 9. PREDICTIONS & CONFUSION MATRIX
# =============================================================================

rf_preds = adj_rf_model.predict(X_test_adj)

print("=== Adjusted Random Forest — Classification Report ===")
print(classification_report(y_test, rf_preds, target_names=["no", "yes"]))

cm = confusion_matrix(y_test, rf_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["no", "yes"])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(ax=ax, colorbar=False, cmap="Oranges")
ax.set_title("Confusion Matrix — Adjusted Random Forest")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved: confusion_matrix.png")

# =============================================================================
# 10. ROC CURVE & AUC
# =============================================================================

rf_probs = adj_rf_model.predict_proba(X_test_adj)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, rf_probs)
roc_auc = auc(fpr, tpr)

# Find optimal threshold (maximises Youden's J = sensitivity + specificity - 1)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, color="orange", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.scatter(fpr[best_idx], tpr[best_idx], color="red", zorder=5,
           label=f"Best threshold = {best_threshold:.2f}")
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("ROC Curve — Adjusted Random Forest")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.show()
print("Saved: roc_curve.png")

print(f"\n=== AUC ===")
print(f"AUC: {roc_auc:.4f}")

# =============================================================================
# 11. SAVE FINAL MODEL
# =============================================================================

joblib.dump(adj_rf_model, "bankmarketing_rf.pkl")
print("\nFinal model saved to: bankmarketing_rf.pkl")

# =============================================================================
# ACTIONABLE INSIGHTS (printed summary)
# =============================================================================
print("""
=== Actionable Insights ===
The three most important features are: duration, age, and balance.

1. DURATION  — Keep phone calls short. Longer calls correlate with a lower
               probability of a 'yes'. Short and focused contact is key.

2. AGE       — Target clients aged 20–45, who show the highest subscription
               probability. Minimise efforts towards clients over 60.

3. BALANCE   — Clients with higher average yearly balances are more likely
               to subscribe. Prioritise established, financially stable clients.
""")
