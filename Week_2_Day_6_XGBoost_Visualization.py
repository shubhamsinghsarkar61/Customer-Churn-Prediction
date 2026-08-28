import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# DAY 6: XGBOOST MODEL VISUALIZATION
# Customer Churn Prediction Project
# ============================================================

# Load prediction results
df = pd.read_csv("06_xgboost_predictions.csv")

print("=" * 60)
print("DAY 6 - XGBOOST MODEL VISUALIZATION")
print("=" * 60)
print(f"Prediction records loaded: {len(df)}")


# ============================================================
# CALCULATE MODEL METRICS
# ============================================================

y_true = df["Actual_Churn"]
y_pred = df["Predicted_Churn"]

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

correct_predictions = (y_true == y_pred).sum()

print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2%}")


# ============================================================
# VISUALIZATION 1: CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot(ax=ax, values_format="d")

plt.title(
    "Day 6: XGBoost Confusion Matrix",
    fontsize=16,
    fontweight="bold",
    pad=15
)

plt.xlabel("Predicted Churn Status", fontsize=12)
plt.ylabel("Actual Churn Status", fontsize=12)

plt.tight_layout()
plt.savefig(
    "06_01_xgboost_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Created: 06_01_xgboost_confusion_matrix.png")


# ============================================================
# VISUALIZATION 2: ACTUAL VS PREDICTED DISTRIBUTION
# ============================================================

actual_counts = y_true.value_counts().sort_index()
predicted_counts = y_pred.value_counts().sort_index()

comparison = pd.DataFrame({
    "Actual": actual_counts,
    "Predicted": predicted_counts
})

comparison.index = ["No Churn", "Churn"]

ax = comparison.plot(
    kind="bar",
    figsize=(9, 6),
    width=0.75
)

plt.title(
    "Day 6: Actual vs Predicted Customer Churn",
    fontsize=16,
    fontweight="bold",
    pad=15
)

plt.xlabel("Customer Status", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.xticks(rotation=0)

# Add values above bars
for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.tight_layout()
plt.savefig(
    "06_02_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Created: 06_02_actual_vs_predicted.png")


# ============================================================
# VISUALIZATION 3: MODEL PERFORMANCE DASHBOARD
# ============================================================

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
}

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(
    list(metrics.keys()),
    list(metrics.values())
)

ax.set_title(
    "Day 6: XGBoost Model Performance",
    fontsize=16,
    fontweight="bold",
    pad=15
)

ax.set_xlabel("Evaluation Metrics", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 1)

# Add percentage labels
for bar, value in zip(bars, metrics.values()):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.2%}",
        ha="center",
        fontsize=11,
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig(
    "06_03_xgboost_performance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("Created: 06_03_xgboost_performance.png")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DAY 6 VISUALIZATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Total Test Records      : {len(df)}")
print(f"Correct Predictions     : {correct_predictions}")
print(f"Accuracy                : {accuracy:.2%}")
print(f"Precision               : {precision:.2%}")
print(f"Recall                  : {recall:.2%}")
print(f"F1-Score                : {f1:.2%}")

print("\nGenerated Files:")
print("1. 06_01_xgboost_confusion_matrix.png")
print("2. 06_02_actual_vs_predicted.png")
print("3. 06_03_xgboost_performance.png")

print("\nDay 6 XGBoost Visualization completed successfully.")