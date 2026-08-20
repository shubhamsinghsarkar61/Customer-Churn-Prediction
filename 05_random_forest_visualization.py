import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ==========================================
# LOAD PREDICTIONS
# ==========================================

df = pd.read_csv("05_random_forest_predictions.csv")

print("Prediction data loaded successfully.")
print("Total records:", len(df))


# ==========================================
# 1. CONFUSION MATRIX
# ==========================================

y_true = df["Actual_Churn"]
y_pred = df["Predicted_Churn"]

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

fig, ax = plt.subplots(figsize=(7, 5))

disp.plot(ax=ax)

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig("05_01_confusion_matrix.png", dpi=300)

plt.close()

print("Created: 05_01_confusion_matrix.png")


# ==========================================
# 2. ACTUAL VS PREDICTED CHURN
# ==========================================

comparison = pd.DataFrame({
    "Actual Churn": y_true.value_counts().sort_index(),
    "Predicted Churn": y_pred.value_counts().sort_index()
})

comparison.index = ["No Churn", "Churn"]

comparison.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Actual vs Predicted Customer Churn")
plt.xlabel("Customer Status")
plt.ylabel("Number of Customers")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("05_02_actual_vs_predicted.png", dpi=300)

plt.close()

print("Created: 05_02_actual_vs_predicted.png")


# ==========================================
# 3. MODEL PERFORMANCE METRICS
# ==========================================

metrics = {
    "Accuracy": 0.78,
    "Precision": 0.6149,
    "Recall": 0.4866,
    "F1-Score": 0.5433
}

metric_names = list(metrics.keys())

metric_values = [
    value * 100
    for value in metrics.values()
]

plt.figure(figsize=(8, 5))

bars = plt.bar(
    metric_names,
    metric_values
)

plt.title("Random Forest Model Performance")
plt.xlabel("Metrics")
plt.ylabel("Score (%)")

plt.ylim(0, 100)

for bar, value in zip(bars, metric_values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig("05_03_model_performance.png", dpi=300)

plt.close()

print("Created: 05_03_model_performance.png")


print("\n===================================")
print("ALL VISUALIZATIONS CREATED")
print("===================================")

print("1. 05_01_confusion_matrix.png")
print("2. 05_02_actual_vs_predicted.png")
print("3. 05_03_model_performance.png")