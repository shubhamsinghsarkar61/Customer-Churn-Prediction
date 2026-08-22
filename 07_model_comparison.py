import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("DAY 7 - CUSTOMER CHURN MODEL COMPARISON")
print("=" * 60)

# Model performance data
data = {
    "Model": [
        "Logistic Regression",
        "Random Forest",
        "XGBoost"
    ],
    "Accuracy": [
        0.80,
        0.7828,
        0.7913
    ],
    "Precision": [
        0.6622,
        0.6149,
        0.6379
    ],
    "Recall": [
        0.5241,
        0.4866,
        0.4947
    ],
    "F1-Score": [
        0.5851,
        0.5433,
        0.5572
    ]
}

# Create DataFrame
results = pd.DataFrame(data)

# Save comparison results
results.to_csv("07_model_comparison_results.csv", index=False)

print("\nMODEL PERFORMANCE COMPARISON")
print("=" * 60)
print(results.to_string(index=False))

# Identify best model based on F1-score
best_model = results.loc[results["F1-Score"].idxmax()]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)
print(f"Model: {best_model['Model']}")
print(f"Accuracy: {best_model['Accuracy']:.2%}")
print(f"Precision: {best_model['Precision']:.2%}")
print(f"Recall: {best_model['Recall']:.2%}")
print(f"F1-Score: {best_model['F1-Score']:.2%}")

# -------------------------------------------------
# Visualization 1: Accuracy Comparison
# -------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    results["Model"],
    results["Accuracy"]
)

plt.title("Customer Churn Model Accuracy Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

for i, value in enumerate(results["Accuracy"]):
    plt.text(i, value + 0.01, f"{value:.2%}", ha="center")

plt.tight_layout()
plt.savefig("07_01_accuracy_comparison.png", dpi=300)
plt.close()

print("\nCreated: 07_01_accuracy_comparison.png")

# -------------------------------------------------
# Visualization 2: Model Performance Comparison
# -------------------------------------------------

metrics = ["Precision", "Recall", "F1-Score"]

x = range(len(results["Model"]))
width = 0.25

plt.figure(figsize=(11, 6))

plt.bar(
    [i - width for i in x],
    results["Precision"],
    width,
    label="Precision"
)

plt.bar(
    x,
    results["Recall"],
    width,
    label="Recall"
)

plt.bar(
    [i + width for i in x],
    results["F1-Score"],
    width,
    label="F1-Score"
)

plt.xticks(x, results["Model"])
plt.title("Precision, Recall and F1-Score Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.savefig("07_02_performance_comparison.png", dpi=300)
plt.close()

print("Created: 07_02_performance_comparison.png")

# -------------------------------------------------
# Visualization 3: Overall Model Comparison
# -------------------------------------------------

comparison_metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

x = range(len(comparison_metrics))
width = 0.25

plt.figure(figsize=(12, 7))

plt.bar(
    [i - width for i in x],
    results.loc[0, comparison_metrics],
    width,
    label="Logistic Regression"
)

plt.bar(
    x,
    results.loc[1, comparison_metrics],
    width,
    label="Random Forest"
)

plt.bar(
    [i + width for i in x],
    results.loc[2, comparison_metrics],
    width,
    label="XGBoost"
)

plt.xticks(x, comparison_metrics)
plt.title("Overall Customer Churn Model Comparison")
plt.xlabel("Performance Metric")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.savefig("07_03_overall_model_comparison.png", dpi=300)
plt.close()

print("Created: 07_03_overall_model_comparison.png")

print("\n" + "=" * 60)
print("DAY 7 MODEL COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files:")
print("1. 07_model_comparison_results.csv")
print("2. 07_01_accuracy_comparison.png")
print("3. 07_02_performance_comparison.png")
print("4. 07_03_overall_model_comparison.png")