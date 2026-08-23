# ============================================================
# WEEK 3 - DAY 1
# LTV REGRESSION MODEL - PERFORMANCE VISUALIZATION
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD LTV PREDICTION RESULTS
# ============================================================

results = pd.read_csv(
    "Week_3_Day_1_LTV_Predictions.csv"
)

print("=" * 60)
print("LTV PREDICTION RESULTS LOADED SUCCESSFULLY")
print("=" * 60)

print("\nDataset Shape:", results.shape)


# ============================================================
# 2. CALCULATE MODEL PERFORMANCE METRICS
# ============================================================

actual = results["Actual_LTV"]
predicted = results["Predicted_LTV"]
residuals = results["Prediction_Error"]

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("\nMODEL PERFORMANCE METRICS")
print("-" * 60)
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")


# ============================================================
# 3. ACTUAL VS PREDICTED LTV
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    actual,
    predicted,
    alpha=0.6
)

min_value = min(actual.min(), predicted.min())
max_value = max(actual.max(), predicted.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

plt.title(
    "Actual vs Predicted Customer Lifetime Value",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Actual LTV", fontsize=12)
plt.ylabel("Predicted LTV", fontsize=12)

plt.legend()
plt.grid(alpha=0.3)

plt.text(
    0.05,
    0.95,
    f"R² Score: {r2:.4f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}",
    transform=plt.gca().transAxes,
    verticalalignment="top",
    bbox=dict(boxstyle="round", alpha=0.8)
)

plt.tight_layout()

plt.savefig(
    "Week_3_Day_1_01_Actual_vs_Predicted_LTV.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 4. RESIDUAL ANALYSIS
# ============================================================

plt.figure(figsize=(10, 7))

plt.scatter(
    predicted,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--",
    linewidth=2
)

plt.title(
    "Residual Analysis of LTV Predictions",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Predicted LTV", fontsize=12)
plt.ylabel("Prediction Error (Actual - Predicted)", fontsize=12)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "Week_3_Day_1_02_Residual_Analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 5. MODEL PERFORMANCE SUMMARY
# ============================================================

metrics = ["MAE", "RMSE", "R² Score"]
values = [mae, rmse, r2]

plt.figure(figsize=(10, 7))

bars = plt.bar(metrics, values)

plt.title(
    "Baseline LTV Regression Model Performance",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel("Metric Value", fontsize=12)

plt.grid(
    axis="y",
    alpha=0.3
)

# Add values above bars
for bar, value in zip(bars, values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    "Week_3_Day_1_03_Model_Performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 6. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("WEEK 3 - DAY 1 LTV VISUALIZATION COMPLETED")
print("=" * 60)

print("\nGenerated Visualization Files:")
print("1. Week_3_Day_1_01_Actual_vs_Predicted_LTV.png")
print("2. Week_3_Day_1_02_Residual_Analysis.png")
print("3. Week_3_Day_1_03_Model_Performance.png")