import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# WEEK 3 - DAY 3A
# LTV REGRESSION MODEL PERFORMANCE EVALUATION
# ============================================================

INPUT_FILE = "Week_3_Day_1_LTV_Predictions.csv"
SUMMARY_FILE = "Week_3_Day_3A_Model_Performance_Summary.csv"
REPORT_FILE = "Week_3_Day_3A_Model_Evaluation_Report.txt"


def load_and_validate_data(file_path):
    """Load prediction results and validate required columns."""

    print("\nLoading LTV prediction results...")

    df = pd.read_csv(file_path)

    required_columns = ["Actual_LTV", "Predicted_LTV"]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print(f"Dataset loaded successfully: {df.shape[0]} rows, "
          f"{df.shape[1]} columns")

    return df


def evaluate_model(actual, predicted):
    """Calculate regression model performance metrics."""

    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    r2 = r2_score(actual, predicted)

    return mae, rmse, r2


def main():

    print("=" * 65)
    print("WEEK 3 - DAY 3A: LTV REGRESSION MODEL EVALUATION")
    print("=" * 65)

    # Load and validate data
    df = load_and_validate_data(INPUT_FILE)

    actual_ltv = df["Actual_LTV"]
    predicted_ltv = df["Predicted_LTV"]

    # Calculate model metrics
    mae, rmse, r2 = evaluate_model(
        actual_ltv,
        predicted_ltv
    )

    # Calculate additional business metrics
    mean_actual_ltv = actual_ltv.mean()
    mean_predicted_ltv = predicted_ltv.mean()

    prediction_error = actual_ltv - predicted_ltv

    mean_prediction_error = prediction_error.mean()
    max_absolute_error = prediction_error.abs().max()

    # Display results
    print("\nMODEL PERFORMANCE METRICS")
    print("-" * 65)

    print(f"MAE Score                 : {mae:.2f}")
    print(f"RMSE Score                : {rmse:.2f}")
    print(f"R² Score                  : {r2:.4f}")

    print("\nLTV SUMMARY")
    print("-" * 65)

    print(f"Mean Actual LTV           : {mean_actual_ltv:.2f}")
    print(f"Mean Predicted LTV        : {mean_predicted_ltv:.2f}")
    print(f"Mean Prediction Error     : {mean_prediction_error:.2f}")
    print(f"Maximum Absolute Error    : {max_absolute_error:.2f}")

    # Create professional summary
    summary = pd.DataFrame({
        "Metric": [
            "Mean Absolute Error",
            "Root Mean Squared Error",
            "R2 Score",
            "Mean Actual LTV",
            "Mean Predicted LTV",
            "Mean Prediction Error",
            "Maximum Absolute Prediction Error"
        ],
        "Value": [
            mae,
            rmse,
            r2,
            mean_actual_ltv,
            mean_predicted_ltv,
            mean_prediction_error,
            max_absolute_error
        ]
    })

    # Save CSV summary
    summary.to_csv(SUMMARY_FILE, index=False)

    # Create evaluation report
    report = f"""
============================================================
WEEK 3 - DAY 3A
LTV REGRESSION MODEL PERFORMANCE EVALUATION REPORT
============================================================

DATASET INFORMATION
-------------------
Total Prediction Records : {len(df)}

MODEL PERFORMANCE
-----------------
Mean Absolute Error (MAE)        : {mae:.2f}
Root Mean Squared Error (RMSE)   : {rmse:.2f}
R² Score                         : {r2:.4f}

LTV SUMMARY
-----------
Mean Actual LTV                  : {mean_actual_ltv:.2f}
Mean Predicted LTV               : {mean_predicted_ltv:.2f}
Mean Prediction Error            : {mean_prediction_error:.2f}
Maximum Absolute Prediction Error: {max_absolute_error:.2f}

CONCLUSION
----------
The LTV regression model was evaluated using MAE, RMSE, and
R² score. These metrics provide an overall assessment of
prediction accuracy and model performance.

The detailed prediction error analysis will be continued in
Week 3 - Day 3B.

============================================================
"""

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write(report.strip())

    print("\nOUTPUT FILES CREATED SUCCESSFULLY")
    print("-" * 65)
    print(f"1. {SUMMARY_FILE}")
    print(f"2. {REPORT_FILE}")

    print("\nDAY 3A COMPLETED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()