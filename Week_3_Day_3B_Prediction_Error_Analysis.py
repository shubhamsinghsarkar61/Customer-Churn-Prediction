import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# WEEK 3 - DAY 3B
# LTV PREDICTION ERROR ANALYSIS AND MODEL FINALIZATION
# ============================================================

INPUT_FILE = "Week_3_Day_1_LTV_Predictions.csv"

ERROR_SUMMARY_FILE = "Week_3_Day_3B_Prediction_Error_Summary.csv"
TOP_ERRORS_FILE = "Week_3_Day_3B_Top_Prediction_Errors.csv"
ERROR_DISTRIBUTION_FILE = "Week_3_Day_3B_Error_Distribution.csv"

ERROR_DISTRIBUTION_CHART = (
    "Week_3_Day_3B_Error_Distribution.png"
)

TOP_ERRORS_CHART = (
    "Week_3_Day_3B_Top_10_Errors.png"
)

REPORT_FILE = (
    "Week_3_Day_3B_Model_Finalization_Report.txt"
)


def load_and_validate_data(file_path):
    """Load prediction results and validate required columns."""

    print("\nLoading LTV prediction results...")

    df = pd.read_csv(file_path)

    required_columns = [
        "Actual_LTV",
        "Predicted_LTV"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Missing values found in Actual_LTV or Predicted_LTV."
        )

    print(
        f"Dataset loaded successfully: "
        f"{df.shape[0]} rows, {df.shape[1]} columns"
    )

    return df


def calculate_prediction_errors(df):
    """Calculate signed and absolute prediction errors."""

    df = df.copy()

    df["Prediction_Error"] = (
        df["Actual_LTV"]
        - df["Predicted_LTV"]
    )

    df["Absolute_Error"] = (
        df["Prediction_Error"].abs()
    )

    return df


def main():

    print("=" * 70)
    print("WEEK 3 - DAY 3B")
    print(
        "LTV PREDICTION ERROR ANALYSIS "
        "AND MODEL FINALIZATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Load and validate data
    # --------------------------------------------------------

    df = load_and_validate_data(INPUT_FILE)

    # --------------------------------------------------------
    # 2. Calculate prediction errors
    # --------------------------------------------------------

    df = calculate_prediction_errors(df)

    # --------------------------------------------------------
    # 3. Calculate error metrics
    # --------------------------------------------------------

    total_predictions = len(df)

    mean_error = df["Prediction_Error"].mean()

    mean_absolute_error = (
        df["Absolute_Error"].mean()
    )

    median_absolute_error = (
        df["Absolute_Error"].median()
    )

    maximum_absolute_error = (
        df["Absolute_Error"].max()
    )

    minimum_absolute_error = (
        df["Absolute_Error"].min()
    )

    # --------------------------------------------------------
    # 4. Create prediction error categories
    # --------------------------------------------------------

    error_bins = [
        0,
        25,
        50,
        100,
        float("inf")
    ]

    error_labels = [
        "0-25",
        "26-50",
        "51-100",
        "Above 100"
    ]

    df["Error_Category"] = pd.cut(
        df["Absolute_Error"],
        bins=error_bins,
        labels=error_labels,
        include_lowest=True
    )

    error_distribution = (
        df["Error_Category"]
        .value_counts()
        .sort_index()
        .reset_index()
    )

    error_distribution.columns = [
        "Error_Range",
        "Customer_Count"
    ]

    error_distribution["Percentage"] = (
        error_distribution["Customer_Count"]
        / total_predictions
        * 100
    )

    # --------------------------------------------------------
    # 5. Identify Top 10 largest prediction errors
    # --------------------------------------------------------

    top_errors = (
        df.sort_values(
            by="Absolute_Error",
            ascending=False
        )
        .head(10)
        .copy()
    )

    top_errors.insert(
        0,
        "Customer_Index",
        top_errors.index
    )

    # --------------------------------------------------------
    # 6. Create prediction error summary
    # --------------------------------------------------------

    error_summary = pd.DataFrame({
        "Metric": [
            "Total Predictions",
            "Mean Prediction Error",
            "Mean Absolute Error",
            "Median Absolute Error",
            "Maximum Absolute Error",
            "Minimum Absolute Error"
        ],
        "Value": [
            total_predictions,
            mean_error,
            mean_absolute_error,
            median_absolute_error,
            maximum_absolute_error,
            minimum_absolute_error
        ]
    })

    # --------------------------------------------------------
    # 7. Save analysis results
    # --------------------------------------------------------

    error_summary.to_csv(
        ERROR_SUMMARY_FILE,
        index=False
    )

    top_errors.to_csv(
        TOP_ERRORS_FILE,
        index=False
    )

    error_distribution.to_csv(
        ERROR_DISTRIBUTION_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 8. Visualization 1:
    #    Error Distribution
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    bars = plt.bar(
        error_distribution["Error_Range"],
        error_distribution["Customer_Count"]
    )

    plt.title("Distribution of LTV Prediction Errors")

    plt.xlabel("Absolute Prediction Error Range")

    plt.ylabel("Number of Predictions")

    # Add customer count labels
    for bar, count in zip(
        bars,
        error_distribution["Customer_Count"]
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(count),
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    plt.savefig(
        ERROR_DISTRIBUTION_CHART,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # 9. Visualization 2:
    #    Top 10 Largest Prediction Errors
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        top_errors["Customer_Index"].astype(str),
        top_errors["Absolute_Error"]
    )

    plt.title("Top 10 Largest LTV Prediction Errors")

    plt.xlabel("Customer Index")

    plt.ylabel("Absolute Prediction Error")

    # Add error value labels
    for bar, error in zip(
        bars,
        top_errors["Absolute_Error"]
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{error:.1f}",
            ha="center",
            va="bottom",
            rotation=90
        )

    plt.tight_layout()

    plt.savefig(
        TOP_ERRORS_CHART,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # 10. Generate final model report
    # --------------------------------------------------------

    report = f"""
================================================================
WEEK 3 - DAY 3B
LTV PREDICTION ERROR ANALYSIS AND MODEL FINALIZATION REPORT
================================================================

DATASET OVERVIEW
----------------------------------------------------------------
Total Prediction Records: {total_predictions}

PREDICTION ERROR METRICS
----------------------------------------------------------------
Mean Prediction Error        : {mean_error:.2f}
Mean Absolute Error          : {mean_absolute_error:.2f}
Median Absolute Error        : {median_absolute_error:.2f}
Maximum Absolute Error       : {maximum_absolute_error:.2f}
Minimum Absolute Error       : {minimum_absolute_error:.2f}

ERROR DISTRIBUTION
----------------------------------------------------------------
"""

    for _, row in error_distribution.iterrows():

        report += (
            f"{row['Error_Range']}: "
            f"{row['Customer_Count']} predictions "
            f"({row['Percentage']:.2f}%)\n"
        )

    report += f"""

TOP ERROR ANALYSIS
----------------------------------------------------------------
The top 10 predictions with the largest absolute errors were
identified and saved for further analysis.

The maximum absolute prediction error was
{maximum_absolute_error:.2f}.

MODEL FINALIZATION
----------------------------------------------------------------
The LTV regression model was evaluated using performance metrics
during Day 3A and detailed prediction error analysis during
Day 3B.

The error distribution shows how prediction accuracy varies
across the {total_predictions} customer predictions.

Two visualizations were generated:

1. Distribution of LTV Prediction Errors
2. Top 10 Largest LTV Prediction Errors

The LTV regression analysis is now complete and can serve as
the analytical foundation for the next project stage:
FastAPI-based LTV prediction services.

================================================================
"""

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report.strip())

    # --------------------------------------------------------
    # 11. Display results
    # --------------------------------------------------------

    print("\nPREDICTION ERROR SUMMARY")
    print("-" * 70)

    print(
        f"Total Predictions          : "
        f"{total_predictions}"
    )

    print(
        f"Mean Prediction Error      : "
        f"{mean_error:.2f}"
    )

    print(
        f"Mean Absolute Error        : "
        f"{mean_absolute_error:.2f}"
    )

    print(
        f"Median Absolute Error      : "
        f"{median_absolute_error:.2f}"
    )

    print(
        f"Maximum Absolute Error     : "
        f"{maximum_absolute_error:.2f}"
    )

    print(
        f"Minimum Absolute Error     : "
        f"{minimum_absolute_error:.2f}"
    )

    print("\nERROR DISTRIBUTION")
    print("-" * 70)

    print(
        error_distribution.to_string(
            index=False
        )
    )

    print("\nTOP 10 LARGEST PREDICTION ERRORS")
    print("-" * 70)

    print(
        top_errors[
            [
                "Customer_Index",
                "Actual_LTV",
                "Predicted_LTV",
                "Prediction_Error",
                "Absolute_Error"
            ]
        ].to_string(index=False)
    )

    print("\nOUTPUT FILES CREATED SUCCESSFULLY")
    print("-" * 70)

    print(f"1. {ERROR_SUMMARY_FILE}")
    print(f"2. {TOP_ERRORS_FILE}")
    print(f"3. {ERROR_DISTRIBUTION_FILE}")
    print(f"4. {ERROR_DISTRIBUTION_CHART}")
    print(f"5. {TOP_ERRORS_CHART}")
    print(f"6. {REPORT_FILE}")

    print("\nDAY 3B COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()