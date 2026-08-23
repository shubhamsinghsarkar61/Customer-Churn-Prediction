# ============================================================
# WEEK 3 - DAY 1
# LTV CALCULATION - BASELINE REGRESSION MODEL
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("telco_customer_churn_day3_feature_engineered.csv")

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

print("Dataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. CLEAN TOTAL CHARGES
# ============================================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

df["TotalCharges"] = df["TotalCharges"].fillna(
    df["TotalCharges"].median()
)


# ============================================================
# 3. SELECT ACTIVE CUSTOMERS
# ============================================================

active_customers = df[df["Churn"] == "No"].copy()

print("\nTotal Active Customers:", active_customers.shape[0])


# ============================================================
# 4. CREATE ESTIMATED LTV TARGET
# ============================================================

# Calculate expected remaining customer lifetime.
# The dataset covers a maximum tenure of 72 months.

active_customers["ExpectedRemainingMonths"] = (
    72 - active_customers["tenure"]
)

# Ensure minimum remaining lifetime is 1 month

active_customers["ExpectedRemainingMonths"] = (
    active_customers["ExpectedRemainingMonths"].clip(lower=1)
)


# Calculate Estimated Lifetime Value

active_customers["Estimated_LTV"] = (
    active_customers["MonthlyCharges"]
    * active_customers["ExpectedRemainingMonths"]
)

print("\nEstimated LTV Created Successfully")

print("\nLTV Summary:")
print(active_customers["Estimated_LTV"].describe())


# ============================================================
# 5. SELECT FEATURES
# ============================================================

features = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]

X = active_customers[features]

y = active_customers["Estimated_LTV"]


# ============================================================
# 6. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 7. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ============================================================
# 8. SPLIT THE DATA
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ============================================================
# 9. BUILD BASELINE LINEAR REGRESSION MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regression",
            LinearRegression()
        )
    ]
)


# ============================================================
# 10. TRAIN THE MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)

print("\nModel Training Completed Successfully")


# ============================================================
# 11. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 12. EVALUATE THE MODEL
# ============================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n" + "=" * 60)
print("BASELINE LTV REGRESSION MODEL RESULTS")
print("=" * 60)

print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")


# ============================================================
# 13. SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual_LTV": y_test.values,
    "Predicted_LTV": y_pred,
    "Prediction_Error": y_test.values - y_pred
})


results.to_csv(
    "Week_3_Day_1_LTV_Predictions.csv",
    index=False
)


print("\nPredictions saved successfully!")

print("\nFirst 10 Predictions:")
print(results.head(10))


# ============================================================
# WEEK 3 - DAY 1 COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("WEEK 3 - DAY 1 COMPLETED SUCCESSFULLY")
print("=" * 60)