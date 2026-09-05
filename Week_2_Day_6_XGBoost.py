import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report
)

from xgboost import XGBClassifier


# ==========================================
# WEEK 2 - XGBOOST CUSTOMER CHURN PREDICTION
# ==========================================

# Load dataset
df = pd.read_csv("telco_customer_churn_day3_feature_engineered.csv")

print("Dataset shape:", df.shape)


# ==========================================
# FEATURES AND TARGET
# ==========================================

X = df.drop(
    columns=["Churn", "ChurnNumeric", "customerID"]
)

y = df["ChurnNumeric"]


# ==========================================
# IDENTIFY COLUMN TYPES
# ==========================================

categorical_cols = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    exclude=["object", "string"]
).columns.tolist()


print("\nCategorical columns:")
print(categorical_cols)

print("\nNumerical columns:")
print(numerical_cols)


# ==========================================
# PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numerical_cols
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_cols
        )
    ]
)


# ==========================================
# XGBOOST MODEL
# ==========================================

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)


# ==========================================
# CREATE PIPELINE
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ]
)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nTraining XGBoost model...")

model.fit(
    X_train,
    y_train
)

print("XGBoost model trained successfully.")


# ==========================================
# SAVE TRAINED MODEL
# ==========================================

joblib.dump(
    model,
    "06_xgboost_model.pkl"
)

print("\nModel saved successfully!")
print("File created: 06_xgboost_model.pkl")


# ==========================================
# PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# MODEL METRICS
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

report = classification_report(
    y_test,
    y_pred
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("XGBOOST RESULTS")
print("==========================================")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")


print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(report)


# ==========================================
# SAVE RESULTS
# ==========================================

with open(
    "06_xgboost_results.txt",
    "w"
) as file:

    file.write(
        "XGBOOST CUSTOMER CHURN PREDICTION\n"
    )

    file.write(
        "=" * 45 + "\n\n"
    )

    file.write(
        f"Dataset shape: {df.shape}\n"
    )

    file.write(
        f"Training records: {len(X_train)}\n"
    )

    file.write(
        f"Testing records: {len(X_test)}\n\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Precision: {precision:.4f}\n"
    )

    file.write(
        f"Recall: {recall:.4f}\n"
    )

    file.write(
        f"F1-score: {f1:.4f}\n\n"
    )

    file.write(
        "CLASSIFICATION REPORT\n"
    )

    file.write(
        "=" * 45 + "\n"
    )

    file.write(report)


# ==========================================
# SAVE PREDICTIONS
# ==========================================

predictions = pd.DataFrame({
    "customerID": df.loc[
        X_test.index,
        "customerID"
    ],

    "Actual_Churn": y_test,

    "Predicted_Churn": y_pred
})


predictions.to_csv(
    "06_xgboost_predictions.csv",
    index=False
)


# ==========================================
# FINAL OUTPUT
# ==========================================

print("\n==========================================")
print("FILES CREATED")
print("==========================================")

print("1. 06_xgboost_model.pkl")
print("2. 06_xgboost_results.txt")
print("3. 06_xgboost_predictions.csv")

print("\n==========================================")
print("WEEK 2 XGBOOST COMPLETED SUCCESSFULLY")
print("==========================================")