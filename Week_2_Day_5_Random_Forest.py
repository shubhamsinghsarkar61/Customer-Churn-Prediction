import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. LOAD DAY 3 FEATURE-ENGINEERED DATASET
# ============================================================

df = pd.read_csv("telco_customer_churn_day3_feature_engineered.csv")

print("Dataset shape:", df.shape)


# ============================================================
# 2. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Churn", "ChurnNumeric", "customerID"])
y = df["ChurnNumeric"]


# ============================================================
# 3. IDENTIFY CATEGORICAL AND NUMERICAL FEATURES
# ============================================================

categorical_cols = X.select_dtypes(
    include=["object", "string"]
).columns

numerical_cols = X.select_dtypes(
    exclude=["object", "string"]
).columns


# ============================================================
# 4. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols
        ),
        (
            "numerical",
            "passthrough",
            numerical_cols
        )
    ]
)


# ============================================================
# 5. RANDOM FOREST MODEL
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ============================================================
# 7. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)

print("\nRandom Forest model trained successfully.")


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 9. CALCULATE EVALUATION METRICS
# ============================================================

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

report = classification_report(y_test, y_pred)


# ============================================================
# 10. DISPLAY RESULTS
# ============================================================

print("\n==========================================")
print("RANDOM FOREST RESULTS")
print("==========================================")

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(report)


# ============================================================
# 11. SAVE EVALUATION RESULTS
# ============================================================

with open("05_random_forest_results.txt", "w") as file:

    file.write("RANDOM FOREST RESULTS\n")
    file.write("=====================\n\n")

    file.write(f"Dataset size: {df.shape}\n")
    file.write(f"Training records: {len(X_train)}\n")
    file.write(f"Testing records: {len(X_test)}\n\n")

    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall:    {recall:.4f}\n")
    file.write(f"F1-score:  {f1:.4f}\n\n")

    file.write("CLASSIFICATION REPORT\n")
    file.write("=====================\n")
    file.write(report)


# ============================================================
# 12. SAVE ACTUAL VS PREDICTED CHURN
# ============================================================

predictions = pd.DataFrame({
    "customerID": df.loc[X_test.index, "customerID"],
    "Actual_Churn": y_test,
    "Predicted_Churn": y_pred
})

predictions.to_csv(
    "05_random_forest_predictions.csv",
    index=False
)


# ============================================================
# 13. FINAL MESSAGE
# ============================================================

print("\n==========================================")
print("FILES CREATED")
print("==========================================")

print("05_random_forest_results.txt")
print("05_random_forest_predictions.csv")

print("\nDay 5 Random Forest completed successfully.")