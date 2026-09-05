import pandas as pd
import matplotlib.pyplot as plt

print("=" * 60)
print("WEEK 3 - DAY 2: LTV CUSTOMER SEGMENTATION")
print("=" * 60)

# --------------------------------------------------
# 1. LOAD LTV PREDICTION DATA
# --------------------------------------------------

df = pd.read_csv("Week_3_Day_1_LTV_Predictions.csv")

print("\nLTV Prediction Dataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# --------------------------------------------------
# 2. CHECK COLUMN NAMES
# --------------------------------------------------

print("\nAvailable Columns:")
print(df.columns.tolist())


# --------------------------------------------------
# 3. CREATE LTV SEGMENTS
# --------------------------------------------------

# Automatically divide customers into Low, Medium and High LTV groups

df["LTV_Segment"] = pd.qcut(
    df["Predicted_LTV"],
    q=3,
    labels=["Low LTV", "Medium LTV", "High LTV"]
)

print("\nLTV Segmentation Completed Successfully!")


# --------------------------------------------------
# 4. SEGMENT DISTRIBUTION
# --------------------------------------------------

segment_counts = df["LTV_Segment"].value_counts()

print("\nCUSTOMER COUNT BY LTV SEGMENT")
print("-" * 40)
print(segment_counts)


# --------------------------------------------------
# 5. AVERAGE LTV BY SEGMENT
# --------------------------------------------------

segment_avg = df.groupby(
    "LTV_Segment",
    observed=True
)["Predicted_LTV"].mean()

print("\nAVERAGE PREDICTED LTV BY SEGMENT")
print("-" * 40)
print(segment_avg)


# --------------------------------------------------
# 6. SAVE SEGMENTED CUSTOMER DATA
# --------------------------------------------------

output_file = "Week_3_Day_2_LTV_Segmented_Customers.csv"

df.to_csv(output_file, index=False)

print(f"\nSegmented customer data saved as:")
print(output_file)


# --------------------------------------------------
# 7. VISUALIZATION 1 - CUSTOMER DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

segment_counts.plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Customer Distribution by LTV Segment")
plt.xlabel("LTV Segment")
plt.ylabel("Number of Customers")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "Week_3_Day_2_01_LTV_Segment_Distribution.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 8. VISUALIZATION 2 - AVERAGE LTV BY SEGMENT
# --------------------------------------------------

plt.figure(figsize=(8, 5))

segment_avg.plot(
    kind="bar",
    edgecolor="black"
)

plt.title("Average Predicted LTV by Customer Segment")
plt.xlabel("LTV Segment")
plt.ylabel("Average Predicted LTV")

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    "Week_3_Day_2_02_Average_LTV_by_Segment.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 9. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 60)
print("WEEK 3 - DAY 2 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated Files:")
print("1. Week_3_Day_2_LTV_Segmented_Customers.csv")
print("2. Week_3_Day_2_01_LTV_Segment_Distribution.png")
print("3. Week_3_Day_2_02_Average_LTV_by_Segment.png")