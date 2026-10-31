
import pandas as pd

# Load original dataset
df = pd.read_csv("data/raw/customer_churn.csv")

# Detect empty or whitespace-only TotalCharges
empty_values = (
    df["TotalCharges"]
    .astype(str)
    .str.strip()
    .eq("")
)

print("TotalCharges Investigation")
print("-" * 40)

print("Whitespace-only values:", empty_values.sum())

# Display affected records
print("\nAffected customer records:")

print(
    df.loc[
        empty_values,
        ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]
    ]
)

# Check numerical conversion without modifying original data
converted = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
)

print("\nInvalid numerical values:", converted.isna().sum())
