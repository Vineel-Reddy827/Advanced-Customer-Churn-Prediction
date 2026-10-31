
import pandas as pd

# 1. Load original dataset
df = pd.read_csv("data/raw/customer_churn.csv")

print("Starting Data Cleaning...")

# 2. Remove unnecessary whitespace
df["TotalCharges"] = df["TotalCharges"].astype(str).str.strip()

# 3. Convert TotalCharges into numerical format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# 4. Count missing values before handling
missing_before = df["TotalCharges"].isnull().sum()

print("Missing values before handling:", missing_before)

# 5. Handle missing TotalCharges
# All 11 affected records have tenure = 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# 6. Verify the cleaning operation
print("Missing values after handling:",
      df["TotalCharges"].isnull().sum())

print("TotalCharges data type:",
      df["TotalCharges"].dtype)

# 7. Save cleaned dataset
df.to_csv(
    "data/processed/customer_churn_clean.csv",
    index=False
)

print("\nCLEANING COMPLETED SUCCESSFULLY")

print("Final dataset shape:", df.shape)
print("Cleaned dataset saved successfully!")
