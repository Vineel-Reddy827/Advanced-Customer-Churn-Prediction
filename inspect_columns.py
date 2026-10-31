
import pandas as pd

df = pd.read_csv("data/raw/customer_churn.csv")

print("\n========== COLUMN ANALYSIS ==========\n")

for column in df.columns:
    print(f"\nColumn: {column}")
    print(f"Data Type: {df[column].dtype}")
    print(f"Unique Values: {df[column].nunique()}")
    print("Sample Values:")
    print(df[column].unique()[:5])
    print("-" * 50)
