
import pandas as pd

# Load the customer churn dataset
df = pd.read_csv("data/raw/customer_churn.csv")

# Display basic dataset information
print("DATASET LOADED SUCCESSFULLY")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()


# ==========================================
# DATA QUALITY ASSESSMENT
# ==========================================

print("\nMISSING VALUES:")
print(df.isnull().sum())

print("\nEMPTY STRING VALUES:")
print((df == "").sum())

print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())

print("\nDATA TYPES:")
print(df.dtypes)

print("\nTOTAL MISSING VALUES:")
print(df.isnull().sum().sum())

