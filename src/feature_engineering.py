import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("data/processed/customer_churn_clean.csv")

print("Dataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# Separate features and target
X = df.drop(columns=['Churn', 'customerID'], errors='ignore')

y = df['Churn'].map({
    'No': 0,
    'Yes': 1
})

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nTarget Distribution:")
print(y.value_counts())

# Identify numerical columns
numerical_features = X.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

# Identify categorical columns
categorical_features = X.select_dtypes(
    include=['object', 'category']
).columns.tolist()

print("Numerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'),
         categorical_features)
    ]
)

print("Preprocessing pipeline created successfully.")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting Target Distribution:")
print(y_test.value_counts(normalize=True))

# Fit transformations using training data only
X_train_processed = preprocessor.fit_transform(X_train)

# Transform testing data using the same fitted transformations
X_test_processed = preprocessor.transform(X_test)

print("Training Data After Processing:",
      X_train_processed.shape)

print("Testing Data After Processing:",
      X_test_processed.shape)