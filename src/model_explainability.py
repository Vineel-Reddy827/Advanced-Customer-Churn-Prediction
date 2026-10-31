
import pandas as pd
import numpy as np
import joblib
import shap
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 1. Load the trained model
model_paths = [
    "models/tuned_churn_model.pkl",
    "models/best_churn_model.pkl"
]

model_path = next(
    (path for path in model_paths if os.path.exists(path)),
    None
)

if model_path is None:
    raise FileNotFoundError(
        "No trained model found in the models folder."
    )

pipeline = joblib.load(model_path)
print("Model loaded:", model_path)

# 2. Load test data
X_test = pd.read_csv("data/processed/X_test.csv")

# 3. Separate preprocessing and classifier
preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

# 4. Transform test data
X_test_transformed = preprocessor.transform(X_test)

# Convert sparse matrix to dense array
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

# 5. Get feature names
feature_names = preprocessor.get_feature_names_out()

X_test_transformed = pd.DataFrame(
    X_test_transformed,
    columns=feature_names
)

# 6. Create background dataset
background = X_test_transformed.sample(
    n=min(100, len(X_test_transformed)),
    random_state=42
)

# 7. Initialize SHAP explainer
explainer = shap.LinearExplainer(
    classifier,
    background
)

# 8. Calculate SHAP values
shap_values = explainer(X_test_transformed)

# 9. Create reports directory
os.makedirs("reports", exist_ok=True)

# 10. SHAP Summary Plot
plt.figure()
shap.summary_plot(
    shap_values.values,
    X_test_transformed,
    feature_names=feature_names,
    show=False
)

plt.title("SHAP Summary - Customer Churn")
plt.tight_layout()
plt.savefig(
    "reports/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 11. SHAP Feature Importance Plot
plt.figure()
shap.summary_plot(
    shap_values.values,
    X_test_transformed,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig(
    "reports/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# 12. Individual Customer Explanation
plt.figure()
shap.plots.waterfall(
    shap_values[0],
    show=False
)

plt.savefig(
    "reports/shap_customer_explanation.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nSHAP EXPLAINABILITY COMPLETED!")
print("Generated reports:")
print("1. SHAP Summary Plot")
print("2. SHAP Feature Importance")
print("3. Individual Customer Explanation")
