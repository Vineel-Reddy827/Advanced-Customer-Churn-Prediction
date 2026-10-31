
<div align="center">

# 🚀 Customer Churn Prediction Using Machine Learning

### An End-to-End Machine Learning Project for Predicting Customer Retention Risk

<p>
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/Frontend-React-61DAFB?logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Explainable%20AI-SHAP-purple" />
  <img src="https://img.shields.io/badge/Status-Completed-success" />
</p>

**Turning customer data into actionable churn predictions.**

</div>

---

## 📌 1. Project Overview

Customer churn is one of the major challenges faced by subscription-based businesses. It occurs when customers discontinue their services or stop doing business with a company.

Acquiring new customers is often more expensive than retaining existing ones. Therefore, identifying customers who are likely to leave can help businesses develop effective customer retention strategies.

This project presents an end-to-end **Customer Churn Prediction System** that uses machine learning classification algorithms to estimate the likelihood of a customer discontinuing a service.

The system combines data analysis, machine learning, model optimization, explainable AI, and web application development into a single workflow.

## 🎯 2. Project Objectives

- 📊 Perform exploratory data analysis to discover customer churn patterns.
- 🧹 Clean and preprocess the customer dataset.
- ⚙️ Engineer and transform features for machine learning.
- 🤖 Train and compare multiple classification algorithms.
- 🔍 Optimize model hyperparameters using GridSearchCV.
- 📈 Improve classification performance through threshold optimization.
- 🧠 Interpret model behavior using SHAP explainability.
- 🌐 Develop a web application for interactive churn predictions.
- 🔗 Integrate the machine learning model with a FastAPI backend and React frontend.

## 🛠️ 3. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python 3.11, JavaScript |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Explainable AI | SHAP |
| Model Serialization | Joblib |
| Backend Development | FastAPI |
| Frontend Development | React, Vite |
| API Communication | REST API, Fetch API |
| Development Environment | Visual Studio Code |

## 📂 4. Dataset Description

The project uses the **Telco Customer Churn dataset**, which contains customer information related to demographics, subscriptions, services, contracts, and billing.

Dataset Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn?resource=download

| Dataset Property | Description |
|---|---|
| Total Records | 7,043 |
| Original Features | 21 |
| Target Variable | Churn |
| Prediction Type | Binary Classification |
| Classes | Yes / No |

### 🔎 Important Feature Categories

- **Customer Demographics:** Gender, Senior Citizen, Partner, Dependents.
- **Account Information:** Tenure, Contract, Paperless Billing.
- **Service Information:** Internet Service, Phone Service, Online Security, Tech Support.
- **Billing Information:** Monthly Charges, Total Charges, Payment Method.

**Target Variable: `Churn`**

- `1` – Customer churned.
- `0` – Customer did not churn.

## 🔄 5. Machine Learning Workflow

The project follows a structured machine learning pipeline.

| Phase | Description |
|---|---|
| 01. Data Collection | Load and inspect the customer dataset. |
| 02. Data Preprocessing | Handle data types, clean records, and prepare features. |
| 03. Exploratory Data Analysis | Analyze churn patterns and visualize relationships. |
| 04. Feature Engineering | Encode categorical features and scale numerical features. |
| 05. Data Splitting | Divide the dataset into training and testing sets. |
| 06. Model Development | Train and compare classification algorithms. |
| 07. Hyperparameter Tuning | Optimize model parameters using GridSearchCV. |
| 08. Threshold Optimization | Select a classification threshold using out-of-fold predictions. |
| 09. Model Evaluation | Evaluate performance using classification metrics. |
| 10. Explainable AI | Interpret feature contributions using SHAP. |
| 11. Application Development | Integrate the trained model with FastAPI and React. |

## 🤖 6. Machine Learning Algorithms

The following algorithms were implemented and evaluated:

| Algorithm | Purpose |
|---|---|
| Logistic Regression | Baseline binary classification |
| Balanced Logistic Regression | Handles class imbalance using class weights |
| Decision Tree | Learns rule-based classification patterns |
| Random Forest | Combines multiple decision trees |
| Gradient Boosting | Builds an ensemble of sequentially improved trees |

### 🏆 Model Selection

The tuned Logistic Regression model was selected based on its F1-score performance.

Hyperparameter tuning was performed using **GridSearchCV with 5-fold cross-validation**.

The selected parameters were:

- `C = 0.1`
- `class_weight = balanced`
- `solver = liblinear`

## 📊 7. Model Performance

The tuned Logistic Regression model achieved the following results on the held-out test dataset using the default classification threshold of 0.5.

| Evaluation Metric | Score |
|---|---:|
| Accuracy | 74.09% |
| Precision | 50.78% |
| Recall | 78.61% |
| F1-Score | 61.70% |
| ROC-AUC | 84.10% |

### 📌 Understanding the Results

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Proportion of predicted churn customers who actually churned.
- **Recall:** Proportion of actual churn customers correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the model's ability to distinguish between the two classes across classification thresholds.

### 🎚️ Classification Threshold Optimization

The default classification threshold of 0.5 was further optimized using out-of-fold predictions generated through Stratified 5-Fold Cross-Validation.

| Metric | Default Threshold | Optimized Threshold |
|---|---:|---:|
| Threshold | 0.5000 | 0.5479 |
| Accuracy | 74.09% | 75.44% |
| Precision | 50.78% | 52.60% |
| Recall | 78.61% | 75.67% |
| F1-Score | 61.70% | 62.06% |

The optimized threshold produced a small improvement in F1-score on the held-out test set.

## 🧠 8. Explainable AI Using SHAP

A machine learning model should not only produce predictions but should also provide insights into the factors influencing its decisions.

To improve model interpretability, this project uses **SHAP (SHapley Additive exPlanations)**.

The following visualizations were generated:

- 📍 SHAP Summary Plot – Shows how features influence model predictions.
- 📊 SHAP Feature Importance – Ranks features according to their contribution.
- 👤 Individual Customer Explanation – Explains feature contributions for an individual customer.

Generated visualizations are available in the `reports/` directory.

## 🌐 9. Web Application

The trained machine learning model is integrated into a web application using FastAPI and React.

### Frontend — React + Vite

- Interactive customer information form.
- Structured input sections.
- Customer subscription and billing details.
- Prediction result display.
- Estimated churn probability.

### Backend — FastAPI

- Loads the trained machine learning pipeline.
- Receives customer information through a REST API.
- Processes the input features.
- Generates churn predictions.
- Returns the predicted class and churn probability.

### 🔗 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome endpoint |
| GET | `/health` | Checks API and model status |
| POST | `/predict` | Generates customer churn prediction |

API documentation is available through FastAPI Swagger UI:

`http://127.0.0.1:8000/docs`

## 🖥️ 10. Project Structure

```text
Customer_Churn_Prediction/
│
├── app/
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── frontend/
│   ├── src/
│   ├── public/
│   ├── package.json
│   └── index.html
│
├── models/
│   ├── best_churn_model.pkl
│   ├── tuned_churn_model.pkl
│   └── optimal_threshold.txt
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   ├── hyperparameter_tuning_results.csv
│   └── threshold_comparison.csv
│
├── src/
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── model_tuning.py
│   ├── threshold_optimization.py
│   └── model_explainability.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## ⚙️ 11. Installation and Execution

Follow these instructions to run the project locally.

### Prerequisites

- Python 3.11
- Node.js and npm
- Visual Studio Code
- Git

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Customer_Churn_Prediction
```

### Step 2: Create a Python Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Start the FastAPI Backend

From the project root directory:

```bash
uvicorn app.main:app --reload
```

Backend address:

`http://127.0.0.1:8000`

### Step 5: Install Frontend Dependencies

Open a new terminal:

```bash
cd frontend
npm install
```

### Step 6: Start the React Frontend

```bash
npm run dev
```

Open the local URL displayed in the terminal, usually:

`http://localhost:5173`

## 📈 12. Evaluation and Visualization

The project includes several evaluation and analysis outputs:

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Classification Report
- Model Comparison Results
- Hyperparameter Tuning Results
- Threshold Comparison
- SHAP Explainability Visualizations
- Exploratory Data Analysis Charts

These outputs are stored in the `reports/` directory.

## ⚠️ 13. Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- The model's performance may vary when applied to different customer populations.
- Churn probability is a model estimate, not a guarantee of future customer behavior.
- Additional validation and monitoring would be necessary before production deployment.

## 🚀 14. Future Enhancements

Potential improvements include:

- 📁 Batch predictions using CSV uploads.
- 📊 Interactive analytics and customer segmentation dashboards.
- ☁️ Cloud deployment of the prediction application.
- 🔄 Model monitoring and periodic retraining.
- 🗄️ Database integration for customer records.
- 🎯 Customer retention recommendations based on churn risk.

## 👨‍💻 15. Project Information

**Project:** Customer Churn Prediction Using Machine Learning

**Domain:** Data Science and Machine Learning

**Application Type:** Full-Stack Machine Learning Application

**Developed As:** Virtual Internship Project

---

<div align="center">

### ⭐ If you find this project interesting, consider giving the repository a star!

**Built with Python, Machine Learning, and React.**

</div>
