
import { useState } from "react";
import "./App.css";

const initialForm = {
  gender: "Male",
  SeniorCitizen: 0,
  Partner: "Yes",
  Dependents: "No",
  tenure: 12,
  PhoneService: "Yes",
  MultipleLines: "No",
  InternetService: "Fiber optic",
  OnlineSecurity: "No",
  OnlineBackup: "No",
  DeviceProtection: "No",
  TechSupport: "No",
  StreamingTV: "No",
  StreamingMovies: "No",
  Contract: "Month-to-month",
  PaperlessBilling: "Yes",
  PaymentMethod: "Electronic check",
  MonthlyCharges: 70,
  TotalCharges: 840,
};

function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const updateField = (event) => {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]:
        name === "SeniorCitizen" ||
        name === "tenure" ||
        name === "MonthlyCharges" ||
        name === "TotalCharges"
          ? Number(value)
          : value,
    }));
  };

  const predictChurn = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ customer: form }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Prediction request failed.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Unable to connect to the API. Make sure your FastAPI server is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setForm(initialForm);
    setResult(null);
    setError("");
  };

  const selectField = (label, name, options) => (
    <div className="field" key={name}>
      <label htmlFor={name}>{label}</label>
      <select
        id={name}
        name={name}
        value={form[name]}
        onChange={updateField}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );

  const numberField = (label, name, min, max, step = 1) => (
    <div className="field" key={name}>
      <label htmlFor={name}>{label}</label>
      <input
        id={name}
        type="number"
        name={name}
        min={min}
        max={max}
        step={step}
        value={form[name]}
        onChange={updateField}
        required
      />
    </div>
  );

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">C</div>
          <div>
            <h2>ChurnIQ</h2>
            <span>Prediction Platform</span>
          </div>
        </div>

        <div className="sidebar-label">WORKSPACE</div>
        <div className="nav-item active">◈ &nbsp; Churn Prediction</div>
        <div className="nav-item">▤ &nbsp; Customer Analysis</div>
        <div className="nav-item">◷ &nbsp; Model Insights</div>

        <div className="sidebar-bottom">
          <div className="status-dot"></div>
          <span>Prediction API</span>
          <strong>Local</strong>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="breadcrumb">WORKSPACE / PREDICTION</p>
            <h1>Customer Churn Prediction</h1>
            <p className="subtitle">
              Analyze customer information and estimate churn probability.
            </p>
          </div>
          <div className="model-badge">
            <span className="status-dot"></span>
            Logistic Regression Model
          </div>
        </header>

        <section className="welcome-card">
          <div>
            <span className="eyebrow">AI-POWERED CUSTOMER ANALYTICS</span>
            <h2>Understand your customers before they leave.</h2>
            <p>
              Enter customer information below to generate a machine learning
              prediction.
            </p>
          </div>
          <div className="welcome-symbol">↗</div>
        </section>

        <form onSubmit={predictChurn}>
          <section className="form-card">
            <div className="section-heading">
              <div>
                <span className="section-number">01</span>
                <h2>Customer Information</h2>
                <p>Basic profile and account information</p>
              </div>
            </div>

            <div className="form-grid">
              {selectField("Gender", "gender", [
                { value: "Male", label: "Male" },
                { value: "Female", label: "Female" },
              ])}

              {selectField("Senior Citizen", "SeniorCitizen", [
                { value: 0, label: "No" },
                { value: 1, label: "Yes" },
              ])}

              {selectField("Partner", "Partner", [
                { value: "Yes", label: "Yes" },
                { value: "No", label: "No" },
              ])}

              {selectField("Dependents", "Dependents", [
                { value: "Yes", label: "Yes" },
                { value: "No", label: "No" },
              ])}

              {numberField("Tenure (months)", "tenure", 0, 100)}

              {selectField("Phone Service", "PhoneService", [
                { value: "Yes", label: "Yes" },
                { value: "No", label: "No" },
              ])}

              {selectField("Multiple Lines", "MultipleLines", [
                { value: "No", label: "No" },
                { value: "Yes", label: "Yes" },
                { value: "No phone service", label: "No phone service" },
              ])}
            </div>
          </section>

          <section className="form-card">
            <div className="section-heading">
              <div>
                <span className="section-number">02</span>
                <h2>Services & Subscription</h2>
                <p>Internet, support and entertainment services</p>
              </div>
            </div>

            <div className="form-grid">
              {selectField("Internet Service", "InternetService", [
                { value: "DSL", label: "DSL" },
                { value: "Fiber optic", label: "Fiber optic" },
                { value: "No", label: "No internet service" },
              ])}

              {selectField("Online Security", "OnlineSecurity", [
                { value: "Yes", label: "Yes" },
                { value: "No", label: "No" },
                { value: "No internet service", label: "No internet service" },
              ])}

              {selectField("Online Backup", "OnlineBackup", [
                { value: "Yes", label: "Yes" },
                { value: "No", label: "No" },
                { value: "No internet service", label: "No internet service" },
              ])}

              {selectField("Device Protection", "DeviceProtection", [
                { value: "Yes", label: "Yes" },
                { value: "No", label: "No" },
                { value: "No internet service", label: "No internet service" },
              ])}

              {selectField("Tech Support", "TechSupport", [
                { value: "Yes", label: "Yes" },
                { value: "No", label: "No" },
                { value: "No internet service", label: "No internet service" },
              ])}

              {selectField("Streaming TV", "StreamingTV", [
                { value: "Yes", label: "Yes" },
                { value: "No", label: "No" },
                { value: "No internet service", label: "No internet service" },
              ])}

              {selectField("Streaming Movies", "StreamingMovies", [
                { value: "Yes", label: "Yes" },
                { value: "No", label: "No" },
                { value: "No internet service", label: "No internet service" },
              ])}
            </div>
          </section>

          <section className="form-card">
            <div className="section-heading">
              <div>
                <span className="section-number">03</span>
                <h2>Billing & Contract</h2>
                <p>Payment preferences and financial information</p>
              </div>
            </div>

            <div className="form-grid">
              {selectField("Contract", "Contract", [
                { value: "Month-to-month", label: "Month-to-month" },
                { value: "One year", label: "One year" },
                { value: "Two year", label: "Two year" },
              ])}

              {selectField("Paperless Billing", "PaperlessBilling", [
                { value: "Yes", label: "Yes" },
                { value: "No", label: "No" },
              ])}

              {selectField("Payment Method", "PaymentMethod", [
                { value: "Electronic check", label: "Electronic check" },
                {
                  value: "Mailed check",
                  label: "Mailed check",
                },
                {
                  value: "Bank transfer (automatic)",
                  label: "Bank transfer (automatic)",
                },
                {
                  value: "Credit card (automatic)",
                  label: "Credit card (automatic)",
                },
              ])}

              {numberField("Monthly Charges ($)", "MonthlyCharges", 0, 1000, 0.01)}

              {numberField("Total Charges ($)", "TotalCharges", 0, 100000, 0.01)}
            </div>
          </section>

          <div className="form-actions">
            <button type="button" className="reset-button" onClick={resetForm}>
              Reset Form
            </button>
            <button type="submit" className="predict-button" disabled={loading}>
              {loading ? "Analyzing Customer..." : "Generate Prediction →"}
            </button>
          </div>
        </form>

        {error && <div className="error-message">{error}</div>}

        {result && (
          <section className="result-card">
            <span className="eyebrow">MODEL PREDICTION RESULT</span>
            <h2>
              {result.prediction === "Churn"
                ? "Customer is likely to churn"
                : "Customer is unlikely to churn"}
            </h2>
            <div className="probability">
              {Number(result.churn_probability).toFixed(2)}%
            </div>
            <p>Estimated churn probability</p>
            <div className="probability-track">
              <div
                className="probability-fill"
                style={{ width: `${result.churn_probability}%` }}
              ></div>
            </div>
            <div className="result-details">
              <span>Classification threshold</span>
              <strong>
                {(Number(result.threshold) * 100).toFixed(2)}%
              </strong>
            </div>
            <div
              className={`prediction-status ${
                result.prediction === "Churn" ? "high-risk" : "low-risk"
              }`}
            >
              {result.prediction}
            </div>
          </section>
        )}

        <footer>
          ChurnIQ · Machine Learning Customer Analytics
        </footer>
      </main>
    </div>
  );
}

export default App;
