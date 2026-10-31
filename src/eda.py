
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/processed/customer_churn_clean.csv")

# Professional visualization settings
sns.set_theme(style="whitegrid")

# Calculate churn distribution
churn_counts = df["Churn"].value_counts()
churn_percentage = df["Churn"].value_counts(normalize=True) * 100

print("CUSTOMER CHURN DISTRIBUTION")
print("-" * 40)

print("\nCustomer Counts:")
print(churn_counts)

print("\nCustomer Percentages:")
print(churn_percentage.round(2))

# Create visualization
plt.figure(figsize=(8, 5))

ax = sns.countplot(
    data=df,
    x="Churn",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Customer Churn Distribution", fontsize=15)
plt.xlabel("Customer Churn")
plt.ylabel("Number of Customers")

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

# Save visualization
plt.savefig(
    "reports/churn_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# CONTRACT TYPE VS CUSTOMER CHURN
# ==========================================

print("\nCONTRACT TYPE VS CUSTOMER CHURN")
print("-" * 45)

# Calculate churn rate for each contract type
contract_churn = pd.crosstab(
    df["Contract"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Contract Type:")
print(contract_churn.round(2))

# Create visualization
plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="Contract",
    y=df["Churn"].map({"Yes": 1, "No": 0}),
    hue="Contract",
    palette="Set2",
    legend=False,
    errorbar=None
)

plt.title("Customer Churn Rate by Contract Type", fontsize=15)
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate")

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f"{y:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/contract_churn_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# CUSTOMER TENURE VS CHURN
# ==========================================

print("\nCUSTOMER TENURE VS CHURN")
print("-" * 45)

# Calculate tenure statistics
tenure_analysis = df.groupby("Churn")["tenure"].agg(
    ["count", "mean", "median", "min", "max"]
)

print("\nTenure Statistics by Churn:")
print(tenure_analysis.round(2))

# Create visualization
plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="tenure",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Customer Tenure Distribution by Churn Status",
          fontsize=14)

plt.xlabel("Customer Churn")
plt.ylabel("Tenure (Months)")

plt.tight_layout()

plt.savefig(
    "reports/tenure_churn_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# MONTHLY CHARGES VS CHURN
# ==========================================

print("\nMONTHLY CHARGES VS CHURN")
print("-" * 45)

# Calculate monthly charge statistics
charges_analysis = df.groupby("Churn")["MonthlyCharges"].agg(
    ["count", "mean", "median", "min", "max"]
)

print("\nMonthly Charges Statistics:")
print(charges_analysis.round(2))

# Create box plot
plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges",
    hue="Churn",
    palette="Set2",
    legend=False
)

plt.title("Monthly Charges Distribution by Churn Status")
plt.xlabel("Customer Churn")
plt.ylabel("Monthly Charges")

plt.tight_layout()

plt.savefig(
    "reports/monthly_charges_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# INTERNET SERVICE VS CUSTOMER CHURN
# ==========================================

print("\nINTERNET SERVICE VS CHURN")
print("-" * 45)

# Calculate churn percentage for each internet service
internet_analysis = pd.crosstab(
    df["InternetService"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Internet Service:")
print(internet_analysis.round(2))

# Create visualization
plt.figure(figsize=(9, 5))

sns.barplot(
    data=df,
    x="InternetService",
    y=df["Churn"].eq("Yes").astype(int),
    hue="InternetService",
    palette="Set2",
    estimator="mean",
    errorbar=None,
    legend=False
)

plt.title("Customer Churn Rate by Internet Service")
plt.xlabel("Internet Service Type")
plt.ylabel("Churn Rate")

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/internet_service_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# PAYMENT METHOD VS CUSTOMER CHURN
# ==========================================

print("\nPAYMENT METHOD VS CHURN")
print("-" * 45)

# Calculate churn percentage
payment_analysis = pd.crosstab(
    df["PaymentMethod"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Payment Method:")
print(payment_analysis.round(2))

# Prepare churn rate for visualization
payment_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
    .groupby("PaymentMethod")["ChurnFlag"]
    .mean()
    .reset_index()
)

# Create visualization
plt.figure(figsize=(10, 5))

sns.barplot(
    data=payment_churn,
    x="PaymentMethod",
    y="ChurnFlag",
    hue="PaymentMethod",
    palette="Set2",
    legend=False,
    errorbar=None
)

plt.title("Customer Churn Rate by Payment Method")
plt.xlabel("Payment Method")
plt.ylabel("Churn Rate")

plt.xticks(rotation=15)

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/payment_method_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# ONLINE SECURITY VS CUSTOMER CHURN
# ==========================================

print("\nONLINE SECURITY VS CHURN")
print("-" * 45)

security_analysis = pd.crosstab(
    df["OnlineSecurity"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Online Security:")
print(security_analysis.round(2))

# Calculate churn rate
security_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
    .groupby("OnlineSecurity")["ChurnFlag"]
    .mean()
    .reset_index()
)

# Visualization
plt.figure(figsize=(8, 5))

sns.barplot(
    data=security_churn,
    x="OnlineSecurity",
    y="ChurnFlag",
    hue="OnlineSecurity",
    palette="Set2",
    legend=False,
    errorbar=None
)

plt.title("Customer Churn Rate by Online Security")
plt.xlabel("Online Security")
plt.ylabel("Churn Rate")

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/online_security_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# ONLINE BACKUP VS CUSTOMER CHURN
# ==========================================

print("\nONLINE BACKUP VS CHURN")
print("-" * 45)

backup_analysis = pd.crosstab(
    df["OnlineBackup"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Online Backup:")
print(backup_analysis.round(2))

# Calculate churn rate
backup_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
    .groupby("OnlineBackup")["ChurnFlag"]
    .mean()
    .reset_index()
)

# Visualization
plt.figure(figsize=(8, 5))

sns.barplot(
    data=backup_churn,
    x="OnlineBackup",
    y="ChurnFlag",
    hue="OnlineBackup",
    palette="Set2",
    legend=False,
    errorbar=None
)

plt.title("Customer Churn Rate by Online Backup")
plt.xlabel("Online Backup")
plt.ylabel("Churn Rate")

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/online_backup_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# TECH SUPPORT VS CUSTOMER CHURN
# ==========================================

print("\nTECH SUPPORT VS CHURN")
print("-" * 45)

support_analysis = pd.crosstab(
    df["TechSupport"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Tech Support:")
print(support_analysis.round(2))

# Calculate churn rate
support_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
    .groupby("TechSupport")["ChurnFlag"]
    .mean()
    .reset_index()
)

# Visualization
plt.figure(figsize=(8, 5))

sns.barplot(
    data=support_churn,
    x="TechSupport",
    y="ChurnFlag",
    hue="TechSupport",
    palette="Set2",
    legend=False,
    errorbar=None
)

plt.title("Customer Churn Rate by Tech Support")
plt.xlabel("Tech Support")
plt.ylabel("Churn Rate")

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/tech_support_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# DEVICE PROTECTION VS CUSTOMER CHURN
# ==========================================

print("\nDEVICE PROTECTION VS CHURN")
print("-" * 45)

device_analysis = pd.crosstab(
    df["DeviceProtection"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Device Protection:")
print(device_analysis.round(2))

# Calculate churn rate
device_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
    .groupby("DeviceProtection")["ChurnFlag"]
    .mean()
    .reset_index()
)

# Visualization
plt.figure(figsize=(8, 5))

sns.barplot(
    data=device_churn,
    x="DeviceProtection",
    y="ChurnFlag",
    hue="DeviceProtection",
    palette="Set2",
    legend=False,
    errorbar=None
)

plt.title("Customer Churn Rate by Device Protection")
plt.xlabel("Device Protection")
plt.ylabel("Churn Rate")

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/device_protection_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# STREAMING TV VS CUSTOMER CHURN
# ==========================================

print("\nSTREAMING TV VS CHURN")
print("-" * 45)

streaming_analysis = pd.crosstab(
    df["StreamingTV"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Streaming TV:")
print(streaming_analysis.round(2))

# Calculate churn rate
streaming_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
    .groupby("StreamingTV")["ChurnFlag"]
    .mean()
    .reset_index()
)

# Visualization
plt.figure(figsize=(8, 5))

sns.barplot(
    data=streaming_churn,
    x="StreamingTV",
    y="ChurnFlag",
    hue="StreamingTV",
    palette="Set2",
    legend=False,
    errorbar=None
)

plt.title("Customer Churn Rate by Streaming TV")
plt.xlabel("Streaming TV")
plt.ylabel("Churn Rate")

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/streaming_tv_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# STREAMING MOVIES VS CUSTOMER CHURN
# ==========================================

print("\nSTREAMING MOVIES VS CHURN")
print("-" * 45)

movies_analysis = pd.crosstab(
    df["StreamingMovies"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Streaming Movies:")
print(movies_analysis.round(2))

# Calculate churn rate
movies_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
    .groupby("StreamingMovies")["ChurnFlag"]
    .mean()
    .reset_index()
)

# Visualization
plt.figure(figsize=(8, 5))

sns.barplot(
    data=movies_churn,
    x="StreamingMovies",
    y="ChurnFlag",
    hue="StreamingMovies",
    palette="Set2",
    legend=False,
    errorbar=None
)

plt.title("Customer Churn Rate by Streaming Movies")
plt.xlabel("Streaming Movies")
plt.ylabel("Churn Rate")

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/streaming_movies_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# MULTIPLE LINES VS CUSTOMER CHURN
# ==========================================

print("\nMULTIPLE LINES VS CHURN")
print("-" * 45)

lines_analysis = pd.crosstab(
    df["MultipleLines"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Multiple Lines:")
print(lines_analysis.round(2))

# Calculate churn rate
lines_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
    .groupby("MultipleLines")["ChurnFlag"]
    .mean()
    .reset_index()
)

# Visualization
plt.figure(figsize=(8, 5))

sns.barplot(
    data=lines_churn,
    x="MultipleLines",
    y="ChurnFlag",
    hue="MultipleLines",
    palette="Set2",
    legend=False,
    errorbar=None
)

plt.title("Customer Churn Rate by Multiple Lines")
plt.xlabel("Multiple Lines")
plt.ylabel("Churn Rate")

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/multiple_lines_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# PHONE SERVICE VS CUSTOMER CHURN
# ==========================================

print("\nPHONE SERVICE VS CHURN")
print("-" * 45)

phone_analysis = pd.crosstab(
    df["PhoneService"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Phone Service:")
print(phone_analysis.round(2))

# Calculate churn rate
phone_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
    .groupby("PhoneService")["ChurnFlag"]
    .mean()
    .reset_index()
)

# Visualization
plt.figure(figsize=(8, 5))

sns.barplot(
    data=phone_churn,
    x="PhoneService",
    y="ChurnFlag",
    hue="PhoneService",
    palette="Set2",
    legend=False,
    errorbar=None
)

plt.title("Customer Churn Rate by Phone Service")
plt.xlabel("Phone Service")
plt.ylabel("Churn Rate")

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/phone_service_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# PAPERLESS BILLING VS CUSTOMER CHURN
# ==========================================

print("\nPAPERLESS BILLING VS CHURN")
print("-" * 45)

paperless_analysis = pd.crosstab(
    df["PaperlessBilling"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Paperless Billing:")
print(paperless_analysis.round(2))

paperless_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
    .groupby("PaperlessBilling")["ChurnFlag"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=paperless_churn,
    x="PaperlessBilling",
    y="ChurnFlag",
    hue="PaperlessBilling",
    palette="Set2",
    legend=False,
    errorbar=None
)

plt.title("Customer Churn Rate by Paperless Billing")
plt.xlabel("Paperless Billing")
plt.ylabel("Churn Rate")

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/paperless_billing_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# SENIOR CITIZEN VS CUSTOMER CHURN
# ==========================================

print("\nSENIOR CITIZEN VS CHURN")
print("-" * 45)

senior_analysis = pd.crosstab(
    df["SeniorCitizen"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Senior Citizen:")
print(senior_analysis.round(2))

senior_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
    .groupby("SeniorCitizen")["ChurnFlag"]
    .mean()
    .reset_index()
)

# Convert numerical labels into readable categories
senior_churn["SeniorCitizen"] = senior_churn[
    "SeniorCitizen"
].map({0: "No", 1: "Yes"})

plt.figure(figsize=(8, 5))

sns.barplot(
    data=senior_churn,
    x="SeniorCitizen",
    y="ChurnFlag",
    hue="SeniorCitizen",
    palette="Set2",
    legend=False,
    errorbar=None
)

plt.title("Customer Churn Rate by Senior Citizen")
plt.xlabel("Senior Citizen")
plt.ylabel("Churn Rate")

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/senior_citizen_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# PARTNER VS CUSTOMER CHURN
# ==========================================

print("\nPARTNER VS CHURN")
print("-" * 45)

partner_analysis = pd.crosstab(
    df["Partner"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Partner:")
print(partner_analysis.round(2))

partner_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
    .groupby("Partner")["ChurnFlag"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=partner_churn,
    x="Partner",
    y="ChurnFlag",
    hue="Partner",
    palette="Set2",
    legend=False,
    errorbar=None
)

plt.title("Customer Churn Rate by Partner Status")
plt.xlabel("Has Partner")
plt.ylabel("Churn Rate")

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/partner_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# DEPENDENTS VS CUSTOMER CHURN
# ==========================================

print("\nDEPENDENTS VS CHURN")
print("-" * 45)

dependents_analysis = pd.crosstab(
    df["Dependents"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Dependents:")
print(dependents_analysis.round(2))

dependents_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
    .groupby("Dependents")["ChurnFlag"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=dependents_churn,
    x="Dependents",
    y="ChurnFlag",
    hue="Dependents",
    palette="Set2",
    legend=False,
    errorbar=None
)

plt.title("Customer Churn Rate by Dependents")
plt.xlabel("Has Dependents")
plt.ylabel("Churn Rate")

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/dependents_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ==========================================
# GENDER VS CUSTOMER CHURN
# ==========================================

print("\nGENDER VS CHURN")
print("-" * 45)

gender_analysis = pd.crosstab(
    df["gender"],
    df["Churn"],
    normalize="index"
) * 100

print("\nChurn Percentage by Gender:")
print(gender_analysis.round(2))

gender_churn = (
    df.assign(ChurnFlag=df["Churn"].eq("Yes").astype(int))
    .groupby("gender")["ChurnFlag"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 5))

sns.barplot(
    data=gender_churn,
    x="gender",
    y="ChurnFlag",
    hue="gender",
    palette="Set2",
    legend=False,
    errorbar=None
)

plt.title("Customer Churn Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Churn Rate")

plt.gca().yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.0%}")
)

plt.tight_layout()

plt.savefig(
    "reports/gender_churn.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================
# CORRELATION ANALYSIS
# ==========================================

print("\nCORRELATION ANALYSIS")
print("=" * 50)

# Select numerical columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Calculate correlation matrix
correlation_matrix = numeric_df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Plot correlation heatmap
plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)

plt.title("Correlation Heatmap of Customer Churn Dataset")
plt.tight_layout()

plt.savefig(
    "reports/figures/correlation_heatmap.png",
    dpi=300
)

plt.show()

# ==========================================
# NUMERICAL FEATURES VS CUSTOMER CHURN
# ==========================================

print("\nNUMERICAL FEATURES VS CUSTOMER CHURN")
print("=" * 50)

# 1. Tenure vs Churn
plt.figure(figsize=(8, 5))

sns.boxplot(
    x='Churn',
    y='tenure',
    data=df,
    palette='Set2'
)

plt.title("Customer Tenure vs Churn")
plt.xlabel("Customer Churn")
plt.ylabel("Tenure (Months)")
plt.tight_layout()

plt.savefig(
    "reports/figures/tenure_vs_churn.png",
    dpi=300
)

plt.show()


# 2. Monthly Charges vs Churn
plt.figure(figsize=(8, 5))

sns.boxplot(
    x='Churn',
    y='MonthlyCharges',
    data=df,
    palette='Set2'
)

plt.title("Monthly Charges vs Churn")
plt.xlabel("Customer Churn")
plt.ylabel("Monthly Charges")
plt.tight_layout()

plt.savefig(
    "reports/figures/monthly_charges_vs_churn.png",
    dpi=300
)

plt.show()


# 3. Total Charges vs Churn
plt.figure(figsize=(8, 5))

sns.boxplot(
    x='Churn',
    y='tenure',
    data=df,
    hue='Churn',
    palette='Set2',
    legend=False
)

plt.title("Total Charges vs Churn")
plt.xlabel("Customer Churn")
plt.ylabel("Total Charges")
plt.tight_layout()

plt.savefig(
    "reports/figures/total_charges_vs_churn.png",
    dpi=300
)

plt.show()

# ==========================================
# CATEGORICAL FEATURE CHURN COMPARISON
# ==========================================

print("\nCATEGORICAL FEATURE CHURN ANALYSIS")
print("=" * 50)

categorical_features = [
    'Contract',
    'InternetService',
    'PaymentMethod',
    'OnlineSecurity',
    'TechSupport',
    'DeviceProtection',
    'PaperlessBilling',
    'SeniorCitizen',
    'Partner',
    'Dependents'
]

fig, axes = plt.subplots(5, 2, figsize=(16, 24))

axes = axes.flatten()

for i, feature in enumerate(categorical_features):

    # Calculate churn percentage
    churn_rate = (
        df.groupby(feature)['Churn']
        .apply(lambda x: (x == 'Yes').mean() * 100)
        .reset_index(name='ChurnRate')
    )

    # Convert SeniorCitizen values for readability
    if feature == 'SeniorCitizen':
        churn_rate[feature] = churn_rate[feature].map({
            0: 'No',
            1: 'Yes'
        })

    # Plot churn percentage
    sns.barplot(
        data=churn_rate,
        x=feature,
        y='ChurnRate',
        hue=feature,
        palette='Set2',
        legend=False,
        ax=axes[i]
    )

    axes[i].set_title(f'Churn Rate by {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Churn Rate (%)')

    axes[i].tick_params(
        axis='x',
        rotation=25
    )

    # Display percentage values
    for container in axes[i].containers:
        axes[i].bar_label(
            container,
            fmt='%.1f%%',
            padding=2
        )

plt.suptitle(
    'Categorical Features vs Customer Churn',
    fontsize=18,
    fontweight='bold'
)

plt.tight_layout(rect=[0, 0, 1, 0.98])

plt.savefig(
    'reports/figures/categorical_churn_comparison.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()


# Print individual churn rates
for feature in categorical_features:

    print(f"\nChurn Percentage by {feature}:")

    result = (
        df.groupby(feature)['Churn']
        .apply(lambda x: (x == 'Yes').mean() * 100)
    )

    print(result.round(2))

















