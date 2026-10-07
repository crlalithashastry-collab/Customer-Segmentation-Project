# ============================================================
# CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 60)
print("CUSTOMER SEGMENTATION PROJECT")
print("=" * 60)

print("\nLoading dataset...")

# Dataset downloaded from the Customer Personality Analysis dataset
# The file uses semicolon (;) as the separator.
dataframe = pd.read_csv(
    "customer segmentation.csv",
    sep=";"
)

print("\nDataset loaded successfully!")

print("\nShape before cleaning:")
print(dataframe.shape)

print("\nFirst 5 rows:")
print(dataframe.head())

print("\nColumn names:")
print(dataframe.columns.tolist())


# ============================================================
# 2. CHECK DATA INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATA INFORMATION")
print("=" * 60)

print("\nDataset information:")
print(dataframe.info())

print("\nMissing values:")
print(dataframe.isnull().sum())


# ============================================================
# 3. REMOVE MISSING VALUES
# ============================================================

print("\nRemoving rows containing missing values...")

dataframe.dropna(inplace=True)

print("Shape after removing missing values:")
print(dataframe.shape)


# ============================================================
# 4. BASIC DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("BASIC DATA ANALYSIS")
print("=" * 60)

print("\nStatistical summary:")
print(dataframe.describe())

print("\nEducation distribution:")
print(dataframe["Education"].value_counts())

print("\nMarital Status distribution:")
print(dataframe["Marital_Status"].value_counts())


# ============================================================
# 5. DATA TYPE CONVERSION
# ============================================================

print("\nConverting customer registration date...")

dataframe["Dt_Customer"] = pd.to_datetime(
    dataframe["Dt_Customer"],
    errors="coerce"
)


# ============================================================
# 6. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

# Calculate customer age
dataframe["Age"] = 2025 - dataframe["Year_Birth"]

# Calculate total children
dataframe["Total_Children"] = (
    dataframe["Kidhome"] +
    dataframe["Teenhome"]
)

# Product spending columns
spending_columns = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

# Calculate total spending
dataframe["Total_Spending"] = dataframe[
    spending_columns
].sum(axis=1)

# Calculate how long the customer has been registered
dataframe["Customer_Since"] = (
    pd.Timestamp.today() -
    dataframe["Dt_Customer"]
).dt.days

print("\nNew features created:")
print("- Age")
print("- Total_Children")
print("- Total_Spending")
print("- Customer_Since")


# ============================================================
# 7. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# ---------------- AGE DISTRIBUTION ----------------

plt.figure(figsize=(8, 5))

sns.histplot(
    dataframe["Age"],
    bins=30,
    kde=True
)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ---------------- INCOME DISTRIBUTION ----------------

plt.figure(figsize=(8, 5))

sns.histplot(
    dataframe["Income"],
    bins=30,
    kde=True
)

plt.title("Customer Income Distribution")
plt.xlabel("Income")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ---------------- TOTAL SPENDING ----------------

plt.figure(figsize=(8, 5))

sns.histplot(
    dataframe["Total_Spending"],
    bins=30,
    kde=True
)

plt.title("Customer Total Spending Distribution")
plt.xlabel("Total Spending")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.show()


# ---------------- INCOME BY EDUCATION ----------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    x="Education",
    y="Income",
    data=dataframe
)

plt.title("Income by Education Level")
plt.xlabel("Education")
plt.ylabel("Income")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ---------------- SPENDING BY MARITAL STATUS ----------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    x="Marital_Status",
    y="Total_Spending",
    data=dataframe
)

plt.title("Total Spending by Marital Status")
plt.xlabel("Marital Status")
plt.ylabel("Total Spending")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ============================================================
# 8. CORRELATION HEATMAP
# ============================================================

correlation_columns = [
    "Age",
    "Income",
    "Total_Spending",
    "NumWebPurchases",
    "NumStorePurchases"
]

plt.figure(figsize=(9, 7))

sns.heatmap(
    dataframe[correlation_columns].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Customer Feature Correlation")

plt.tight_layout()
plt.show()


# ============================================================
# 9. ADDITIONAL CUSTOMER ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("ADDITIONAL CUSTOMER ANALYSIS")
print("=" * 60)


# Average income by education and marital status

pivot_table = dataframe.pivot_table(
    values="Income",
    index="Education",
    columns="Marital_Status",
    aggfunc="mean"
)

print("\nAverage Income by Education and Marital Status:")

print(pivot_table)


# Average spending by education

average_spending = dataframe.groupby(
    "Education"
)["Total_Spending"].mean()

print("\nAverage Spending by Education:")

print(average_spending)


# Campaign response by marital status

campaign_response = dataframe.groupby(
    "Marital_Status"
)["Response"].mean()

print("\nCampaign Response by Marital Status:")

print(campaign_response)


# ============================================================
# 10. AGE GROUP ANALYSIS
# ============================================================

bins = [
    0,
    30,
    40,
    50,
    60,
    70,
    100
]

labels = [
    "18-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "70+"
]

dataframe["Age_Group"] = pd.cut(
    dataframe["Age"],
    bins=bins,
    labels=labels,
    right=False
)

age_income = dataframe.groupby(
    "Age_Group",
    observed=False
)["Income"].mean()

print("\nAverage Income by Age Group:")

print(age_income)


# ============================================================
# 11. SELECT FEATURES FOR CLUSTERING
# ============================================================

print("\n" + "=" * 60)
print("CUSTOMER SEGMENTATION")
print("=" * 60)

features = [
    "Age",
    "Income",
    "Total_Spending",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth"
]

print("\nFeatures used for clustering:")

for feature in features:
    print("-", feature)


# Create feature dataset

X = dataframe[features]


# ============================================================
# 12. FEATURE SCALING
# ============================================================

print("\nScaling customer features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Feature scaling completed!")


# ============================================================
# 13. ELBOW METHOD
# ============================================================

print("\nCalculating optimal number of clusters...")

inertia = []

for k in range(2, 10):

    kmeans_test = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans_test.fit(X_scaled)

    inertia.append(
        kmeans_test.inertia_
    )


# Plot elbow graph

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 10),
    inertia,
    marker="o"
)

plt.title("Elbow Method for Optimal Number of Clusters")

plt.xlabel("Number of Clusters")

plt.ylabel("Inertia")

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# 14. FINAL K-MEANS MODEL
# ============================================================

print("\nTraining final K-Means model...")

kmeans = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=10
)

dataframe["Cluster"] = kmeans.fit_predict(
    X_scaled
)

print("K-Means clustering completed!")


# ============================================================
# 15. CLUSTER ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("CLUSTER ANALYSIS")
print("=" * 60)


# Average characteristics of every cluster

cluster_summary = dataframe.groupby(
    "Cluster"
)[features].mean().round(2)

print("\nAverage characteristics of each customer segment:")

print(cluster_summary)


# Number of customers in each cluster

cluster_counts = dataframe[
    "Cluster"
].value_counts().sort_index()

print("\nNumber of customers in each cluster:")

print(cluster_counts)


# ============================================================
# 16. PCA VISUALIZATION
# ============================================================

print("\nCreating PCA visualization...")

pca = PCA(
    n_components=2
)

pca_data = pca.fit_transform(
    X_scaled
)

dataframe["PCA1"] = pca_data[:, 0]

dataframe["PCA2"] = pca_data[:, 1]


# Plot clusters

plt.figure(figsize=(10, 7))

sns.scatterplot(
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    data=dataframe,
    palette="Set1",
    s=80
)

plt.title(
    "Customer Segmentation Using K-Means"
)

plt.xlabel("PCA Component 1")

plt.ylabel("PCA Component 2")

plt.legend(
    title="Customer Segment"
)

plt.tight_layout()

plt.show()


# ============================================================
# 17. SAVE MODEL AND SCALER
# ============================================================

print("\n" + "=" * 60)
print("SAVING MODEL")
print("=" * 60)

joblib.dump(
    kmeans,
    "model.pkl"
)

joblib.dump(
    scaler,
    "scaler.pkl"
)

print("\nModel saved as: model.pkl")

print("Scaler saved as: scaler.pkl")


# ============================================================
# 18. SAVE CLUSTER SUMMARY
# ============================================================

cluster_summary.to_csv(
    "cluster_summary.csv"
)

print(
    "Cluster summary saved as: cluster_summary.csv"
)


# ============================================================
# 19. SAVE SEGMENTED CUSTOMER DATA
# ============================================================

dataframe.to_csv(
    "segmented_customers.csv",
    index=False
)

print(
    "Segmented customer data saved as: segmented_customers.csv"
)


# ============================================================
# 20. PROJECT COMPLETED
# ============================================================

print("\n" + "=" * 60)

print(
    "CUSTOMER SEGMENTATION PROJECT COMPLETED SUCCESSFULLY!"
)

print("=" * 60)

print("\nGenerated files:")

print("1. model.pkl")
print("2. scaler.pkl")
print("3. cluster_summary.csv")
print("4. segmented_customers.csv")

print("\nYou can now run the Streamlit application using:")

print("streamlit run segmentation.py")

print("=" * 60)