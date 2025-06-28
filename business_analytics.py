# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
import os

# === Step 1: Load data ===
df = pd.read_csv("sales_data_large.csv", parse_dates=["Order Date"])
output_dir = "ba_output"
os.makedirs(output_dir, exist_ok = True)

#first overview
print("Preview of the Data:")
print(df.head())
print("\nData description:")
print(df.describe())

# === Step 2: Clean up ===
df = df.dropna()
df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Month"] = df["Order Date"].dt.to_period("M")
df["Total Value"] = df["Sales"] * df["Quantity"]

# === Step 3: Customer analysis ===
customer_sales = df.groupby("Customer").agg({
    "Order ID": "nunique",
    "Total Value": "sum",
    "Sales": "mean",
    "Quantity": "sum"
}).rename(columns={
    "Order ID": "Unique Orders",
    "Total Value": "Total Revenue",
    "Sales": "Avg Basket",
    "Quantity": "Total Items"
}).reset_index()

# recognize repeat buyers
repeat_customers = df.groupby("Customer")["Order Date"].nunique().reset_index()
repeat_customers["Repeat Buyer"] = repeat_customers["Order Date"] > 1
customer_sales = customer_sales.merge(repeat_customers[["Customer", "Repeat Buyer"]], on="Customer")

print("\nTop 5 Customers:")
print(customer_sales.sort_values("Total Revenue", ascending=False).head())

# === Step 4: Sales analysis ===
monthly_sales = df.groupby("Month")["Sales"].sum().reset_index()
monthly_sales["Month"] = monthly_sales["Month"].astype(str)

plt.figure(figsize=(10,5))
sns.lineplot(data=monthly_sales, x="Month", y="Sales", marker="o")
plt.xticks(rotation=45)
plt.title("Monthly sales")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "monthly_sales_trend.png"))
plt.close()

# === Step 5: Product analysis ===
product_perf = df.groupby("Product").agg({
    "Sales": "sum",
    "Quantity": "sum",
    "Order ID": "count"
}).rename(columns={"Order ID": "Num Orders"}).reset_index()

plt.figure(figsize=(10,5))
sns.barplot(data=product_perf.sort_values("Sales", ascending=False), x="Product", y="Sales")
plt.title("Top products by sales")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top_products.png"))
plt.close()

# === Step 6: Heatmap correlations ===
plt.figure(figsize=(8,6))
corr = customer_sales[["Unique Orders", "Total Revenue", "Avg Basket", "Total Items"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlations between customer metrics")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()

# Clustering
clustering_data = customer_sales[["Total Revenue", "Unique Orders", "Avg Basket"]].copy()
scaler = StandardScaler()
scaled = scaler.fit_transform(clustering_data)

kmeans = KMeans(n_clusters=4, random_state=42)
customer_sales["Segment"] = kmeans.fit_predict(scaled)

# === Step 7: Segment analysis ===
segment_summary = customer_sales.groupby("Segment").agg({
    "Total Revenue": "mean",
    "Avg Basket": "mean",
    "Unique Orders": "mean",
    "Customer": "count"
}).rename(columns={"Customer": "Num Customers"})

print("\nSegment overview:")
print(segment_summary)

# === Step 8: Visualization: Customer clusters ===
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=customer_sales["Total Revenue"],
    y=customer_sales["Avg Basket"],
    hue=customer_sales["Segment"],
    palette="tab10"
)
plt.title("Customer segments by sales and shopping cart size")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "customer_segments.png"))
plt.close()

# === Step 9: Export results ===
customer_sales.to_csv(os.path.join(output_dir, "customer_segmentation.csv"), index=False)
segment_summary.to_csv(os.path.join(output_dir, "segment_summary.csv"))

report_path = os.path.join(output_dir, "summary_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=== Business Analytics Summary ===\n\n")
    
    f.write("Top 5 Customers by Total Revenue:\n")
    top_customers = customer_sales.sort_values("Total Revenue", ascending=False).head()
    f.write(top_customers.to_string(index=False))
    f.write("\n\n")

    f.write("Monthly Sales Overview:\n")
    f.write(monthly_sales.to_string(index=False))
    f.write("\n\n")

    f.write("Top Products by Sales:\n")
    top_products = product_perf.sort_values("Sales", ascending=False).head()
    f.write(top_products.to_string(index=False))
    f.write("\n\n")

    f.write("Customer Segment Summary:\n")
    f.write(segment_summary.to_string())
    f.write("\n\n")

    f.write("Correlations Between Customer Metrics:\n")
    f.write(corr.to_string())

print("\nReports saved.")