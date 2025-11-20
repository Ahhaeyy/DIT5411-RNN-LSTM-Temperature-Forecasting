import pandas as pd

# ============================
# FIXED DATA PREPROCESSING
# ============================

# 1. Load CSV (skip ONLY the first metadata row)
df = pd.read_csv("daily_HKO_GMT_ALL.csv", skiprows=1)

# Rename the columns properly
df.columns = ["Year", "Month", "Day", "Value", "Data_Completeness"]

# Convert to numeric
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
df["Day"] = pd.to_numeric(df["Day"], errors="coerce")
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

# Remove invalid rows
df = df.dropna(subset=["Year", "Month", "Day", "Value"])

# Convert to integer
df["Year"] = df["Year"].astype(int)
df["Month"] = df["Month"].astype(int)
df["Day"] = df["Day"].astype(int)

# Create datetime
df["date"] = pd.to_datetime(df[["Year", "Month", "Day"]])

# Keep only needed columns
df = df[["date", "Value"]].sort_values("date")

# Filter the date range
df = df[(df["date"] >= "1980-01-01") & (df["date"] <= "2025-10-30")]

# ============================
# FIX: Set index before interpolation
# ============================
df = df.set_index("date")
df["Value"] = df["Value"].interpolate(method="time")

# Reset index back for saving
df = df.reset_index()

# Save output
df.to_csv("processed_HKO_GMT_ALL.csv", index=False)

print("Processed CSV saved successfully.")
print(df.head())
print("Total rows:", len(df))
