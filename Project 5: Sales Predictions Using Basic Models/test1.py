# ----- Import Libraries -----
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# ----- Load Your Dataset -----
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")

# ----- Rename columns to match forecasting code -----
df = df.rename(columns={
    "ORDERDATE": "Date",
    "PRODUCTLINE": "Product",
    "QUANTITYORDERED": "Quantity",
    "PRICEEACH": "Price"
})

# ----- Convert Date column to datetime -----
df["Date"] = pd.to_datetime(df["Date"])

# ----- Calculate total sales amount -----
df["Sales"] = df["Quantity"] * df["Price"]

# ----- Group by date to get daily total sales -----
daily_sales = df.groupby("Date")["Sales"].sum().reset_index()

# ----- Sort by date -----
daily_sales.sort_values("Date", inplace=True)

# ----- Basic Trend Analysis & Forecast for Next Month -----
daily_sales.set_index("Date", inplace=True)

# Decompose to identify trend
decomposition = seasonal_decompose(daily_sales, model='additive', period=30)
trend = decomposition.trend

print("\n🔹 Trend Summary (last 10 values):")
print(trend.tail(10))

# Forecast next month's sales using last 30-day trend average
last_30_days = trend.dropna().tail(30)
forecast_next_month = last_30_days.mean()
print(f"\n📌 Estimated total sales for next month: {round(forecast_next_month, 2)}")

# ----- Plot Trend -----
plt.figure(figsize=(10,6))
plt.plot(daily_sales.index, daily_sales["Sales"], label="Daily Sales")
plt.plot(trend.index, trend, label="Trend", linewidth=3)
plt.title("Sales Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()

# ----- Suggest Business Adjustments -----
if trend.iloc[-1] > trend.iloc[-2]:
    print("\n Suggestion: Sales are rising — increase inventory or marketing.")
elif trend.iloc[-1] < trend.iloc[-2]:
    print("\n Suggestion: Sales are declining — consider discounts or promotions.")
else:
    print("\n Suggestion: Sales are stable — maintain the current strategy.")
