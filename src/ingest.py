import yfinance as yf
import pandas as pd
import os

# Create folders if they don't exist
os.makedirs("data/raw", exist_ok=True)

# Companies to analyze
tickers = ["AAPL", "NVDA", "TSLA"]

all_data = []

for ticker in tickers:
    print(f"Downloading data for {ticker}...")

    stock = yf.Ticker(ticker)

    # Pull 6 months of historical data
    hist = stock.history(period="6mo")

    hist.reset_index(inplace=True)

    hist["Ticker"] = ticker

    # Save individual company file
    hist.to_csv(f"data/raw/{ticker}_stock_data.csv", index=False)

    all_data.append(hist)

# Combine all stock data
combined_df = pd.concat(all_data)

# Save combined dataset
combined_df.to_csv("data/raw/all_stock_data.csv", index=False)

print("Stock data downloaded successfully!")