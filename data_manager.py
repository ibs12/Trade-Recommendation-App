import pandas as pd
import numpy as np
import requests


def download_stock_data(symbol, api_key):
    """Fetches stock data from Alpha Vantage and returns a DataFrame."""
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['Time Series (Daily)']).T
    df = df.rename(columns={'1. open': 'Open', '2. high': 'High',
                   '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'})
    df = df.astype(float).iloc[::-1]  # Ensure chronological order
    return df



def preprocess_data(df):
    """Preprocesses the stock data by adding technical indicators and normalizing values."""
    # Calculate moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Normalize price columns
    for col in ['Open', 'High', 'Low', 'Close', 'SMA_20', 'SMA_50']:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    df.dropna(inplace=True)  # Drop rows with NaN values
    return df


def save_data(df, filename):
    """Saves preprocessed data to a CSV file."""
    path = f"stockData/{filename}"
    df.to_csv(
        path)  # Ensure the index column is labeled 'Date'
    print(f"Data saved to {filename}")


# def manage_data(symbols, api_key):
#     """Manages the data downloading and preprocessing for multiple stock symbols."""
#     for symbol in symbols:
#         print(f"Processing data for {symbol}")
#         raw_data = download_stock_data(symbol, api_key)
#         processed_data = preprocess_data(raw_data)
#         save_data(processed_data, f"{symbol}_daily_data.csv")




def manage_data(symbol, api_key):
    """Manages the data downloading and preprocessing for multiple stock symbols."""
    print(f"Processing data for {symbol}")
    raw_data = download_stock_data(symbol, api_key)
    processed_data = preprocess_data(raw_data)
    save_data(processed_data, f"{symbol.upper()}_daily_data.csv")

# if __name__ == "__main__":
#     symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN',
#                'FB', 'TSLA', 'NFLX', 'INTC', 'AMD', 'NVDA']
#     api_key = 'WRGH4XCNUZA425TB'
#     manage_data(symbols, api_key)
