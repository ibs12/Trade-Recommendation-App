from flask import Flask, render_template, request
import numpy as np
from dqn_model import DQN
import requests 

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommendation', methods=['POST'])
def recommendation():
    symbol = request.form['stock_symbol']
    price = get_stock_price(symbol)
    if price is None:
        return render_template('index.html', error="Failed to retrieve stock price.")

    # Assuming get_features returns an appropriately shaped numpy array for your state
    state = np.array([price])  # Simplified example; you need to define get_features to generate the state
    action = dqn.act(state.reshape(1, -1))  # Ensure the state shape matches the DQN input expectation
    actions = ['Buy', 'Sell', 'Hold']
    recommendation = actions[action]

    return render_template('index.html', symbol=symbol, price=price, recommendation=recommendation)

if __name__ == '__main__':
    app.run(debug=True)

def get_stock_price(symbol):
    API_KEY = 'WRGH4XCNUZA425TB'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={API_KEY}'
    response = requests.get(url)
    data = response.json()

    # Parse the JSON response
    time_series = data.get('Time Series (5min)', {})
    if not time_series:
        return None  # Handle cases where the data is not available

    latest_time = max(time_series.keys())  # Get the most recent time
    latest_data = time_series[latest_time]
    latest_price = latest_data.get('4. close')

    if latest_price:
        return float(latest_price)  # Convert to float if needed
    else:
        return None

def get_features(symbol):
    # Fetch price as part of the features
    price = get_stock_price(symbol)
    # Add more features such as moving averages, etc.
    features = [price]  # This should include other financial indicators
    return np.array(features, dtype=float)