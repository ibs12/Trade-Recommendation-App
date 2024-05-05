from data_manager import manage_data
from data_manager import download_stock_data, preprocess_data, save_data

import pandas as pd
from flask import Flask, render_template, request, jsonify
import numpy as np
from dqn_model import DQN
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers, models

app = Flask(__name__)

dqn_model = load_model('best_dqn_reduced_aapl_model-2.h5', compile=False)
dqn_model.compile(optimizer='adam', loss='mse')


@app.route('/')
def home():
    return render_template('index.html')




@app.route('/manage_stock_data', methods=['POST'])
def manage_stock_data():
    symbol = request.form.get('stock_symbol')
    api_key = 'R80Z1WBOPY9NHKN0'  # Replace with your actual API key

    if not symbol:
        return jsonify({'error': 'No stock symbol provided'}), 400

    try:
        manage_data(symbol, api_key)
        return jsonify({'message': f'Data management process completed for {symbol}'}), 200
    except Exception as e:
        app.logger.error(f"Error processing data for {symbol}: {str(e)}")
        return jsonify({'error': str(e)}), 500





    
@app.route('/search_stocks')
def search_stocks():
    search_query = request.args.get('query', '')
    api_key = 'R80Z1WBOPY9NHKN0'
    url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={search_query}&apikey={api_key}'
    response = requests.get(url)
    return jsonify(response.json())



@app.route('/get_stock_data')
def get_stock_data():
    # Get the symbol from query parameters
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({'error': 'No symbol provided'}), 400

    # Construct the file name based on the symbol
    file_name = f'{symbol.upper()}_daily_data.csv'
    path = f"stockData/{file_name}"
    try:
        # Reading the CSV file, assuming the first column contains the dates
        df = pd.read_csv(path, header=0, names=[
                         'Date', 'Open', 'High', 'Low', 'Close', 'Volume','SMA_20','SMA_50'], index_col='Date', parse_dates=True)
        one_year_ago = pd.Timestamp('today') - pd.DateOffset(years=1)

        # Filter data for the last year
        filtered_df = df[df.index > one_year_ago]

        # Ensure data is sorted by date
        filtered_df.sort_values('Date', inplace=True)

        # Convert DataFrame to JSON format expected by Chart.js
        data = [{
            # Ensure date is in a format JavaScript can parse
            'timestamp': index.strftime('%Y-%m-%d'),
            'close': row['Close']
        } for index, row in filtered_df.iterrows()]
        return jsonify(data)
    except FileNotFoundError:
        return jsonify({'error': f'Stock data for {symbol} not found'}), 404


@app.route('/recommendation', methods=['POST'])
def recommendation():
    symbol = request.form['stock_symbol']
    try:
        csv_path = f'stockData/{symbol.upper()}_daily_data.csv'
        data = pd.read_csv(csv_path)
        required_features = ['Open', 'High', 'Low', 'Close']
        last_row = data[required_features].iloc[-1]

        # Normalize features if necessary
        normalized_row = (last_row - last_row.min()) / (last_row.max() - last_row.min())
        last_row_normalized = normalized_row.values.reshape(1, -1)

        prediction = dqn_model.predict(last_row_normalized, verbose=0)
        predicted_action_index = np.argmax(prediction)
        actions = ['Buy', 'Sell', 'Hold']
        recommended_action = actions[predicted_action_index]

        # Ensure that NaN or other problematic values don't break serialization
        safe_prediction = [float(x) if not np.isnan(x) else None for x in prediction.flatten()]
        print(jsonify({'recommendation': recommended_action, 'raw_output': safe_prediction}))
        return jsonify({'recommendation': recommended_action, 'raw_output': safe_prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)




def get_stock_price(symbol, full_data=False):
    # Assuming your CSV file is formatted with columns like 'timestamp', 'open', 'high', 'low', 'close', 'volume'
    df = pd.read_csv(f'/stockData{symbol.upper()}_daily_data.csv')

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Ensure data is sorted by date
    df = df.sort_values(by='timestamp', ascending=True)

    if full_data:
        # Return data suitable for full graphing: timestamps and close prices
        return df[['timestamp', 'close']].to_dict(orient='records')
    else:
        # Return the most recent close price
        latest_close = df.iloc[-1]['close']
        return latest_close


