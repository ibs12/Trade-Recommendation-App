import pandas as pd
import numpy as np
import requests

class StockTradingEnv:
    def __init__(self, symbol, api_key):
        self.symbol = symbol
        self.api_key = api_key
        self.data = self.download_data()
        self.n_step = len(self.data)
        self.current_step = None
        self.cash_in_hand = None
        self.shares_held = None
        self.current_stock_price = None
        self.total_shares_owned = 0
        self.initial_investment = 10000
        self.action_space = 3  # Buy, Sell, Hold
        self.state_size = 4  # Adjust based on what you decide to track (price, owned shares, etc.)
        self.reset()

    def download_data(self):
        """Fetches and returns stock data in a DataFrame, also saves it to a CSV file."""
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={self.symbol}&outputsize=full&apikey={self.api_key}"
        response = requests.get(url)
        data = response.json()
        df = pd.DataFrame(data['Time Series (Daily)']).T
        df = df.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'})
        df = df.astype(float).iloc[::-1]  # Reverse to ensure chronological order
        
        # # Save DataFrame to CSV
        # filename = f"{self.symbol}_daily_data.csv"
        # df.to_csv(filename)
        # print(f"Data saved to {filename}")
        
        return df

    def reset(self):
        """Reset the state of the environment to an initial state."""
        self.current_step = 0
        self.cash_in_hand = self.initial_investment
        self.shares_held = 0
        self.total_shares_owned = 0
        self.current_stock_price = self.data['Close'].iloc[self.current_step]
        return self._get_state()

    def step(self, action):
        """Execute one time step within the environment."""
        self.current_stock_price = self.data['Close'].iloc[self.current_step]

        prev_val = self._get_val()
        
        if action == 0:  # Buy
            self._buy_stock()
        elif action == 1:  # Sell
            self._sell_stock()
        # else hold, do nothing

        self.current_step += 1
        if self.current_step > len(self.data) - 1:
            self.current_step = 0  # reset to the start
        
        cur_val = self._get_val()
        reward = cur_val - prev_val
        done = self.current_step == len(self.data) - 1
        info = {'current_portfolio_value': cur_val}
        
        return self._get_state(), reward, done, info

    def _get_state(self):
        """Retrieve the current state."""
        return np.array([self.current_stock_price, self.shares_held, self.cash_in_hand, self.total_shares_owned])

    def _get_val(self):
        """Calculate the value of the current portfolio."""
        return self.shares_held * self.current_stock_price + self.cash_in_hand

    def _buy_stock(self):
        """Buy one share of stock if possible."""
        if self.cash_in_hand >= self.current_stock_price:
            self.shares_held += 1
            self.cash_in_hand -= self.current_stock_price
            self.total_shares_owned += 1

    def _sell_stock(self):
        """Sell one share of stock if possible."""
        if self.shares_held > 0:
            self.shares_held -= 1
            self.cash_in_hand += self.current_stock_price
            self.total_shares_owned = max(0, self.total_shares_owned - 1)



if __name__ == '__main__':
    env = StockTradingEnv('AAPL', 'WRGH4XCNUZA425TB')