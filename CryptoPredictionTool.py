import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tkinter import Tk, BOTH
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Dynamic INR conversion rate (hardcoded as fallback)
USD_TO_INR = 94

# Default to 20s if unknown timeframe
update_interval = 20000 

# Timeframe selection (User-defined)
timeframe = input("Enter timeframe (e.g., 1m, 5m, 15m, 1h, 4h, 1d): ").strip()

# User input for symbol and exchange
symbol = input("Enter the coin symbol (e.g., btc): ").strip().upper() + "/USDT"
exchange_name = input("Enter the exchange name (e.g., binance, bitget, kucoin, mexc): ").strip().lower()

# Fetch real-time crypto data using CCXT
def fetch_data(symbol, exchange_name, limit=500, timeframe=timeframe):
    exchange = getattr(ccxt, exchange_name)()
    markets = exchange.load_markets()
    if symbol not in markets:
        raise ValueError(f"Symbol {symbol} not found on {exchange_name}")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Fetch real-time price directly
def fetch_real_time_price(symbol, exchange_name):
    exchange = getattr(ccxt, exchange_name)()
    ticker = exchange.fetch_ticker(symbol)
    return ticker['last']

# Preprocess the data
def preprocess_data(df):
    df['returns'] = df['close'].pct_change()
    df.dropna(inplace=True)
    
    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df[['close', 'volume', 'returns']])
    return scaled_features, scaler

# Create LSTM model
def create_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Prepare data for training
def prepare_data(scaled_data):
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, :])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y)

# GUI application class
class CryptoPredictionApp:
    def __init__(self, root, symbol, exchange_name):
        self.root = root
        self.symbol = symbol
        self.exchange_name = exchange_name
        self.actual_prices = []
        self.predictions = []
        
        # Initialize Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=BOTH, expand=1)
        
        # Initialize model and fetch initial data
        print("Fetching initial data...")
        data = fetch_data(self.symbol, self.exchange_name)
        self.scaled_data, self.scaler = preprocess_data(data)
        self.model = create_model((60, 3))
        
        print("Training the model...")
        X, y = prepare_data(self.scaled_data)
        self.model.fit(X, y, epochs=50, batch_size=32, verbose=1)
        
        self.actual_prices = list(data['close'][-60:] * USD_TO_INR)
        self.predictions = self.actual_prices[:]
        self.root.after(update_interval, self.update_plot)  # Start the real-time updates

    def update_plot(self):
        # Fetch the latest real-time price in USD
        real_time_price_usd = fetch_real_time_price(self.symbol, self.exchange_name)
        real_time_price_inr = real_time_price_usd * USD_TO_INR

        # Append the latest price to actual prices for plotting
        self.actual_prices.append(real_time_price_inr)

        # Fetch the most recent 60 data points (actual historical data) for the model prediction
        data = fetch_data(self.symbol, self.exchange_name, limit=500, timeframe=timeframe)
        self.scaled_data, _ = preprocess_data(data)
        
        # Prepare the input data for prediction (use last 60 minutes of actual data)
        latest_scaled_data = self.scaled_data[-60:]  # Last 60 data points
        latest_input = np.array([latest_scaled_data])

        # Predict the next closing price
        predicted_price = self.model.predict(latest_input)
        predicted_price = self.scaler.inverse_transform(
            np.column_stack([predicted_price, np.zeros((1, 2))])
        )[0, 0]
        predicted_price_inr = predicted_price * USD_TO_INR

        # Update the predicted prices list
        self.predictions.append(predicted_price_inr)

        # Check for significant drop prediction
        significant_drop = predicted_price_inr < self.actual_prices[-1] * 0.95

        # Clear and redraw the plot
        self.ax.clear()

        # Plot the actual and predicted prices
        self.ax.plot(self.actual_prices[-100:], label='Actual Prices (INR)', color='blue')
        self.ax.plot(self.predictions[-100:], label='Predicted Prices (INR)', color='red')

        # Add text an?ations for real-time and predicted prices
        self.ax.text(0.02, 0.95, f"Real-Time Price: ₹{real_time_price_inr:.4f}", transform=self.ax.transAxes, fontsize=10, color='blue')
        self.ax.text(0.02, 0.90, f"Predicted Price: ₹{predicted_price_inr:.4f}", transform=self.ax.transAxes, fontsize=10, color='red')

        # Highlight significant drop alert
        if significant_drop:
            self.ax.text(0.5, 0.85, "Alert: Significant Drop Predicted!", transform=self.ax.transAxes, fontsize=12, color='red', ha='center')

        # Set the title, labels, and legend
        self.ax.set_title('Real-Time Crypto Price Prediction with Alerts')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Price (INR)')
        self.ax.legend()

        # Redraw the updated plot
        self.canvas.draw()

        # Schedule the next update
        self.root.after(update_interval, self.update_plot)  # Update every user-defined second

# Main function to launch the GUI
def main():
    root = Tk()
    root.title("Real-Time Crypto Prediction with Alerts")
    root.geometry("1200x700")
    app = CryptoPredictionApp(root, symbol, exchange_name)
    root.mainloop()

if __name__ == "__main__":
    main()
