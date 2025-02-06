📈 Crypto Price Prediction Tool
🚀 Overview
This project is a real-time cryptocurrency price prediction tool using LSTM (Long Short-Term Memory) neural networks. It fetches live market data from multiple exchanges via CCXT, preprocesses it, trains an LSTM model, and provides real-time price predictions with alerts for significant drops. The tool also features a GUI-based visualization built with Tkinter and Matplotlib.

🛠️ Features
✅ Real-time data fetching using CCXT
✅ Predicts future crypto prices using LSTM
✅ Supports multiple timeframes (1m, 5m, 15m, 1h, etc.)
✅ Graphical visualization of actual vs. predicted prices
✅ Alert system for significant price drops
✅ Dynamic INR conversion for local currency insights

📌 Requirements
Ensure you have the following installed:
pip install ccxt pandas numpy matplotlib scikit-learn tensorflow

🔧 How to Run
Clone the repository:
git clone https://github.com/Code-With-Shanmukh/CryptoPredictionTool.git  
cd CryptoPredictionTool

Run the script:
python CryptoPredictionTool.py

Enter user inputs when prompted:
Timeframe (e.g., 1m, 5m, 1h, 1d):
Crypto symbol (e.g., BTC, ETH):
Exchange name (e.g., Binance, Bitget, KuCoin, MEXC):

📊 How It Works
Fetches real-time OHLCV data (Open, High, Low, Close, Volume) using CCXT.
Preprocesses data (normalization, feature engineering).
Trains an LSTM model using past 60 timestamps.
Predicts the next closing price every few seconds.
Displays real-time graph updates with actual and predicted prices.

🖥️ GUI Preview
The application launches a Tkinter GUI showing live price updates and predictions:

Blue line → Actual prices
Red line → Predicted prices
Red warning → Significant price drop detected
📌 Future Enhancements
🔹 Multi-crypto portfolio tracking
🔹 Advanced deep learning architectures
🔹 Web-based version using Flask/Django

🤝 Contributing
Pull requests are welcome! If you find an issue, open a GitHub issue.
