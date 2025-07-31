# Stock Price Prediction

This project is a web application for predicting stock prices using LSTM neural networks. It utilizes historical stock data and technical indicators to forecast future prices for selected stocks.

## Features
- Fetches 5 years of historical data for selected stocks (AAPL, MSFT, GOOGL, TSLA, AMZN) using Yahoo Finance.
- Calculates technical indicators: SMA, EMA, RSI, Bollinger Bands, Momentum, Volatility, Daily and Log Returns.
- Preprocesses and normalizes data for LSTM model training.
- Trains an LSTM-based neural network to predict stock closing prices.
- Predicts prices for the next 10 days and displays results in a table and interactive Plotly chart.
- Built with Streamlit for an interactive web interface.

## How to Run
1. Install required packages:
   ```bash
   pip install yfinance pandas numpy plotly keras scikit-learn streamlit
   ```
2. Start the app:
   ```bash
   streamlit run app.py
   ```
3. Select a stock, click "Train and Predict" to view predictions and charts.

## File Structure
- `app.py`: Main application code.
- `README.md`: Project documentation.

## Requirements
- Python 3.7+
- Internet connection (for fetching stock data)

## License
MIT