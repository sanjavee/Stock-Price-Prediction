import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import streamlit as st

model_cache = {}

def fetch_data(symbol):
    """Fetches historical closing price data for a given stock symbol over the past 5 years.
    
    Retrieves the 'Close' price column from Yahoo Finance for the specified symbol.

    Args:
        symbol (str): The stock ticker symbol to fetch data for.

    Returns:
        pandas.DataFrame: A DataFrame containing the 'Close' prices for the stock.
    """
    stock = yf.Ticker(symbol)
    data = stock.history(period="5y")
    return data[['Close']]

def preprocess_data(df):
    """Adds technical indicators and features to a stock price DataFrame.

    Computes various technical indicators such as moving averages, returns, RSI, Bollinger Bands, momentum, and volatility.

    Args:
        df (pandas.DataFrame): DataFrame containing at least a 'Close' price column.

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns for technical indicators.
    """
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_50'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

    delta = df['Close'].diff(1)
    gain = (delta.where(delta >0,0)).rolling(window=14).mean()
    loss = (-delta.where(delta<0,0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 -(100 / (1 + rs))
    df['Middle_Band'] = df['Close'].rolling(window=20).mean()
    df['Upper_Band'] = df['Middle_Band'] + (df['Close'].rolling(window=20).std() * 2)
    df['Lower_Band'] = df['Middle_Band'] - (df['Close'].rolling(window=20).std() * 2)
    df['Momentum'] = df['Close'] - df['Close'].shift(4)
    df['Volatility'] = df['Close'].rolling(window=21).std()

    df.dropna(inplace=True)
    return df

def normalize_data(df):
    """
    Normalizes the 'Close' column of a DataFrame using MinMax scaling.

    Args:
        df (pandas.DataFrame): Input DataFrame containing a 'Close' column.

    Returns:
        tuple: A tuple containing:
            - scaled_data (numpy.ndarray): The normalized 'Close' column values.
            - scaler (MinMaxScaler): The fitted MinMaxScaler object.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Close']])
    return scaled_data, scaler

def prepare_data(scaled_data, time_steps=60):
    """
    Prepares input and output sequences for time series prediction models.

    Args:
        scaled_data (np.ndarray): The scaled time series data, typically a 2D numpy array.
        time_steps (int, optional): The number of previous time steps to use for each input sequence. Defaults to 60.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - X: 3D numpy array of input sequences with shape (samples, time_steps, features).
            - y: 1D numpy array of target values corresponding to each input sequence.
    """
    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i - time_steps:i])
        y.append(scaled_data[i,0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    """
    Builds and compiles a Sequential LSTM-based neural network model for time series prediction.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).

    Returns:
        keras.models.Sequential: Compiled Keras Sequential model with two LSTM layers and two Dense layers.
    """
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss = 'mean_squared_error')
    return model

st.title('Stock Price Prediction')

stock_list = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
selected_stock = st.selectbox('Select the Stock:', stock_list)

st.write(f"Fetching data for {selected_stock}...")
data = fetch_data(selected_stock)
latest_price = data['Close'].iloc[-1]
st.write(f'Latest Price: ${latest_price:.2f}')

if st.button("Train and Predict"):
    st.write("Please Wait...")

    if selected_stock in model_cache:
        model, scaler = model_cache[selected_stock]
    else:
        data = preprocess_data(data)
        scaled_data, scaler = normalize_data(data)

        X, y = prepare_data(scaled_data)

        x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
        model = build_model((x_train.shape[1], x_train.shape[2]))
        model.fit(x_train, y_train, batch_size=32, epochs=5)

        loss = model.evaluate(x_test, y_test, verbose=0)
        st.write(f"Model Evaluation - MSE: {loss:.4f}")

        model_cache[selected_stock] = (model, scaler)

    st.write("Predicting prices for the next 10 days...")
    predictions = []
    input_sequence = scaled_data[-60:]

    for day in range(10):
        input_sequence = input_sequence.reshape(1, -1, 1)
        predicted_price = model.predict(input_sequence)[0][0]
        predictions.append(predicted_price)

        input_sequence = np.append(input_sequence[0][1:], [[predicted_price]], axis = 0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))[:,0]

    days = pd.date_range(start=pd.Timestamp.now() + pd.DateOffset(1), periods=10).strftime('%Y-%m-%d').tolist()
    prediction_df = pd.DataFrame({
        'Date': days,
        "Predicted Price":predictions

    })

    st.write("Predicted Price:")
    st.table(prediction_df)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prediction_df['Date'],
        y=prediction_df['Predicted Price'],
        mode = 'lines+markers',
        name = 'Predicted Prices'
    ))
    fig.update_layout(
        title=f"10-Days Price Prediction for {selected_stock}",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template = "plotly_dark"
    )
    st.plotly_chart(fig)