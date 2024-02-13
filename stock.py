import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import datetime

# Function to fetch historical stock data from Yahoo Finance for the last 90 days
def fetch_stock_data(symbol):
    # Fetch historical stock data for the last 90 days
    end_date = datetime.datetime.now().date()
    start_date = end_date - datetime.timedelta(days=90)
    data = yf.download(symbol, start=start_date, end=end_date)
    
    return data

# Function to preprocess data and create features
def preprocess_data(data):
    # Calculate additional features: moving average and RSI
    data['Moving Average'] = calculate_moving_average(data, window=14)
    data['RSI'] = calculate_rsi(data, window=14)

    # Shift 'n' days for prediction
    n = 15  # Predict stock price for the next 15 days
    data['Prediction'] = data['Close'].shift(-n)
    
    # Drop rows with missing values
    data.dropna(inplace=True)
    
    return data

# Function to calculate moving average
def calculate_moving_average(data, window):
    return data['Close'].rolling(window=window).mean()

# Function to calculate RSI
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Function to train the model
def train_model(data):
    # Split data into features (X) and target variable (y)
    X = data.drop(['Prediction'], axis=1)
    y = data['Prediction']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define pipeline with preprocessing steps
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', GradientBoostingRegressor())
    ])

    # Define hyperparameters to tune
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.05, 0.1, 0.2],
        'regressor__max_depth': [3, 4, 5]
    }

    # Perform randomized search for hyperparameter tuning
    search = RandomizedSearchCV(pipeline, param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=5)
    search.fit(X_train, y_train)

    # Get best estimator from randomized search
    best_model = search.best_estimator_

    # Evaluate model
    mse = mean_squared_error(y_test, best_model.predict(X_test))
    print(f'Mean Squared Error: {mse}')

    return best_model

# Function to visualize predictions
def visualize_predictions(model, data):
    # Predict 'n' days into the future
    n = 15  # Predicting for 15 days
    # Select the last 90 days of historical data
    last_90_days = data[-90:]
    # Select the features for prediction (excluding the target variable)
    x_future = last_90_days.drop(['Prediction'], axis=1)
    # Make prediction
    future_predictions = model.predict(x_future)
    
    # Create date range for the next 15 days starting from the current date
    start_date = last_90_days.index[-1] + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=n, freq='B')  # Using business days only
    
    # Plot data and predictions
    plt.figure(figsize=(12, 6))
    
    # Plot the last 90 days of historical data
    plt.plot(last_90_days.index, last_90_days['Close'], label='Historical Stock Price')

    # Plot the predicted stock prices for the next 15 days
    plt.plot(future_dates, future_predictions[-n:], label='Predicted Stock Price')

    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Main function
def main():
    # Fetch stock data for 'AAPL' (Apple Inc.)
    symbol = 'NVDA'
    stock_data = fetch_stock_data(symbol)
    
    # Preprocess data
    preprocessed_data = preprocess_data(stock_data)

    # Train the model
    model = train_model(preprocessed_data)

    # Visualize predictions
    visualize_predictions(model, preprocessed_data)

if __name__ == "__main__":
    main()
