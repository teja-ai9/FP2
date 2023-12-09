#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

def xgboost_forecast(data):
    # Preprocess and set up data for XGBoost
    data['Date'] = pd.to_datetime(data['Date'])
    data = data[['Date', 'EnergyMet']].set_index('Date')

    # Prepare data for training
    short_term_window = 30
    X, y = [], []
    for i in range(short_term_window, len(data)):
        X.append(data.iloc[i - short_term_window:i, 0])
        y.append(data.iloc[i, 0])
    X, y = np.array(X), np.array(y)

    # Train the model
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
    model.fit(X, y)

    # Short-term forecast
    X_future_short_term = data.iloc[-short_term_window:].values.reshape(1, -1)
    short_term_forecast = model.predict(X_future_short_term)

    # Long-term forecast
    long_term_forecast = []
    for _ in range(6):  # 6 months
        long_term_prediction = model.predict(X[-short_term_window:])
        long_term_forecast.append(long_term_prediction[0])
        X = np.roll(X, shift=-1)
        X[-1, -1] = long_term_prediction[0]

    # Convert forecasts to DataFrame
    months = ['January', 'February', 'March', 'April', 'May', 'June']
    short_term_forecast_df = pd.DataFrame({'Month': ['December'], 'EnergyMet': short_term_forecast})
    long_term_forecast_df = pd.DataFrame({'Month': months, 'EnergyMet': long_term_forecast})

    # Calculate performance metrics
    X_test = np.array([data.iloc[i - short_term_window:i, 0] for i in range(len(data) - short_term_window, len(data))])
    y_test = data.iloc[len(data) - short_term_window:, 0].values
    y_pred = model.predict(X_test)

    xgb_metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }

    return short_term_forecast_df, long_term_forecast_df, xgb_metrics




def lstm_forecast(data):
    # Preprocess data
    look_back = 30
    data['Date'] = pd.to_datetime(data['Date'])
    data = data[['Date', 'EnergyMet']].set_index('Date')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values)

    # Prepare the dataset for LSTM
    def create_dataset(dataset, look_back=30):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            X.append(dataset[i:(i + look_back), 0])
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    X, y = create_dataset(scaled_data)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Define and fit the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(30, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)

    # Short-term and long-term forecasts
    last_30_days = scaled_data[-30:]
    short_term_forecast_scaled = model.predict(np.reshape(last_30_days, (1, 30, 1)))
    short_term_forecast = scaler.inverse_transform(short_term_forecast_scaled)[0, 0]

    long_term_forecast = []
    current_batch = last_30_days.copy()
    for i in range(6):  # 6 months
        current_pred_scaled = model.predict(np.reshape(current_batch, (1, 30, 1)))
        current_batch = np.append(current_batch[1:], current_pred_scaled, axis=0)
        long_term_forecast.append(scaler.inverse_transform(current_pred_scaled)[0, 0])

    # Convert forecasts to DataFrame
    months = ['January', 'February', 'March', 'April', 'May', 'June']
    short_term_forecast_df = pd.DataFrame({'Month': ['December'], 'EnergyMet': [short_term_forecast]})
    long_term_forecast_df = pd.DataFrame({'Month': months, 'EnergyMet': long_term_forecast})

    # Calculate performance metrics
    test_size = 60
    X_test, y_test = create_dataset(scaled_data[-(test_size + look_back):], look_back)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_actual = scaler.inverse_transform([y_test])

    lstm_metrics = {
        'MAE': mean_absolute_error(y_actual[0], y_pred[:, 0]),
        'MSE': mean_squared_error(y_actual[0], y_pred[:, 0]),
        'RMSE': np.sqrt(mean_squared_error(y_actual[0], y_pred[:, 0]))
    }

    return short_term_forecast_df, long_term_forecast_df, lstm_metrics





def sarimax_forecast(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

    # Assuming 'EnergyMet' is your target and all other columns are exogenous
    y = data['EnergyMet']
    exog = data.drop('EnergyMet', axis=1)

    # Fill any missing values in exogenous variables
    exog = exog.fillna(method='ffill')  # Forward fill for simplicity

    # Define and fit the SARIMAX model
    # Note: You might need to adjust the order and seasonal_order parameters
    model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()

    # Short-term forecast (1 month ahead)
    # For exogenous variables, use the last known values (assumption)
    exog_forecast = exog.iloc[-1:].values  # Last row extended for 1 month
    short_term_forecast = results.get_forecast(steps=1).predicted_mean

    # Long-term forecast (6 months ahead)
    # Repeating the last known exogenous variables for 6 months (assumption)
    exog_forecast = np.tile(exog.iloc[-1:].values, (6, 1))  # Repeat last row for 6 months
    long_term_forecast = results.get_forecast(steps=6).predicted_mean

    # Convert forecasts to DataFrame
    short_term_forecast_df = pd.DataFrame({'Month': ['December'], 'EnergyMet': short_term_forecast.values})
    long_term_forecast_df = pd.DataFrame({'Month': ['January', 'February', 'March', 'April', 'May', 'June'], 'EnergyMet': long_term_forecast.values})




    n_test = 60
    train = data.iloc[:-n_test]
    test = data.iloc[-n_test:]

    model = SARIMAX(train['EnergyMet'], exog=train.drop(['EnergyMet'], axis=1), order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()

    exog_test = test.drop(['EnergyMet'], axis=1)
    predictions = results.get_forecast(steps=n_test, exog=exog_test).predicted_mean

    mae = mean_absolute_error(test['EnergyMet'], predictions)
    mse = mean_squared_error(test['EnergyMet'], predictions)
    rmse = np.sqrt(mse)

    sarimax_metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse}
    
    return short_term_forecast_df, long_term_forecast_df, sarimax_metrics






def calculate_metrics(xgb_metrics, lstm_metrics, sarimax_metrics):
    metrics_df = pd.DataFrame({
        'XGBoost': [xgb_metrics['MAE'], xgb_metrics['MSE'], xgb_metrics['RMSE']],
        'LSTM': [lstm_metrics['MAE'], lstm_metrics['MSE'], lstm_metrics['RMSE']],
        'SARIMAX': [sarimax_metrics['MAE'], sarimax_metrics['MSE'], sarimax_metrics['RMSE']]
    }, index=['MAE', 'MSE', 'RMSE'])

    return metrics_df.transpose()


uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # XGBoost Forecasts
    st.header("XGBoost Forecasts")
    xgb_short, xgb_long, xgb_metrics = xgboost_forecast(data)
    st.subheader("Short Term Forecast")
    st.write(xgb_short)
    st.subheader("Long Term Forecast")
    st.write(xgb_long)

    # LSTM Forecasts
    st.header("LSTM Forecasts")
    lstm_short, lstm_long, lstm_metrics = lstm_forecast(data)
    st.subheader("Short Term Forecast")
    st.write(lstm_short)
    st.subheader("Long Term Forecast")
    st.write(lstm_long)

    # SARIMAX Forecasts
    st.header("SARIMAX Forecasts")
    sarimax_short, sarimax_long, sarimax_metrics = sarimax_forecast(data)
    st.subheader("Short Term Forecast")
    st.write(sarimax_short)
    st.subheader("Long Term Forecast")
    st.write(sarimax_long)

    # Metrics
    st.header("Model Performance Metrics")
    metrics = calculate_metrics(xgb_metrics, lstm_metrics, sarimax_metrics)
    st.write(metrics)

    # Best Model
    best_model = metrics['RMSE'].idxmin()
    st.success(f"Based on RMSE values, {best_model} is the best performing model.")

    

