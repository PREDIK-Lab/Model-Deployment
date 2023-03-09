import pandas as pd
import yfinance as yf
from yahoo_earnings_calendar import YahooEarningsCalendar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import math
import datetime
import json

def predict_model(kode_saham, algoritma):
    # Fetch the data
    ticker = kode_saham

    current_date = datetime.date.today()
    csv_data = yf.download(ticker, '2017-01-01', current_date)
    csv_data.head()

    # Create the data
    csv_data['TradeDate'] = csv_data.index
    
    # Plot the stock prices
    # csv_data.plot(x='TradeDate', y='Close', kind='line', figsize=(20,6), rot=20)

    full_data = csv_data[['Close']].values
    # print(full_data[-15:])
    
    # Choosing between Standardization or normalization
    sc = MinMaxScaler()
    
    data_scaler = sc.fit(full_data)
    x = data_scaler.transform(full_data)

    # Making predictions on test data
    actual_fifteen = full_data[-22:-7]
    actual_seven = full_data[-7:]

    # Reshaping the data to (-1, 1) because its a single entry
    actual_fifteen = actual_fifteen.reshape(-1, 1)
    
    # Scaling the data on the same level on which model was trained
    x_test = data_scaler.transform(actual_fifteen)

    n_sample = 1
    time_step = x_test.shape[0]
    n_feature = x_test.shape[1]    
    # Reshaping the data as 3D input
    x_test = x_test.reshape(n_sample, time_step, n_feature)

    #regressor = keras.models.load_model('./bbybjk_training_' + algoritma + '_model_' + current_date + '.h5')

    regressor = keras.models.load_model('./bbybjk_training_model.h5')
    
    # Generating the predictions for next 7 days
    predicted_seven = regressor.predict(x_test)

    # Generating the prices in original scale
    predicted_seven = data_scaler.inverse_transform(predicted_seven)

    print("kk")
    print(predicted_seven)
    print(actual_seven)
    
    rmse = np.sqrt(np.mean(((predicted_seven - actual_seven) ** 2)))

    # Making predictions on test data
    actual_fifteen = full_data[-15:]
    
    # Reshaping the data to (-1,1 )because its a single entry
    actual_fifteen = actual_fifteen.reshape(-1, 1)
    
    # Scaling the data on the same level on which model was trained
    x_test = data_scaler.transform(actual_fifteen)
    
    n_sample = 1
    time_step = x_test.shape[0]
    n_feature = x_test.shape[1]
    # Reshaping the data as 3D input
    x_test = x_test.reshape(n_sample, time_step, n_feature)
    
    # Generating the predictions for next 7 days
    predicted_seven = regressor.predict(x_test)

    # Generating the prices in original scale
    predicted_seven = data_scaler.inverse_transform(predicted_seven)

    list_actual = list(actual_seven.flat)
    list_predict = list(predicted_seven.flat)

    list_actual = [float(i) for i in list_actual]
    list_predict = [float(i) for i in list_predict]

    # json_str = json.dumps({
    #     "prediksi": list_predict,
    #     "rmse": rmse
    # })

    #return json.loads(json_str)

    return {
        "prediksi": list_predict,
        "rmse": rmse
    }