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

def predict_model(stock_code, algorithm):
    # Fetch the data
    ticker = stock_code

    current_date = datetime.date.today()
    csv_data = yf.download(ticker, '2017-01-01', current_date)
    csv_data.head()

    # Create the data
    csv_data['TradeDate']=csv_data.index
    
    # Plot the stock prices
    # csv_data.plot(x='TradeDate', y='Close', kind='line', figsize=(20,6), rot=20)

    FullData=csv_data[['Close']].values
    # print(FullData[-15:])
    
    # Choosing between Standardization or normalization
    sc=MinMaxScaler()
    
    DataScaler = sc.fit(FullData)
    X=DataScaler.transform(FullData)

    # Making predictions on test data
    Last10DaysPrices = FullData[-22:-7]
    real_seven_days = FullData[-7:]

    # Reshaping the data to (-1, 1) because its a single entry
    Last10DaysPrices=Last10DaysPrices.reshape(-1, 1)
    
    # Scaling the data on the same level on which model was trained
    X_test=DataScaler.transform(Last10DaysPrices)

    NumberofSamples=1
    TimeSteps=X_test.shape[0]
    NumberofFeatures=X_test.shape[1]
    # Reshaping the data as 3D input
    X_test=X_test.reshape(NumberofSamples,TimeSteps,NumberofFeatures)

    #regressor = keras.models.load_model('./bbybjk_training_' + algorithm + '_model_' + current_date + '.h5')

    regressor = keras.models.load_model('./bbybjk_training_model.h5')
    
    # Generating the predictions for next 5 days
    Next5DaysPrice = regressor.predict(X_test)

    # Generating the prices in original scale
    Next5DaysPrice = DataScaler.inverse_transform(Next5DaysPrice)

    print(Next5DaysPrice)
    print(real_seven_days)
    
    rmse = np.sqrt(np.mean(((Next5DaysPrice - real_seven_days) ** 2)))

    # Making predictions on test data
    Last10DaysPrices=FullData[-15:]
    
    # Reshaping the data to (-1,1 )because its a single entry
    Last10DaysPrices=Last10DaysPrices.reshape(-1, 1)
    
    # Scaling the data on the same level on which model was trained
    X_test=DataScaler.transform(Last10DaysPrices)
    
    NumberofSamples=1
    TimeSteps=X_test.shape[0]
    NumberofFeatures=X_test.shape[1]
    # Reshaping the data as 3D input
    X_test=X_test.reshape(NumberofSamples,TimeSteps,NumberofFeatures)
    
    # Generating the predictions for next 5 days
    Next5DaysPrice = regressor.predict(X_test)

    # Generating the prices in original scale
    Next5DaysPrice = DataScaler.inverse_transform(Next5DaysPrice)

    list_actual = list(real_seven_days.flat)
    list_predict = list(Next5DaysPrice.flat)

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