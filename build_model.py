import pandas as pd
import yfinance as yf
from yahoo_earnings_calendar import YahooEarningsCalendar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pandas as pd
import numpy as np
import math
import datetime
import time

def build_lstm_model():
    ticker = yf.Ticker('BBYB.JK')

    # Fetch the data
    ticker = 'BBYB.JK'
    last_date = datetime.date.today()
    csv_data = yf.download(ticker, '2017-01-01', last_date)

    csv_data.head()

    # Create the data
    csv_data['TradeDate'] = csv_data.index
    
    # Plot the stock prices
    csv_data.plot(x = 'TradeDate', y = 'Close', kind = 'line', figsize = (20,6), rot = 20)

    full_data=csv_data[['Close']].values
    print(full_data[0:5])
        
    # Choosing between Standardization or normalization
    sc = MinMaxScaler()
    
    data_scaler = sc.fit(full_data)
    x = data_scaler.transform(full_data)
    
    print('### After Normalization ###')
    x[0:5]

    # Printing last 10 values of the scaled data which we have created above for the last model
    # Here I am changing the shape of the data to one dimensional array because
    # for Multi step data preparation we need to x input in this fashion
    x = x.reshape(x.shape[0],)
    print('Scaled Prices')
    print(x[-10:])

    # Split into samples
    x_samples = list()
    y_samples = list()

    n_row = len(x)
    last_time_step = 15  # next few day's Price Prediction is based on last how many past day's prices
    future_time_step = 7 # How many days in future you want to predict the prices
    
    # Iterate thru the values to create combinations
    for i in range(last_time_step , n_row - future_time_step , 1):
        x_sample = x[i-last_time_step:i]
        y_sample = x[i:i+future_time_step]
        x_samples.append(x_sample)
        y_samples.append(y_sample)
    
    ################################################
    
    # Reshape the Input as a 3D (samples, Time Steps, Features)
    x_data = np.array(x_samples)
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)
    print('### Input Data Shape ###') 
    print(x_data.shape)
    
    # We do not reshape y as a 3D data  as it is supposed to be a single column only
    y_data = np.array(y_samples)
    print('### Output Data Shape ###') 
    print(y_data.shape)

    # Choose the number of testing data records
    test_record = int(len(csv_data) - (len(csv_data) * 80 / 100))

    # Split the data into train and test
    x_train = x_data[:-test_record]
    x_test = x_data[-test_record:]
    y_train = y_data[:-test_record]
    y_test = y_data[-test_record:]

    ############################################

    # Print the shape of training and testing
    print('\n#### Training Data shape ####')
    print(x_train.shape)
    print(y_train.shape)
    print('\n#### Testing Data shape ####')
    print(x_test.shape)
    print(y_test.shape)

    # # Visualizing the input and output being sent to the LSTM model
    # for inp, out in zip(x_train[0:2], y_train[0:2]):
    #     print(inp,'--', out)

    # Based on last 10 days prices we are learning the next 5 days of prices
    for inp, out in zip(x_train[0:2], y_train[0:2]):
        print(inp)
        print('====>')
        print(out)
        print('#'*20)

    # Defining Input shapes for LSTM
    last_time_step = x_train.shape[1]
    n_feature = x_train.shape[2]
    print("Number of last_time_step:", last_time_step)
    print("Number of Features:", n_feature)

    # Initialising the RNN
    regressor = Sequential()

    # Adding the First input hidden layer and the LSTM layer
    # return_sequences = True, means the output of every time step to be shared with hidden next layer
    regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (last_time_step, n_feature), return_sequences = True))

    # Adding the Second Second hidden layer and the LSTM layer
    regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (last_time_step, n_feature), return_sequences = True))

    # Adding the Second Third hidden layer and the LSTM layer
    regressor.add(LSTM(units = 15, activation = 'relu', return_sequences = False ))

    # Adding the output layer
    regressor.add(Dense(units = future_time_step))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    ##################################################

    # Measuring the time taken by the model to train
    start_time = time.time()

    # Fitting the RNN to the Training set
    result = regressor.fit(x_train, y_train, batch_size = 10, epochs = 10, validation_data = [x_test, y_test])

    end_time = time.time()
    print("## Total Time Taken: ", round((end_time - start_time)/60), 'Minutes ##')

    # Making predictions on test data
    predicted_price = regressor.predict(x_test)
    predicted_price = data_scaler.inverse_transform(predicted_price)
    print(predicted_price)

    # Getting the actual price values for testing data
    actual_value = y_test
    actual_value = data_scaler.inverse_transform(y_test)
    print(actual_value)

    # Accuracy of the predictions
    # print('Accuracy:', 100 - (100*(abs(actual_value-predicted_price)/actual_value)).mean())

    regressor.save('/content/drive/MyDrive/bbybjk_training_lstm_model_' + last_date + '.h5')
    
    return "Success"

def build_gru_model():
    ticker = yf.Ticker('BBYB.JK')

    # Fetch the data
    ticker = 'BBYB.JK'
    last_date = datetime.date.today()
    csv_data = yf.download(ticker, '2017-01-01', last_date)

    csv_data.head()

    # Create the data
    csv_data['TradeDate'] = csv_data.index
    
    # Plot the stock prices
    csv_data.plot(x = 'TradeDate', y = 'Close', kind = 'line', figsize = (20,6), rot = 20)

    full_data=csv_data[['Close']].values
    print(full_data[0:5])
        
    # Choosing between Standardization or normalization
    sc = MinMaxScaler()
    
    data_scaler = sc.fit(full_data)
    x = data_scaler.transform(full_data)
    
    print('### After Normalization ###')
    x[0:5]

    # Printing last 10 values of the scaled data which we have created above for the last model
    # Here I am changing the shape of the data to one dimensional array because
    # for Multi step data preparation we need to x input in this fashion
    x = x.reshape(x.shape[0],)
    print('Scaled Prices')
    print(x[-10:])

    # Split into samples
    x_samples = list()
    y_samples = list()

    n_row = len(x)
    last_time_step = 15  # next few day's Price Prediction is based on last how many past day's prices
    future_time_step = 7 # How many days in future you want to predict the prices
    
    # Iterate thru the values to create combinations
    for i in range(last_time_step , n_row - future_time_step , 1):
        x_sample = x[i-last_time_step:i]
        y_sample = x[i:i+future_time_step]
        x_samples.append(x_sample)
        y_samples.append(y_sample)
    
    ################################################
    
    # Reshape the Input as a 3D (samples, Time Steps, Features)
    x_data = np.array(x_samples)
    x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)
    print('### Input Data Shape ###') 
    print(x_data.shape)
    
    # We do not reshape y as a 3D data  as it is supposed to be a single column only
    y_data = np.array(y_samples)
    print('### Output Data Shape ###') 
    print(y_data.shape)

    # Choose the number of testing data records
    test_record = int(len(csv_data) - (len(csv_data) * 80 / 100))

    # Split the data into train and test
    x_train = x_data[:-test_record]
    x_test = x_data[-test_record:]
    y_train = y_data[:-test_record]
    y_test = y_data[-test_record:]

    ############################################

    # Print the shape of training and testing
    print('\n#### Training Data shape ####')
    print(x_train.shape)
    print(y_train.shape)
    print('\n#### Testing Data shape ####')
    print(x_test.shape)
    print(y_test.shape)

    # # Visualizing the input and output being sent to the GRU model
    # for inp, out in zip(x_train[0:2], y_train[0:2]):
    #     print(inp,'--', out)

    # Based on last 10 days prices we are learning the next 5 days of prices
    for inp, out in zip(x_train[0:2], y_train[0:2]):
        print(inp)
        print('====>')
        print(out)
        print('#'*20)

    # Defining Input shapes for GRU
    last_time_step = x_train.shape[1]
    n_feature = x_train.shape[2]
    print("Number of last_time_step:", last_time_step)
    print("Number of Features:", n_feature)

    # Initialising the RNN
    regressor = Sequential()

    # Adding the First input hidden layer and the LSTM layer
    # return_sequences = True, means the output of every time step to be shared with hidden next layer
    regressor.add(GRU(units = 10, activation = 'relu', input_shape = (last_time_step, n_feature), return_sequences = True))

    # Adding the Second Second hidden layer and the GRU layer
    regressor.add(GRU(units = 5, activation = 'relu', input_shape = (last_time_step, n_feature), return_sequences = True))

    # Adding the Second Third hidden layer and the GRU layer
    regressor.add(GRU(units = 5, activation = 'relu', return_sequences = False))

    # Adding the output layer
    regressor.add(Dense(units = future_time_step))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    ##################################################

    # Measuring the time taken by the model to train
    start_time = time.time()

    # Fitting the RNN to the Training set
    result = regressor.fit(x_train, y_train, batch_size = 10, epochs = 10, validation_data = [x_test, y_test])

    end_time = time.time()
    print("## Total Time Taken: ", round((end_time-start_time)/60), 'Minutes ##')

    # Making predictions on test data
    predicted_price = regressor.predict(x_test)
    predicted_price = data_scaler.inverse_transform(predicted_price)
    print(predicted_price)

    # Getting the actual_valueinal price values for testing data
    actual_value=y_test
    actual_value=data_scaler.inverse_transform(y_test)
    print(actual_value)

    # Accuracy of the predictions
    # print('Accuracy:', 100 - (100*(abs(actual_value-predicted_price)/actual_value)).mean())

    regressor.save('/content/drive/MyDrive/bbybjk_training_gru_model_' + last_date + '.h5')
    
    return "Success"