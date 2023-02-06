import pandas as pd
import yfinance as yf
from yahoo_earnings_calendar import YahooEarningsCalendar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import numpy as np
import math
from datetime import datetime

def predict_model():
    ticker = yf.Ticker('BBYB.JK')

    # Fetch the data
    ticker = 'BBYB.JK'
    csv_data = yf.download(ticker, '2017-01-01', '2022-12-31')

    csv_data.head()

    # Create the data
    csv_data['TradeDate'] = csv_data.index
    
    # Plot the stock prices
    csv_data.plot(x = 'TradeDate', y = 'Close', kind = 'line', figsize = (20,6), rot = 20)

    FullData=csv_data[['Close']].values
    print(FullData[0:5])
    
    # Feature Scaling for fast training of neural networks
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    # Choosing between Standardization or normalization
    sc = MinMaxScaler()
    
    DataScaler = sc.fit(FullData)
    X = DataScaler.transform(FullData)
    
    print('### After Normalization ###')
    X[0:5]

    # Printing last 10 values of the scaled data which we have created above for the last model
    # Here I am changing the shape of the data to one dimensional array because
    # for Multi step data preparation we need to X input in this fashion
    X = X.reshape(X.shape[0],)
    print('Scaled Prices')
    print(X[-10:])

    # Split into samples
    X_samples = list()
    y_samples = list()

    NumerOfRows = len(X)
    TimeSteps = 15  # next few day's Price Prediction is based on last how many past day's prices
    FutureTimeSteps = 7 # How many days in future you want to predict the prices
    
    # Iterate thru the values to create combinations
    for i in range(TimeSteps , NumerOfRows - FutureTimeSteps , 1):
        x_sample = X[i-TimeSteps:i]
        y_sample = X[i:i+FutureTimeSteps]
        X_samples.append(x_sample)
        y_samples.append(y_sample)
    
    ################################################
    
    # Reshape the Input as a 3D (samples, Time Steps, Features)
    X_data = np.array(X_samples)
    X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)
    print('### Input Data Shape ###') 
    print(X_data.shape)
    
    # We do not reshape y as a 3D data  as it is supposed to be a single column only
    y_data = np.array(y_samples)
    print('### Output Data Shape ###') 
    print(y_data.shape)

    # Choose the number of testing data records
    TestingRecords = int(len(csv_data) - (len(csv_data) * 80 / 100))

    # Split the data into train and test
    X_train = X_data[:-TestingRecords]
    X_test = X_data[-TestingRecords:]
    y_train = y_data[:-TestingRecords]
    y_test = y_data[-TestingRecords:]

    ############################################

    # Print the shape of training and testing
    print('\n#### Training Data shape ####')
    print(X_train.shape)
    print(y_train.shape)
    print('\n#### Testing Data shape ####')
    print(X_test.shape)
    print(y_test.shape)

    # # Visualizing the input and output being sent to the LSTM model
    # for inp, out in zip(X_train[0:2], y_train[0:2]):
    #     print(inp,'--', out)

    # Based on last 10 days prices we are learning the next 5 days of prices
    for inp, out in zip(X_train[0:2], y_train[0:2]):
        print(inp)
        print('====>')
        print(out)
        print('#'*20)

    # Defining Input shapes for LSTM
    TimeSteps = X_train.shape[1]
    TotalFeatures = X_train.shape[2]
    print("Number of TimeSteps:", TimeSteps)
    print("Number of Features:", TotalFeatures)

    # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM

    # Initialising the RNN
    regressor = Sequential()

    # Adding the First input hidden layer and the LSTM layer
    # return_sequences = True, means the output of every time step to be shared with hidden next layer
    regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences = True))

    # Adding the Second Second hidden layer and the LSTM layer
    regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences = True))

    # Adding the Second Third hidden layer and the LSTM layer
    regressor.add(LSTM(units = 15, activation = 'relu', return_sequences = False ))

    # Adding the output layer
    regressor.add(Dense(units = FutureTimeSteps))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    ##################################################

    import time
    # Measuring the time taken by the model to train
    StartTime = time.time()

    # Fitting the RNN to the Training set
    result = regressor.fit(X_train, y_train, batch_size = 10, epochs = 10, validation_data = [X_test, y_test])

    EndTime = time.time()
    print("## Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ##')

    # Making predictions on test data
    predicted_Price = regressor.predict(X_test)
    predicted_Price = DataScaler.inverse_transform(predicted_Price)
    print(predicted_Price)

    # Getting the original price values for testing data
    orig=y_test
    orig=DataScaler.inverse_transform(y_test)
    print(orig)

    # Accuracy of the predictions
    # print('Accuracy:', 100 - (100*(abs(orig-predicted_Price)/orig)).mean())

    regressor.save('/content/drive/MyDrive/bbybjk_training_model.h5')

    return "Aa"