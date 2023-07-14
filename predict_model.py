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
from pytanggalmerah import TanggalMerah
import pandas as pd
import numpy as np
import math
import datetime
import json
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

def predict_model(kode_saham, algoritma):
    # gauth = GoogleAuth()     

    # gauth.LoadCredentialsFile("credentials.txt")

    # if gauth.credentials is None:
    #     gauth.LocalWebserverAuth()
    # elif gauth.access_token_expired:
    #     gauth.Refresh()
    # else:
    #     gauth.Authorize()

    # gauth.SaveCredentialsFile("credentials.txt")

    # drive = GoogleDrive(gauth)

    tanggal_merah = TanggalMerah(cache_path = None, cache_time = 600) # cache_path = None berarti directory cache automatis

    # Fetch the data
    ticker = yf.Ticker(kode_saham)

    csv_data = ticker.history(period="3mo")
    csv_data.head()
    csv_data['Date'] = csv_data.index

    graph_date = list(csv_data['Date'].values.flat)
    tanggal_merah = TanggalMerah(cache_path = None, cache_time = 600) # cache_path = None berarti directory cache automatis

    last_date = (pd.to_datetime(str(graph_date[-1])) + datetime.timedelta(days = 1))
    last_date = (pd.to_datetime(str(last_date)) + datetime.timedelta(days = 1)) if last_date == datetime.date.today() else last_date
    
    print(graph_date[-1:])

    start_date = (pd.to_datetime(str(graph_date[-1])) + datetime.timedelta(days = 2))
    #start_date = (pd.to_datetime(str(graph_date[-1])) + datetime.timedelta(days = 1))
    current_year = datetime.date.today().strftime("%Y")
    current_month = datetime.date.today().strftime("%m")
    current_day_date = datetime.date.today().strftime("%d")
    current_day = datetime.date.today().strftime("%A")
    
    tanggal_merah.set_date(current_year, current_month, current_day_date)

    start_date = start_date if (start_date == datetime.date.today()) else (pd.to_datetime(str(start_date)) - datetime.timedelta(days = 1))
    start_date = datetime.date.today() if(tanggal_merah.check() or tanggal_merah.is_holiday() or tanggal_merah.is_sunday() or current_day == "Saturday") else start_date

    initial_date = "2017-01-01"
    current_date = "2023-06-01" #last_date.strftime("%Y-%m-%d") #datetime.date.today()
    current_year = last_date.strftime("%Y")
    current_month = last_date.strftime("%m")
    current_day_date = last_date.strftime("%d")

    #return current_year #current_date + datetime.timedelta(days=1)

    ticker = kode_saham
    csv_data = yf.download(ticker, initial_date, last_date)
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

    # model_name = ""
    # file_list = drive.ListFile({'q': "'root/Model' in parents and trashed=false"}).GetList()
    
    # for item in file_list:
    #     if item['title'] == kode_saham.replace('.', '').lower() + '_training_' + algoritma + '_model_' + initial_date + '_' + current_date + '.h5' :
    #         model_name = item['title']
    #         break

    # regressor = keras.models.load_model(model_name)
    regressor = keras.models.load_model('./model/' + kode_saham.replace('.', '').lower() + '_training_' + algoritma + '_model_' + initial_date + '_' + current_date + '.h5')
    #regressor = keras.models.load_model('./bbybjk_training_model.h5')
    
    # Generating the predictions for next 7 days
    predicted_seven = regressor.predict(x_test)

    # Generating the prices in original scale
    predicted_seven = data_scaler.inverse_transform(predicted_seven)

    # print(predicted_seven)
    # print(actual_seven)
    
    rmse = np.sqrt(np.mean(((predicted_seven - actual_seven) ** 2)))

    # Making predictions on test data
    actual_fifteen = full_data[-15:]
    
    # Reshaping the data to (-1, 1) because its a single entry
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
    #list_predict = [float(i) for i in list_predict]

    list_date = []
    list_predict_date = []

    i = 0

    while i < 7:
        start_date = (pd.to_datetime(str(start_date)) + datetime.timedelta(days = 1))
        current_year = start_date.strftime("%Y")
        current_month = start_date.strftime("%m")
        current_day_date = start_date.strftime("%d")
        current_day = start_date.strftime("%A")
        
        tanggal_merah.set_date(current_year, current_month, current_day_date)
        
        if(tanggal_merah.check() or tanggal_merah.is_holiday() or tanggal_merah.is_sunday() or current_day == "Saturday"):
            i += 0
        else:
            list_date.append(start_date.strftime("%Y-%m-%d"))

            i += 1

    # for i in list_predict:
    #     list_predict.append({
    #         "tanggal": "1",
    #         "prediksi_harga_penutupan": float(i)
    #     })    

    for i, j in enumerate(list_predict):
        list_predict_date.append({
            "tanggal": list_date[i],
            "prediksi_harga_penutupan": json.dumps(float(j))
        })    

    datetime.date.today().strftime("%Y-%m-%d")

    # json_str = json.dumps({
    #     "prediksi": list_predict,
    #     "rmse": rmse
    # })

    #return json.loads(json_str)

    return {
        "harga_penutupan_sebelumnya": json.dumps(float(full_data[-2])),
        "harga_penutupan_saat_ini": json.dumps(float(full_data[-1])),
        "prediksi": list_predict_date,
        "rmse": rmse
    }