import pandas as pd
import yfinance as yf
from yahoo_earnings_calendar import YahooEarningsCalendar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import math
import datetime
import json

def get_graph(stock_code):
    ticker = yf.Ticker(stock_code)

    csv_data = ticker.history(period="max")
    csv_data.head()

    # Create the data
    csv_data['Date'] = csv_data.index

    graph_date = list(csv_data['Date'].values.flat)
    graph_close = list(csv_data['Close'].values.flat)

    graph = []

    for i, j in zip(graph_date, graph_close):
        graph.append(
            {
                "tanggal": (pd.to_datetime(str(i)) + datetime.timedelta(days = 1)).strftime('%Y-%m-%d'),
                "harga_penutupan": float(j)
            }
        )

    #graph = [(pd.to_datetime(str(i)) + datetime.timedelta(days = 1)).strftime('%Y-%m-%d') for i in graph]
    #list_actual = [float(i) for i in graph]
    
    return {
        "grafik": graph
    }