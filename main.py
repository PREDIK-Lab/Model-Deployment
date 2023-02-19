from predict_model import *
from get_graph import *
from flask import Flask, jsonify, request
import yfinance as yf
from yahoo_earnings_calendar import YahooEarningsCalendar
import os
import asyncio
import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1 style='color:green'>Hello World!</h1>"

# @app.route("/bantu")
# def help():
#     return "eid"

@app.route("/info", methods = ['GET'])
def get_info():
    stock_code = 'BBYB.JK'

    info = yf.Ticker(stock_code).info

    return {
        "hasil": info['longBusinessSummary']
    }

@app.route("/grafik", methods = ['GET'])
def get_graph_info():
    stock_code = 'BBYB.JK'

    return get_graph(stock_code)

@app.route("/prediksi", methods = ['GET'])
def predict():
    stock_code = 'BBYB.JK'

    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(predict_concurrently(stock_code))

    return result

async def predict_concurrently(stock_code):
    lstm_prediction = asyncio.create_task(give_lstm_prediction_result(stock_code))
    gru_prediction = asyncio.create_task(give_gru_prediction_result(stock_code))

    lstm_prediction = await lstm_prediction
    gru_prediction = await gru_prediction

    if(request.method == 'GET'):
        data = {
            "success" : True,
            "hasil_lstm" : lstm_prediction,
            "hasil_gru" : gru_prediction,
        }
  
        return jsonify(data)

async def give_lstm_prediction_result(stock_code):
    result = predict_model(stock_code, 'lstm')

    await asyncio.sleep(1)

    return result

async def give_gru_prediction_result(stock_code):
    result = predict_model(stock_code, 'gru')

    await asyncio.sleep(1)

    return result

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000), host="0.0.0.0")
