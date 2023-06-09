from build_model import *
from predict_model import *
from get_graph import *
from get_info import *
from flask import Flask, jsonify, request
import yfinance as yf
from yahoo_earnings_calendar import YahooEarningsCalendar
from yahooquery import Ticker
from pytanggalmerah import TanggalMerah
from apscheduler.schedulers.background import BackgroundScheduler, BlockingScheduler
import os
import asyncio
import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return "Hai"

def build():
    result = build_lstm_model("BBYB.JK")
    result = build_gru_model("BBYB.JK")

scheduler = BackgroundScheduler()
scheduler.add_job(func=build, trigger="interval", minutes=10)
scheduler.start()

def build_lstm_prediction(kode_saham):
    result = build_lstm_model(kode_saham)

    return result

def build_gru_prediction(kode_saham):
    result = build_gru_model(kode_saham)

    return result

@app.route("/info", methods = ['GET'])
def get_info_info():
    return get_info(kode_saham)

@app.route("/grafik", methods = ['GET'])
def get_graph_info():
    args = request.args
    kode_saham = args.get("kode_saham", type=str)
    #kode_saham = 'BBYB.JK'

    return get_graph(kode_saham)

@app.route("/prediksi", methods = ['GET'])
def predict():
    args = request.args
    kode_saham = args.get("kode_saham", type=str) #'BBYB.JK'
    #kode_saham = 'BBYB.JK'

    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(predict_concurrently(kode_saham))

    return result

async def predict_concurrently(kode_saham):
    lstm_prediction = asyncio.create_task(give_lstm_prediction_result(kode_saham))
    gru_prediction = asyncio.create_task(give_gru_prediction_result(kode_saham))

    lstm_prediction = await lstm_prediction
    gru_prediction = await gru_prediction

    print("-")

    if(request.method == 'GET'):
        data = {
            "success" : True,
            "hasil_lstm" : lstm_prediction,
            "hasil_gru" : gru_prediction,
        }
  
        return jsonify(data)

async def give_lstm_prediction_result(kode_saham):
    result = predict_model(kode_saham, 'lstm')

    await asyncio.sleep(1)

    return result

async def give_gru_prediction_result(kode_saham):
    result = predict_model(kode_saham, 'gru')

    await asyncio.sleep(1)

    return result

if __name__ == '__main__':
    app.run(debug=False, port=os.getenv("PORT", default=5000), host="0.0.0.0")
