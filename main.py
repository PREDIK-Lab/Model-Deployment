from predict_model import *
from flask import Flask, jsonify, request
import os
import asyncio

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1 style='color:green'>Hello World!</h1>"

# @app.route("/bantu")
# def help():
#     return "eid"

@app.route("/prediksi", methods = ['GET'])
def predict():
    asyncio.set_event_loop(asyncio.new_event_loop())
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(predict_concurrently())

    return result

async def predict_concurrently():
    lstm_prediction = asyncio.create_task(give_lstm_prediction_result())
    gru_prediction = asyncio.create_task(give_gru_prediction_result())

    lstm_prediction = await lstm_prediction
    gru_prediction = await gru_prediction

    if(request.method == 'GET'):
        data = {
            "success" : True,
            "lstm_result" : lstm_prediction,
            "gru_result" : gru_prediction,
        }
  
        return jsonify(data)

async def give_lstm_prediction_result():
    result = predict_model()

    await asyncio.sleep(1)

    return result

async def give_gru_prediction_result():
    result = predict_model()

    await asyncio.sleep(1)

    return result

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000), host="0.0.0.0")
