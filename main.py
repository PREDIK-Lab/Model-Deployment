from predict_model import *
from flask import Flask, jsonify, request
import os

app = Flask(__name__)


@app.route('/')
def index():
    return "<h1 style='color:green'>Hello World!</h1>"

@app.route("/bantu")
def help():
    return "eid"

@app.route("/prediksi", methods = ['GET'])
def make():
    result = predict_model()

    if(request.method == 'GET'):
        data = {
            "success" : True,
            "result" : result,
        }
  
        return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000), host="0.0.0.0")
