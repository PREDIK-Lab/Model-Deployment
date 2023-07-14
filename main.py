from build_model import *
from predict_model import *
from get_graph import *
from get_info import *
from flask import Flask, jsonify, request
import yfinance as yf
from yahoo_earnings_calendar import YahooEarningsCalendar
from yahooquery import Ticker
from pytanggalmerah import TanggalMerah
# from apscheduler.schedulers.background import BackgroundScheduler, BlockingScheduler
from multiprocessing import Process, Queue
import os
import asyncio
import datetime
import json
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

app = Flask(__name__)

@app.route('/')
def index():    
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
   
    # myFile = drive.CreateFile({'title':'output_sasasasa.xlsx', "parents": [{"id": '1pzschX3uMbxU0lB5WZ6IlEEeAUE8MZ-t'}] })
    # myFile.Upload()
    # file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()

    # for file1 in file_list:
    #     print('title: %s, id: %s' % (file1['title'], file1['id']))

    return 'Hai'

def build():
    result = build_lstm_model("BBYB.JK")
    result = build_gru_model("BBYB.JK")

# scheduler = BackgroundScheduler()
# scheduler.add_job(func=build, trigger="interval", minutes=10)
# scheduler.start()

def build_lstm_prediction(kode_saham):
    result = build_lstm_model(kode_saham)

    return result

def build_gru_prediction(kode_saham):
    result = build_gru_model(kode_saham)

    return result

@app.route("/info", methods = ['GET'])
def get_info_info():
    args = request.args
    kode_saham = args.get("kode_saham", type=str)

    try:
        info = Ticker(kode_saham)

        global tentangPerusahaan
        global sektor
        global industri
        # sektor = "-"
        # industri = "-"

        if kode_saham == "BBYB.JK":
            tentangPerusahaan = "PT Bank Neo Commerce Tbk menyediakan berbagai produk dan layanan perbankan komersial di Indonesia. Perusahaan beroperasi melalui tiga segmen: Pinjaman, Pendanaan, dan Treasury. Perusahaan menawarkan produk-produk pendanaan, seperti giro, tabungan, deposito berjangka, dan on-call deposit; produk manajemen kekayaan, yang meliputi reksa dana dan layanan bank assurance; dan produk keuangan, seperti pinjaman pensiun, pinjaman channeling, pinjaman multiguna, hipotek, pinjaman pribadi, pinjaman modal kerja, pinjaman investasi, pinjaman langsung, dan layanan lainnya. Perusahaan juga menyediakan valuta asing, anjak piutang, kartu kredit, wali amanat, leasing, asuransi, dan layanan penempatan dana. Perusahaan ini sebelumnya bernama PT Bank Yudha Bhakti Tbk dan berganti nama menjadi PT Bank Neo Commerce pada September 2020. PT Bank Neo Commerce Tbk didirikan pada tahun 1989 dan berkantor pusat di Jakarta Selatan, Indonesia."
            sektor = "Layanan Keuangan"
            industri = "Bank Regional"
        elif kode_saham == "ARTO.JK":
            tentangPerusahaan = "PT Bank Jago Tbk menyediakan berbagai produk dan layanan perbankan untuk usaha kecil dan menengah di Indonesia. Perusahaan menerima giro dan tabungan, dan deposito berjangka, serta deposito Mudharabah; dan menawarkan kredit modal kerja, kredit investasi, kredit konsumsi, kredit multiguna, dan pembiayaan modal kerja dengan akad murabahah bil wakalah, serta menyediakan bank garansi. Selain itu juga melayani pembayaran tagihan, pengiriman uang/RTGS/SKN, warkat antar kota, warkat dalam kota layanan kliring, ATM, dan rekening dana nasabah, serta kartu debit. Perusahaan ini sebelumnya bernama PT Bank Artos Indonesia Tbk dan berganti nama menjadi PT Bank Jago Tbk pada Juni 2020. PT Bank Jago Tbk didirikan pada tahun 1992 dan berkantor pusat di Jakarta, Indonesia."
            sektor = "Layanan Keuangan"
            industri = "Bank Regional"
        elif kode_saham == "BBHI.JK":
            tentangPerusahaan = "PT Allo Bank Indonesia Tbk menyediakan berbagai produk dan layanan perbankan di Indonesia. Perusahaan menawarkan tabungan dan giro; deposito berjangka; dan modal kerja, investasi, dan pinjaman konsumen. Perusahaan ini sebelumnya bernama PT Bank Harda Internasional Tbk dan berganti nama menjadi PT Allo Bank Indonesia Tbk pada Juni 2021. Perusahaan ini didirikan pada tahun 1992 dan berkantor pusat di Jakarta Selatan, Indonesia. PT Allo Bank Indonesia Tbk merupakan anak perusahaan dari PT Mega Corpora."
            sektor = "Layanan Keuangan"
            industri = "Bank Regional"

        # info.summary_profile[kode_saham]['longBusinessSummary']
        # info.summary_profile[kode_saham]['sector']
        # info.summary_profile[kode_saham]['industry']

        #return info.summary_profile[kode_saham]#['previousClose']

        # return str(info.summary_profile[kode_saham]['country'])

        response = {
            "success": True,
            "tentang_perusahaan": "Bank",
            "sektor": "Layanan Keuangan",
            "industri": "Bank Regional",
            "negara": "Indonesia", #info.summary_profile[kode_saham]['country'],
            "alamat": "Alamat",
            # "alamat": info.summary_profile[kode_saham]['address1'] + ", " + info.summary_profile[kode_saham]['address2'] +  ", " + info.summary_profile[kode_saham]['city'] + ", " + info.summary_profile[kode_saham]['country'],
            "jumlah_pegawai": 10,
            "tanggal_dividen_terakhir": "-",
            # "jumlah_pegawai": info.summary_profile[kode_saham]['fullTimeEmployees'],
            # "tanggal_dividen_terakhir": info.summary_detail[kode_saham]['exDividendDate'], 
        }

        return jsonify(response)
    except:
        response = {
            "success": True,
            "tentang_perusahaan": "Bank",
            "sektor": "Layanan Keuangan",
            "industri": "Bank Regional",
            "negara": "Indonesia", #info.summary_profile[kode_saham]['country'],
            "alamat": "Alamat",
            # "alamat": info.summary_profile[kode_saham]['address1'] + ", " + info.summary_profile[kode_saham]['address2'] +  ", " + info.summary_profile[kode_saham]['city'] + ", " + info.summary_profile[kode_saham]['country'],
            "jumlah_pegawai": 10,
            "tanggal_dividen_terakhir": "-",
            # "jumlah_pegawai": info.summary_profile[kode_saham]['fullTimeEmployees'],
            # "tanggal_dividen_terakhir": info.summary_detail[kode_saham]['exDividendDate'], 
        }

        return jsonify(response)
    # finally:
    #     return jsonify(response)

@app.route("/grafik", methods = ['GET'])
def get_graph_info():
    args = request.args
    kode_saham = args.get("kode_saham", type=str)
    #kode_saham = 'BBYB.JK'

    return jsonify(get_graph(kode_saham))

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
