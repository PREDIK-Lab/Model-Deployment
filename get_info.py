import pandas as pd
import yfinance as yf
from yahoo_earnings_calendar import YahooEarningsCalendar
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import math
import datetime
import json

def get_info(kode_saham):
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

        response = {
            "success": True,
            "tentang_perusahaan": tentangPerusahaan,
            "sektor": sektor,
            "industri": industri,
            "negara": info.summary_profile[kode_saham]['country'],
            "alamat": info.summary_profile[kode_saham]['address1'] + ", " + info.summary_profile[kode_saham]['address2'] +  ", " + info.summary_profile[kode_saham]['city'] + ", " + info.summary_profile[kode_saham]['country'],
            "jumlah_pegawai": info.summary_profile[kode_saham]['fullTimeEmployees'],
            "tanggal_dividen_terakhir": info.summary_detail[kode_saham]['exDividendDate'], 
        }
    except:
        response = {
            "success": True,
            "tentang_perusahaan": tentangPerusahaan,
            "sektor": sektor,
            "industri": industri,
            "negara": info.summary_profile[kode_saham]['country'],
            "alamat": info.summary_profile[kode_saham]['address1'] + ", " + info.summary_profile[kode_saham]['address2'] +  ", " + info.summary_profile[kode_saham]['city'] + ", " + info.summary_profile[kode_saham]['country'],
            "jumlah_pegawai": info.summary_profile[kode_saham]['fullTimeEmployees'],
            "tanggal_dividen_terakhir": "-", 
        }
    finally:
        return response