import os
import pandas as pd
from datetime import datetime
import pandas_datareader.data as web
import pandas_datareader as web2
import pandas_datareader.famafrench as ff
import pandas_datareader.wb as wb
import yfinance as yf
from pandas_datareader.nasdaq_trader import get_nasdaq_symbols

# Set Quandl API Key
os.environ["QUANDL_API_KEY"] = 'Ui8sHsJzf-m8wFJuLRK8'


def get_yahoo_data(symbol, start, end):
    start_date = datetime.strptime(start, '%Y-%m-%d')
    end_date = datetime.strptime(end, '%Y-%m-%d')
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def get_sp500_constituents():
    sp_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    return pd.read_html(sp_url, header=0)[0]

def get_quandl_data(symbol, start):
    return web.DataReader(symbol, 'quandl', start)

def get_fred_data(symbol, start, end):
    return web.DataReader(symbol, 'fred', start, end)

def get_wb_data(indicator, country, start, end):
    return wb.download(indicator=indicator, country=country, start=start, end=end)

def get_oecd_data(symbol, start, end):
    return web.DataReader(symbol, 'oecd', start=start, end=end)

def get_stooq_data(symbol):
    data = web2.DataReader(symbol, 'stooq')
    return data

def get_symbols():
    symbols = get_nasdaq_symbols()
    return symbols

