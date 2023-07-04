import pandas as pd
import yfinance as yf

_all_=['get_ticker','show_ticker_info','get_market_data','get_company_actions','get_financials','get_balance_sheet','get_multiple_tickers']

# Function to get a Ticker object
def get_ticker(symbol):
    """
    Fetch the yfinance Ticker object for the provided symbol.
    
    :param symbol: Stock symbol
    :return: yfinance Ticker object
    """
    return yf.Ticker(symbol)

# Function to show ticker info
def show_ticker_info(ticker):
    """
    Display information for the provided yfinance Ticker object.
    
    :param ticker: yfinance Ticker object
    :return: None
    """
    print(pd.Series(ticker.info).head(20))

# Function to get market data
def get_market_data(ticker, period='5d', interval='1m', start=None, end=None, actions=True, auto_adjust=True):
    """
    Get market data for the provided yfinance Ticker object.
    
    :param ticker: yfinance Ticker object
    :param period: Data period to download 
    :param interval: Data interval 
    :param start: Start date to download
    :param end: End date to download
    :param actions: Download stock actions (dividends, splits)
    :param auto_adjust: Adjust all OHLC 
    :return: DataFrame
    """
    return ticker.history(period=period, interval=interval, start=start, end=end, actions=actions, auto_adjust=auto_adjust)

# Function to get company actions
def get_company_actions(ticker):
    """
    Get company actions for the provided yfinance Ticker object.
    
    :param ticker: yfinance Ticker object
    :return: DataFrame
    """
    return ticker.actions, ticker.dividends, ticker.splits

# Function to get financials
def get_financials(ticker):
    """
    Get financials for the provided yfinance Ticker object.
    
    :param ticker: yfinance Ticker object
    :return: DataFrame
    """
    return ticker.financials, ticker.quarterly_financials

# Function to get balance sheet
def get_balance_sheet(ticker):
    """
    Get balance sheet for the provided yfinance Ticker object.
    
    :param ticker: yfinance Ticker object
    :return: DataFrame
    """
    return ticker.balance_sheet, ticker.quarterly_balance_sheet

# Function to get multiple symbols
def get_multiple_tickers(symbols):
    """
    Fetch yfinance Ticker objects for multiple symbols.
    
    :param symbols: List of stock symbols
    :return: yfinance Tickers object
    """
    return yf.Tickers(symbols)
