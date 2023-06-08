from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from django.http import JsonResponse
import yfinance as yf
import pandas as pd


class PricesView(APIView):
    def get(self, request):
        tickers = request.GET.getlist('tickers')  # Get the tickers from the request parameters
        ohlc = yf.download(tickers, period="max")
        prices = ohlc["Adj Close"].dropna(how="all")
        prices.reset_index(inplace=True)
        prices['Date'] = prices['Date'].dt.strftime('%Y-%m-%d')  # convert the datetime index to string
        return JsonResponse(prices.to_dict(orient='records'), safe=False)


