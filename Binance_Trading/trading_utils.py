import json 
import numpy as np
from binance.client import Client

def get_api_key_secret(api_auth_path=__file__.replace("/trading_utils.py","/api_auth.json")):
    with open(api_auth_path) as f:
        api_auth = json.load(f)
    return api_auth['api_key'], api_auth['api_secret']

def get_client():
    key, secret = get_api_key_secret()
    return Client(key, secret)

def get_prev_30_min_OHLC(client, ticker="BTCUSDT"):
    klines = client.get_historical_klines(ticker, Client.KLINE_INTERVAL_30MINUTE, "1 day ago UTC")
    klines = np.array(klines)
    ohcl = klines[:, 1:6] # get only ohcl prices and nothing else
    return ohcl.T
