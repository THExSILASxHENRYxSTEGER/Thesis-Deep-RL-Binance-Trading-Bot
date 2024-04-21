import requests 
import json 
import pandas as pd
import datetime as dt 
import os
from pandas_datareader.data import DataReader
import numpy as np
from copy import deepcopy

###### Globale Variablen #####

Tickers = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT", "LTCUSDT", "BNBUSDT", "DOGEUSDT", "AVAXUSDT"]

COLUMNS = {
    "open_time": 0,
    "open": 1,
    "high": 2,
    "low": 3,
    "close": 4,
    "volume": 5,
    "close_time": 6,
    "quote_volume": 7,
    "count": 8,
    "taker_buy_volume": 9,
    "taker_buy_quote_volume": 10,	
    "ignore": 11
    }

DATASET_PERIODS = {
    #"train_set" : [dt.datetime(2019,9,8,0,0), dt.datetime(2023,9,30,23,55)],   # Training set: 8. September 2019 bis  31. September 2023
    "test_set"  : [dt.datetime(2023,10,1,0,0), dt.datetime(2024, 1,15,23,55)], # Testing set: 1. Juli 2023 bis 30. November 2023
    "valid_set" : [dt.datetime(2024,1,16,0,0), dt.datetime(2024,4,15,23,55)],  # Validation set: 1. Dezember 2023 bis 31. März 2024
}

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"

TREASURY_INTEREST_API_CODES = {
    "1_Year" : "DGS1",   # 1 year Treasury Interest rate
    "5_Year" : "DGS5",   # 5 year Treasury Interest rate
    "10_Year" : "DGS10", # 10 year Treasury Interest rate
}

TREASURY_DATA_SOURCE = "fred"

###### Funktionen #######

def get_n_step_binance_bars(symbol, interval, start_date, end_time, delete_cols=["close_time", "ignore"],limit = "1000"):
    start_date = str(int(start_date.timestamp()*1000))
    end_time = str(int(end_time.timestamp()*1000))
    req_params = {"symbol" : symbol,"interval" : interval,"startTime" : start_date, "endTime" : end_time, "limit" : limit}
    df = pd.DataFrame(json.loads(requests.get(BINANCE_API_URL, params=req_params).text))
    if len(df) == 0:
        return None
    df = df.drop(labels=[COLUMNS[col] for col in delete_cols], axis=1)
    cols = ["open_time"]
    for key in COLUMNS.keys():
        if key not in delete_cols and key != "open_time":
            df[COLUMNS[key]] = df[COLUMNS[key]].astype("float")
            cols.append(key)
    df.columns = cols
    df["open_time"] = [int(x/1000) for x in df["open_time"]]
    return df

def get_binance_data(ticker, start_date=dt.datetime(2019, 8, 9), end_time=dt.datetime.now(), interval="5m", set_type="train_set", delete_cols=["close_time", "ignore"], limit = "1000", csv_path=False):
    df_list = []
    while True:
        new_df = get_n_step_binance_bars(ticker, interval, start_date, end_time, delete_cols=delete_cols, limit=limit)
        if type(new_df) != type(pd.DataFrame()):
            break
        df_list.append(new_df) 
        start_date = dt.datetime.fromtimestamp(max(new_df["open_time"])) + dt.timedelta(0,1)
    df = pd.concat(df_list)
    if type(csv_path) != str:
        return df
    data = df.to_dict("records")
    data = pd.DataFrame(data)
    data.reset_index(inplace=True)
    path = os.path.join(csv_path, f"{ticker}_{interval}_{set_type}.csv")
    data.to_csv(path, index_label="open_time")
    return path

def fill_nans_with_previous_val(series):
    prev_val = np.nan
    for val in series: 
        if not np.isnan(val):
            prev_val = val
            break
    assert prev_val != np.nan, "There are only nan Treasury Bond values"
    corrected_series = [deepcopy(prev_val)]
    for val in series[1:]:
        if np.isnan(val): corrected_series.append(prev_val)
        else:
            corrected_series.append(val)
            prev_val = val
    return corrected_series

def get_treasury_data(start_time, end_time, time_series):
    assert len(time_series) > 0, "There is no price data to get Treasury Bond data for"
    interest_rates_interval = list()
    for timeframe in TREASURY_INTEREST_API_CODES.keys():
        series_code = TREASURY_INTEREST_API_CODES[timeframe]
        treasury_data = DataReader(series_code, TREASURY_DATA_SOURCE, start_time-dt.timedelta(days=5), end_time+dt.timedelta(days=5))
        timestamps = [dt.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S') for x in list(treasury_data.index)]
        interest_rates = list(treasury_data[series_code])
        interest_rates = fill_nans_with_previous_val(interest_rates)
        prev_ts, prev_intrst = timestamps.pop(0), interest_rates.pop(0)
        price_time = dt.datetime.fromtimestamp(time_series.pop(0))
        for ts, intrst in zip(timestamps, interest_rates):
            while prev_ts <= price_time and price_time < ts and time_series:
                interest_rates_interval.append(prev_intrst)
                price_time = dt.datetime.fromtimestamp(time_series.pop(0))
            prev_ts, prev_intrst = ts, intrst

def split_into_data_sets(ticker, interval="5m"):
    for data_set in DATASET_PERIODS.keys():
        start_date, end_date = DATASET_PERIODS[data_set]
        out_path = get_binance_data(ticker, start_date, end_date, interval, data_set, csv_path=os.path.join(os.getcwd(), "Data_Fetcher/Data"))
        time_series = list(pd.read_csv(out_path)["open_time.1"])
        treasury_data = get_treasury_data(start_date, end_date ,time_series) # !!!!! fetch treasury data
        break


#get_binance_data("BTCUSDT", dt.datetime(2024,2,8,12,29,24), dt.datetime.now(), "5m", csv_path=os.path.join(os.getcwd(), "Data"))

#get_treasury_data(dt.datetime(2024,2,8,12,29,24), dt.datetime.now())

split_into_data_sets(Tickers[0])

