import requests 
import json 
import pandas as pd
import datetime as dt 
import os
from pandas_datareader.data import DataReader
import numpy as np
from copy import deepcopy
from global_variables import *

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

def fill_missing_treasury_days_with_nan(timestamps, interest_rates):
    ts_filled, intrst_filled = list(), list() 
    for i in range(len(timestamps)):
        ts_filled.append(timestamps[i])
        intrst_filled.append(interest_rates[i])
        if i+1 < len(timestamps) and timestamps[i] + dt.timedelta(days=1) != timestamps[i+1]:
            j = 0
            while timestamps[i]+dt.timedelta(days=1+j) != timestamps[i+1]:
                ts_filled.append(timestamps[i]+dt.timedelta(days=1+j))
                intrst_filled.append(np.nan)
                j+=1
    return ts_filled, intrst_filled


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
    intrst_intrvls = list()
    for timeframe in TREASURY_INTEREST_API_CODES.keys():
        series_code = TREASURY_INTEREST_API_CODES[timeframe]
        treasury_data = DataReader(series_code, TREASURY_DATA_SOURCE, start_time-dt.timedelta(days=5), end_time+dt.timedelta(days=5))
        timestamps = [dt.datetime.strptime(str(x), '%Y-%m-%d %H:%M:%S') for x in list(treasury_data.index)]
        interest_rates = list(treasury_data[series_code])
        timestamps, interest_rates =  fill_missing_treasury_days_with_nan(timestamps, interest_rates)
        interest_rates = fill_nans_with_previous_val(interest_rates)
        prev_ts, prev_intrst = timestamps.pop(0), interest_rates.pop(0)
        time_series_cpy = deepcopy(time_series)
        price_time = dt.datetime.fromtimestamp(time_series_cpy.pop(0))
        interest_rate_interval = list()
        for ts, intrst in zip(timestamps, interest_rates):
            while prev_ts <= price_time and price_time < ts and time_series_cpy:
                interest_rate_interval.append(prev_intrst)
                price_time = dt.datetime.fromtimestamp(time_series_cpy.pop(0))
            prev_ts, prev_intrst = ts, intrst
        interest_rate_interval.append(interest_rate_interval[len(interest_rate_interval)-1])
        intrst_intrvls.append(interest_rate_interval)
    return intrst_intrvls

def split_into_data_sets(ticker, interval="5m"):
    for data_set in DATASET_PERIODS.keys():
        start_date, end_date = DATASET_PERIODS[data_set]
        out_path = get_binance_data(ticker, start_date, end_date, interval, data_set, csv_path=os.path.join(os.getcwd(), "Data"))
        ticker_data = pd.read_csv(out_path)
        time_series = list(ticker_data["open_time.1"])
        treasury_data = get_treasury_data(start_date, end_date, deepcopy(time_series)) # fetch treasury data
        tckr_data_updtd = ticker_data.to_dict("list")
        tckr_data_updtd["open_time"] = tckr_data_updtd["open_time.1"]
        tckr_data_updtd.pop("open_time.1")
        tckr_data_updtd.pop("index")
        for i, intrsts in enumerate(TREASURY_INTEREST_API_CODES.keys()):
            tckr_data_updtd[f"{intrsts}_Treasury_Yield"] = treasury_data[i]
        os.remove(out_path)
        pd.DataFrame(tckr_data_updtd).to_csv(out_path)

def collect_all_datasets(): # run this function to get all datasets
    for ticker in TICKERS:
        for freq in DATA_FREQUENCIES:
            split_into_data_sets(ticker, freq)
