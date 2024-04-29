import datetime as dt 

TICKERS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "XRPUSDT", "LTCUSDT", "BNBUSDT", "DOGEUSDT"] #, "AVAXUSDT", "SOLUSDT"]

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

DATA_FREQUENCIES = ["5m","15m","30m","1h"]

DATASET_PERIODS = {
    "train_set" : [dt.datetime(2019,9,8,0,0), dt.datetime(2023,9,30,23,55)],   # Training set: 8. September 2019 bis  31. September 2023
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