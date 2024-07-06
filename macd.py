from baselines import MACD
from data_interface import Interface
import pandas as pd
import numpy as np
from Data_Fetcher.global_variables import BINANCE_TRANSACTION_COST, SET_TYPE_ENCODING
from datetime import datetime

plot = False

def get_macd_cum_rtrns(set_type="train", interval="1h"):
    intrfc = Interface()
    train_data = intrfc.get_set_type_dataset(set_type, interval)
    gnrl_train_data, spcfc_train_data = intrfc.get_overall_data_and_ticker_dicts(train_data)
    prcs = {key:list(spcfc_train_data[key]["close"]) for key in spcfc_train_data.keys()}
    train_prcs = pd.DataFrame(prcs)
    macd = MACD()
    macd_ts = macd.calculate_macd(train_prcs)
    A_ts = macd.calculate_actions(macd_ts)
    rtrns = macd.get_rtrns(list(prcs.values()))
    weights = Interface.prtflio_weights_from_actions(np.array(A_ts))
    cum_rtrns = Interface.avg_weighted_cum_rtrns(weights, rtrns, BINANCE_TRANSACTION_COST)
    len_diffrnc = len(gnrl_train_data["open_time"])-len(cum_rtrns)
    return np.concatenate([np.zeros(len_diffrnc), cum_rtrns]), [datetime.fromtimestamp(ts) for ts in gnrl_train_data["open_time"]]

if plot == True:
    for set_type in SET_TYPE_ENCODING.keys():
        cum_rtrns, time_steps = get_macd_cum_rtrns(set_type=set_type)
        Interface.plot_rtrns(cum_rtrns, time_steps, f"Average Cumulative Returns MACD {set_type} set")
