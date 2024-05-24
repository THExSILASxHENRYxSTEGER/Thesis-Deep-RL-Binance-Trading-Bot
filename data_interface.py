import os
import pandas as pd
from Data_Fetcher.global_variables import *
from itertools import zip_longest
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class Interface:
    
    def __init__(self, data_dir = os.path.abspath("Data_Fetcher/Data")): # change path for training in kaggle
        self.data_dir = data_dir

    def get_dataset(self, interval="5m"):
        try: data = pd.read_csv(os.path.join(self.data_dir, f"{interval}_data.csv"))
        except: print("The dataset in question does not exist")
        return dict(data)
    
    def get_multiple_datasets(self, intervals:list):
        return {f"{intrvl}" : self.get_dataset(intrvl) for intrvl in intervals}
    
    def get_all_datasets(self):
        return self.get_multiple_datasets(DATA_FREQUENCIES)
    
    def get_set_type_dataset(self, set_type="train", interval="5m"):
        ds = pd.DataFrame(self.get_dataset(interval))
        return dict(ds.loc[ds["set_type"] == SET_TYPE_ENCODING[set_type]])

    def get_multiple_specific_datasets(self, set_types:list, intervals:list):
        return {f"{intrvl}_{set_type}_set" : self.get_set_type_dataset(set_type, intrvl) for set_type in set_types for intrvl in intervals}

    def get_all_datasets(self):
        return self.get_multiple_specific_datasets(list(SET_TYPE_ENCODING.keys()), DATA_FREQUENCIES)
    
    def get_overall_data_and_ticker_dicts(self, ds:dict): # ds is a dataset previously gotten through functions above
        gnrl_data, ticker_dicts = dict(), dict()
        gnrl_data["open_time"] = ds["open_time"]
        gnrl_data["set_type"] = ds["set_type"]
        for trsry_code in TREASURY_INTEREST_API_CODES:
            key = f"{trsry_code}_Treasury_Yield"
            gnrl_data[key] = ds[key]
        for tckr in TICKERS:
            ticker_dicts[tckr] = dict()
            for key in ds.keys():
                if tckr in key:
                    ticker_dicts[tckr][key.replace(f"{tckr}_","")] = ds[key]
        return gnrl_data, ticker_dicts

    @staticmethod
    def plot_rtrns(rtrns, timeline, title, cum=True, labels=TICKERS):
        plt.xticks(rotation=45)
        if cum:
            plt.plot(timeline, rtrns)
        else:
            for rtrn, lbl in zip(rtrns, labels):
                plt.plot(timeline, rtrn, label=lbl)
            plt.legend() 
        plt.title(title)
        plt.xlabel("date")
        plt.ylabel("cumulative return")
        plt.show()        
    
    @staticmethod
    def make_prices_to_returns(price_series:list):
        rtrns, p_0 = list(), price_series[0]
        for i in range(1, len(price_series)):
            rtrns.append((price_series[i]-p_0)/p_0)
            p_0 = price_series[i]
        return rtrns
    
    @staticmethod
    def norm_srs(series):
        max_axis = np.max(np.abs(series), axis=1)
        return (series.T/max_axis).T

    @staticmethod
    def avg_series(rtrns:dict):
        columns = reversed(list(zip_longest(*[reversed(row) for row in rtrns], fillvalue=np.nan)))
        avg_rtrn = list()
        for col in np.array([x for x in columns]):
            avg_rtrn.append(np.mean(col[~np.isnan(col)]))
        return np.array(avg_rtrn)
    
    @staticmethod
    def prtflio_weights_from_actions(A_ts:np.array, max_stake=MAX_STAKE, a_threshold=A_THRESHOLD): # get the portfoli weights when the goal is to be as diversified as possible and not have less than n_currencies in any case
        n, _ = A_ts.shape
        w = np.zeros(n+1)
        w[n] = 1.0
        w_s = [w]
        for A_t in A_ts.T:
            w = np.zeros(n+1)
            invst_indcs = np.where(A_t>a_threshold)[0]
            if len(invst_indcs) == 0:    
                w[n] = 1.0
            else:
                pot_weight = 1/len(invst_indcs)
                if pot_weight <= max_stake:
                    w[invst_indcs] = pot_weight
                else:
                    w[invst_indcs] = max_stake
                    w[n] = 1-len(invst_indcs)*max_stake
            w_s.append(w)
        return np.array(w_s).T
    
    @staticmethod
    def avg_weighted_cum_rtrns(weights:np.array, rtrns:np.array, trnsctn_cost=BINANCE_TRANSACTION_COST, only_cumulative=False): 
        n, T = rtrns.shape
        rtrns = np.concatenate((rtrns, np.atleast_2d(np.zeros(T))), axis=0)
        cum_rtrns, prev_ws = [weights[:,0]], weights[:,0]
        weights = weights[:,1:] # the first column has all initial weight in cash and has no return value
        for i in range(T):
            prev_rtrns = deepcopy(cum_rtrns[i])
            crnt_ws = weights[:,i]
            if not np.array_equal(crnt_ws, prev_ws): #  if new weights change do the operation below to change weighting dynamically 
                w_delta = crnt_ws-prev_ws
                neg_w_indcs = np.where(w_delta<0)[0]
                total_delta = np.sum(np.abs(w_delta[neg_w_indcs]))
                delta_prev_rtrns = prev_rtrns[neg_w_indcs]@np.abs(w_delta[neg_w_indcs])#*(1-trnsctn_cost)
                prev_rtrns[neg_w_indcs] *= 1+w_delta[neg_w_indcs]
                pos_w_indcs = np.where(w_delta>0)[0]
                for indx in pos_w_indcs:
                    prev_rtrns[indx] += (w_delta[indx]/total_delta)*delta_prev_rtrns*(1-trnsctn_cost)
            crnt_rtrns = np.ones_like(weights[:,i])+rtrns[:,i]
            cum_rtrns.append(prev_rtrns*crnt_rtrns)
            prev_ws = deepcopy(crnt_ws)
        if only_cumulative:
            return np.array(np.array(cum_rtrns))-1/(n-1)
        cum_rtrns = np.sum(np.array(cum_rtrns), axis=1)-1
        return cum_rtrns