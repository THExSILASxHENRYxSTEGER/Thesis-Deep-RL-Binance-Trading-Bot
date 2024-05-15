from data_interface import Interface
from Data_Fetcher.global_variables import *
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np

# This class is the Buy and Hold baseline Implementation

class BuyAndHold():

    def __init__(self, interval="1h") -> None:
        intrfc = Interface()
        subsets = [intrfc.get_set_type_dataset(set_type, interval) for set_type in SET_TYPE_ENCODING.keys()]
        self.timeframes, prcs_ds = list(), list()
        for sbst in subsets:
            gnrl, spcfc = intrfc.get_overall_data_and_ticker_dicts(sbst)
            tmfrm = [datetime.fromtimestamp(stmp) for stmp in gnrl["open_time"]][1:]
            self.timeframes.append(tmfrm)
            prcs_ds.append([list(spcfc[key]["open"]) for key in spcfc.keys()])
        self.rtrns = [np.array([Interface.make_prices_to_returns(prc_srs) for prc_srs in prcs]) for prcs in prcs_ds]
        self.weights = list()
        for rtrns in self.rtrns:
            n, T = rtrns.shape
            weights = np.ones_like(rtrns)/n
            weights = np.concatenate([weights, np.atleast_2d(np.zeros(T))], axis=0)
            first_col = np.zeros(n+1)
            first_col[n] = 1.0
            weights = np.concatenate([np.atleast_2d(first_col).T, weights], axis=1)
            self.weights.append(weights)
        self.avg_rtrns = [Interface.avg_weighted_cum_rtrns(weights, rtrns)[1:] for rtrns, weights in zip(self.rtrns, self.weights)]
        self.cum_rtrns = [Interface.avg_weighted_cum_rtrns(weights, rtrns, only_cumulative=True).T[:,1:] for rtrns, weights in zip(self.rtrns, self.weights)]
   
    def get_avg_returns(self): # returns in this order the train-, test-, validation- set average cumulative returns
        return self.avg_rtrns
    
    def get_individual_cum_returns(self): # returns in this order the train-, test-, validation- set the weighted cumulative returns for all cryptocurrencies
        return self.cum_rtrns

    def print_set_avg_rtrns(self):
        for set_type in SET_TYPE_ENCODING.keys():
            print(f"In the {set_type}-set the overall return is {self.avg_rtrns[SET_TYPE_ENCODING[set_type]][-1]}")

    def plot_rtrns(self, avg=True): 
        rtrns_list = self.avg_rtrns if avg else self.cum_rtrns
        titles = list()
        for set_name in SET_TYPE_ENCODING.keys():
            title = f"Cumulative Returns of {set_name} Set"
            if avg:
                title = f"Average {title}"
            titles.append(f"Buy and Hold {title}")         
        for rtrns, timeline, title in zip(rtrns_list, self.timeframes, titles):
            Interface.plot_rtrns(rtrns, timeline, title, avg)

# This class is the Moving Average Convergence Divergence (MACD) baseline Implementation

class MACD:
    
    def __init__(self, S=12, L=26, p_timescale=63, q_timescale=252, action_scale=0.89) -> None:
        self.S = S
        self.L = L
        self.p_timescale = p_timescale
        self.q_timescale = q_timescale
        self.action_scale = action_scale

    def calculate_macd(self, rtrns:pd.DataFrame):
        m_S, m_L = list(), list()
        for key in rtrns.keys():
            m_S.append(list(rtrns[key].ewm(span=self.S, adjust=False).mean()))
            m_L.append(list(rtrns[key].ewm(span=self.L, adjust=False).mean()))
        m_S, m_L = np.array(m_S), np.array(m_L)
        signal_lines = m_S-m_L 
        q_ts = [MACD.n_day_rolling_std(signal_line, self.p_timescale) for signal_line in signal_lines] 
        macd_ts = [MACD.n_day_rolling_std(q_t, self.q_timescale) for q_t in q_ts]
        self.macd_len = len(macd_ts[0])
        return macd_ts
    
    def calculate_actions(self, macd_ts):
        A_ts = list()
        for macd_t in macd_ts:
            A_t = [x*np.exp(-(x**2/4))/self.action_scale for x in macd_t]
            A_ts.append(A_t)
        return A_ts
    
    def get_rtrns(self, prcs): # since MACD starts at p_timescale + q_timescale istead of index 0 returns need adjustment
        rtrns = [Interface.make_prices_to_returns(prc_srs[len(prc_srs)-self.macd_len-1:]) for prc_srs in prcs]    
        return np.array(rtrns)
        
    @staticmethod
    def n_day_rolling_std(series:iter ,window:int):
        assert len(series) > window, "Window is bigger than the length of the data series"
        q_t = list()
        for j in range(len(series)-window+1): 
            rolling_std = np.std(series[j:j+window])
            q_t.append(series[j+window-1]/rolling_std)
        return q_t
    