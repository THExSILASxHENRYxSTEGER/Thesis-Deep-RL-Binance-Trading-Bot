import os
import pandas as pd
from Data_Fetcher.global_variables import *
from itertools import zip_longest
import numpy as np

class Interface:

    def __init__(self, data_dir = os.path.join(__file__.replace("/data_interface.py", ""), "Data_Fetcher", "Data")): # change path for training in kaggle
        self.data_dir = data_dir
        self.csv_files = os.listdir(data_dir)

    def get_specific_dataset(self, ticker, set_type="train_set", interval="5m"):
        data = None
        for file in self.csv_files:
            if ticker in file and set_type in file and interval in file:
                data = pd.read_csv(os.path.join(self.data_dir, file))
        assert data is not None, "The dataset in question does not exist"
        return data
    
    def get_multiple_datasets(self, tickers:list, set_types:list, intervals:list):
        return {f"{ticker}_{interval}_{set_type}" : 
                pd.read_csv(os.path.join(self.data_dir, f"{ticker}_{interval}_{set_type}.csv"))
                for ticker in tickers for set_type in set_types for interval in intervals 
                }

    def get_all_datasets(self):
        return self.get_multiple_datasets(TICKERS,DATASET_PERIODS,DATA_FREQUENCIES)
    
    @staticmethod
    def make_prices_to_returns(price_series:list):
        rtrns, p_0 = list(), price_series[0]
        for i in range(1, len(price_series)):
            rtrns.append((price_series[i]-p_0)/p_0)
            p_0 = price_series[i]
        return rtrns

    @staticmethod
    def avg_series(rtrns:dict):
        columns = reversed(list(zip_longest(*[reversed(row) for row in rtrns], fillvalue=np.nan)))
        avg_rtrn = list()
        for col in np.array([x for x in columns]):
            avg_rtrn.append(np.mean(col[~np.isnan(col)]))
        return np.array(avg_rtrn)



































#    @staticmethod
#    def avg_cumulative_rtrns(rtrns:dict):
#        rtrns_filled = [np.array(x) for x in reversed(list(zip_longest(*[reversed(row) for row in rtrns], fillvalue=None)))]
#        rtrnscumulative = [list() for _ in rtrns_filled[0]]
#        indcs = np.where(rtrns_filled[0]!=None)
#        n_array = np.zeros(len(rtrns))
#        n_array[indcs] = 1/len(indcs[0])
#        for i, col in enumerate(rtrns_filled):
#             if i > 1:
#                new_crncy_indcs, active_currencies = list(), 0
#                for prev_rtrn, current_rtrn in zip(rtrns_filled[i-1], col):
#                    if current_rtrn != None:
#                        active_currencies += 1
#                    if prev_rtrn == None and current_rtrn != None: 
#                        new_crncy_indcs.append(True)
#                    else:
#                        new_crncy_indcs.append(False)
#                if any(new_crncy_indcs):
#                    n_array *= 1/active_currencies
#                    n_array[new_crncy_indcs.index(True)] = 1/active_currencies
#            for j, period_rtrn in enumerate(col):
#                if period_rtrn == None:
#                    rtrnscumulative[j].append(np.nan)
#                else:
#                    n_array[j] *= 1+period_rtrn
#                    rtrnscumulative[j].append(n_array[j])
#                    pass # when new currency comes along needs change ie redistribution of percentages as described below        
#        #avg_rtrn = Interface.avg_series(rtrnscumulative)
#        #return np.array(avg_rtrn)
#
#
##at the beginning of series instead of 1 you have to multiply 1/n where n is number of 
##currencies and if one other currency is added in between do multiply all the others 
##with (n/n+1) and the other one either with (1/n+1) or with the sum of (1/n+1)*(return of currency n)


