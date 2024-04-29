from data_interface import Interface
from Data_Fetcher.global_variables import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# for the buy and hold strategy the time interval is not important, since the overall return 
# is only dependent on the first and the last price in the entire interval to be measured
# ie. overall return = (p_T-p_0)/p_0 
# where p_0 is the first price and p_T the last price in any price series

def avg_rtrns(rtrns):
    n_currencies = len(rtrns)
    n = np.ones(n_currencies)*1/n_currencies
    rtrns_cumulative = [list() for _ in rtrns]
    for col in np.array(list(rtrns)).T:
        for j, period_rtrn in enumerate(col):
            n[j] *= 1+period_rtrn
            rtrns_cumulative[j].append(n[j])
    rtrns_cumulative = np.sum(np.array(rtrns_cumulative), axis=0)-1 
    return rtrns_cumulative         

intrfc = Interface()

train_sets, test_sets, valid_sets = [intrfc.get_multiple_datasets(TICKERS, [period], ["1h"]) 
                                        for period in DATASET_PERIODS]

train_prcs, test_prcs, valid_prcs = [{key:list(data_set[key]["close"]) for key in data_set.keys()}  
                                        for data_set in [train_sets ,test_sets, valid_sets]]

train_timeframe, test_timeframe, valid_timeframe = [[datetime.fromtimestamp(stmp) for stmp in list(data_set[list(data_set.keys())[0]]["open_time"])][1:]
                                                    for data_set in [train_sets ,test_sets, valid_sets]]

timeframes = [train_timeframe, test_timeframe, valid_timeframe]

avg_train_prcs, avg_test_prcs, avg_valid_prcs = [Interface.avg_series(rtrns.values()) for rtrns in [train_prcs ,test_prcs, valid_prcs]]

train_rtrns, test_rtrns, valid_rtrns = [{key:Interface.make_prices_to_returns(price_series[key]) for key in price_series.keys()}
                                            for price_series in [train_prcs ,test_prcs, valid_prcs]]

avg_train_rtrns, avg_test_rtrns, avg_valid_rtrns = [avg_rtrns(rtrns.values()) for rtrns in [train_rtrns ,test_rtrns, valid_rtrns]]

avg_rtrns = [avg_train_rtrns, avg_test_rtrns, avg_valid_rtrns]

set_names = ["Train", "Test", "Validation"]

print(f"In the train-set the overall return is {avg_train_rtrns[-1]}")
print(f"In the test-set the overall return is {avg_test_rtrns[-1]}")
print(f"In the validation-set the overall return is {avg_valid_rtrns[-1]}")

for rtrns, timeline, set_name in zip(avg_rtrns, timeframes, set_names):
    ax = plt.gca()
    plt.xticks(rotation=45)
    plt.plot(timeline, rtrns, label=set_name)
    plt.title(f"Cumulative Returns of the {set_name} Set")
    plt.legend()
    plt.xlabel("date")
    plt.ylabel("cumulative return")
    plt.show()
