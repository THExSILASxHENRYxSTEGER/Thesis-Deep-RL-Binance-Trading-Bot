from baselines import BuyAndHold
from data_interface import Interface
from macd import get_macd_cum_rtrns
from Data_Fetcher.global_variables import SET_TYPE_ENCODING

def compare_baselines(skip_set_types=["train"]): 
    buy_hold = BuyAndHold()
    buy_and_hold_rtrns = buy_hold.get_avg_returns()
    for i, set_type in enumerate(SET_TYPE_ENCODING.keys()):
        if set_type in skip_set_types: continue
        cum_rtrns, time_steps = get_macd_cum_rtrns(set_type=set_type)
        Interface.plot_rtrns([cum_rtrns[1:], buy_and_hold_rtrns[i]], time_steps[1:], "Baseline Average Cumulative Returns", False, ["MACD", "Buy and Hold"])

compare_baselines()