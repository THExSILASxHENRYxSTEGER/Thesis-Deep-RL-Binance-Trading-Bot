from baselines import BuyAndHold
from data_interface import Interface
from macd import get_macd_cum_rtrns
from Data_Fetcher.global_variables import SET_TYPE_ENCODING
from RL_utils import load_q_func
from dqn import dqn_cum_rtrns

def compare_baselines(q_func, interval="30m", skip_set_types=["train"]): 
    buy_hold = BuyAndHold(interval=interval)
    buy_and_hold_rtrns = buy_hold.get_avg_returns()
    for i, set_type in enumerate(SET_TYPE_ENCODING.keys()):
        if set_type in skip_set_types: continue
        macd_cum_rtrns, time_steps = get_macd_cum_rtrns(set_type=set_type, interval=interval)
        dqn_rtrns = dqn_cum_rtrns(q_func, set_type, interval)
        Interface.plot_rtrns([buy_and_hold_rtrns[i], macd_cum_rtrns[1:], dqn_rtrns], time_steps[1:], f"Average Cumulative Returns {set_type} set", False, ["Buy and Hold", "MACD", "DQN CNN"])

q_func = load_q_func("DQN_CNN_8_8_16_2_4_4_1_16_128_2_1")
compare_baselines(q_func, "30m")
