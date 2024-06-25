from baselines import BuyAndHold
from data_interface import Interface
from macd import get_macd_cum_rtrns
from Data_Fetcher.global_variables import SET_TYPE_ENCODING
from RL_utils import load_q_func, load_policy_value_func
from model_metrics import model_cum_rtrns

def compare_baselines(models:dict, interval="30m", skip_set_types=["train"]): 
    buy_hold = BuyAndHold(interval=interval)
    buy_and_hold_rtrns = buy_hold.get_avg_returns()
    for i, set_type in enumerate(SET_TYPE_ENCODING.keys()):
        if set_type in skip_set_types: continue
        macd_cum_rtrns, time_steps = get_macd_cum_rtrns(set_type=set_type, interval=interval)
        model_rtrns = list()
        for key in models.keys():
            agent_type = key.split("_")[0]
            model_rtrns.append(model_cum_rtrns(models[key], agent_type, set_type, interval))
        Interface.plot_rtrns([buy_and_hold_rtrns[i], macd_cum_rtrns[1:], *model_rtrns], time_steps[1:], f"Average Cumulative Returns {set_type} set", False, ["Buy and Hold", "MACD", *list(models.keys())])

models = {
    "AC_no_self_play" :      load_policy_value_func("AC_2_100_CNN_8_8_16_100_4_4_1_16_128_2_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/AC_2_100_CNN_8_8_16_100_4_4_1_16_128_2_1/no_pretraining"),
    #"DQN_CNN_no_self_play":  load_q_func("DQN_CNN_8_8_16_2_4_4_1_16_128_2_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_CNN_8_8_16_2_4_4_1_16_128_2_1/no_self_play"), 
    "DQN_CNN_self_play":     load_q_func("DQN_CNN_8_8_16_2_4_4_1_16_128_2_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_CNN_8_8_16_2_4_4_1_16_128_2_1/self_play"),
    #"DQN_LSTM_no_self_play": load_q_func("DQN_LSTM_8_32_16_2_1_128_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_LSTM_8_32_16_2_1_128_1/random_actions"),
    #"DQN_LSTM_self_play":    load_q_func("DQN_LSTM_8_7_16_2_1_128_1", path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_LSTM_8_7_16_2_1_128_1"),
    }

compare_baselines(models, "30m")