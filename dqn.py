import torch
from data_interface import Interface
from environment import Environment
from RL_utils import DQN_AGENT, load_q_func
from Data_Fetcher.global_variables import DEVICE, DQN_ACTIONS, EPSILON
import numpy as np
import matplotlib.pyplot as plt

plot_cum_rtrns = False

def dqn_cum_rtrns(q_func, set_type="train", interval="30m"):
    intfc = Interface()
    env = Environment(intfc, set_type=set_type, interval=interval) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! if test which is case here normalize with data from train set
    crncy_episodes = env.get_episode_windows()

    n_episodes, n_steps, _, window_size = crncy_episodes.shape

    action_space = len(DQN_ACTIONS)

    agent = DQN_AGENT(EPSILON, action_space, q_func, device=DEVICE, training=False)

    A_ts = list()
    prev_position = torch.zeros(n_episodes)

    for i in range(n_steps):
        windows_t = crncy_episodes[:,i,:,:]
        crncy_actions = list()
        for j, window in enumerate(windows_t):
            crncy_A_t = agent.take_action((window, prev_position[j]), i)
            crncy_actions.append(crncy_A_t)
        A_ts.append(crncy_actions)
        prev_position = torch.tensor(crncy_actions)

    weights = Interface.prtflio_weights_from_actions(np.array(A_ts).T)

    train_data = intfc.get_set_type_dataset(set_type, interval)
    _, spcfc_train_data = intfc.get_overall_data_and_ticker_dicts(train_data)
    prcs = {key:list(spcfc_train_data[key]["open"]) for key in spcfc_train_data.keys()}
    rtrns = np.array([intfc.make_prices_to_returns(prcs[key]) for key in prcs.keys()])

    _, weights_cols = weights.shape
    _, rtrns_cols = rtrns.shape
    dffrce = rtrns_cols-weights_cols

    rtrns = rtrns[:,dffrce+1:]

    cum_rtrns = Interface.avg_weighted_cum_rtrns(weights, rtrns)
    filler = np.zeros(window_size-1)
    cum_rtrns = np.concatenate([filler, cum_rtrns])
    return cum_rtrns

if plot_cum_rtrns:
    q_func = load_q_func("DQN_CNN_8_8_16_2_4_4_1_16_128_2_1")
    cum_rtrns = dqn_cum_rtrns(q_func, set_type="test") 
    plt.plot(range(len(cum_rtrns)), cum_rtrns)
    plt.show()