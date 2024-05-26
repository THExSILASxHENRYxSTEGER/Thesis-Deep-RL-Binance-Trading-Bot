import os
import torch
from data_interface import Interface
from environment import Environment
from dqn_utils import DQN_AGENT, CNN, LSTM
from Data_Fetcher.global_variables import DEVICE, DQN_ACTIONS, EPSILON
import numpy as np
import matplotlib.pyplot as plt

def dqn_cum_rtrns(model_path, set_type="train", interval="30m", model="CNN"):
    intfc = Interface()
    env = Environment(intfc, set_type=set_type, interval=interval) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! if test which is case here normalize with data from train set
    crncy_episodes = env.get_episode_windows()

    n_episodes, n_steps, data_cols, window_len = crncy_episodes.shape

    action_space = len(DQN_ACTIONS)

    if model == "CNN":
        cnn_layers, mlp_layers = CNN.create_conv1d_layers(data_cols, data_cols, window_len, action_space, n_cnn_layers=4, n_mlp_layers=2) 
        model = CNN(cnn_layers, mlp_layers)
    else:
        #model = LSTM(...)
        pass 
    
    model_state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(model_state_dict)
    model.eval()

    agent = DQN_AGENT(EPSILON, action_space, model, device=DEVICE, training=False)

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
    return cum_rtrns

model_path = f"{os.path.abspath('')}/Models/qnet"
cum_rtrns = dqn_cum_rtrns(model_path, set_type="test") 
plt.plot(range(len(cum_rtrns)), cum_rtrns)
plt.show()