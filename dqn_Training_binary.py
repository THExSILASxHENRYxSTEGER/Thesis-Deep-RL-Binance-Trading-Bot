from data_interface import Interface
from environment import ENVIRONMENT_DQN
from RL_utils import DQN_Q_FUNC, ReplayBuffer_DQN, CNN2, LSTM, load_q_func, DQN_AGENT_2
from Data_Fetcher.global_variables import EPSILON, TRAINING_FREQUENCY, BATCH_SIZE, WARM_START, DEVICE, N_EPIODES
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from copy import deepcopy

intfc = Interface()

for q_func_params in [{"q_func_type":"CNN", "n_episodes":100}, {"q_func_type":"LSTM", "n_episodes":100}]:
    for explore_frac in reversed([0.2, 0.4, 0.6]):
        for gamma in [0.66, 0.33, 0.99]:

            N_EPIODES = q_func_params["n_episodes"]
            EXPLORE_FRAC = explore_frac
            EPSILON = lambda i: 1 - 0.999999 * min(1, i/(N_EPIODES * EXPLORE_FRAC))

            env = ENVIRONMENT_DQN(intfc, interval="1h", set_type="train", n_root=4)
            windows_t0 = env.windows[0]
            episode_len = len(env.windows)
            data_cols, window_len = windows_t0[0].shape
            data_cols_gnrl, _ = windows_t0[len(windows_t0)-1].shape
            action_space = env.get_action_space() # for DDPG action space is # of currencies ie a weighting
            func_type = q_func_params["q_func_type"]
            model_q_func_name = None # "DQN_CNN_8_8_16_2_4_4_1_16_128_2_1"
            crncy_encoders = list()
            mlp_in_size = 0
            if model_q_func_name != None:
                q_func = load_q_func(model_q_func_name, eval=False, path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_CNN_8_8_16_2_4_4_1_16_128_2_1/self_play")
            else:
                if func_type == "CNN":
                    for window in windows_t0:
                        in_chnls, _ = window.shape
                        model_parameters = {"in_chnls":in_chnls, "out_chnls":1, "out_sz":window_len, "n_cnn_layers":2, "kernel_size":3, "kernel_div":1, "cnn_intermed_chnls":2}
                        cnn_layers, out_size  = CNN2.create_conv1d_layers(**model_parameters)
                        q_func = CNN2(cnn_layers, out_size)
                        crncy_encoders.append(q_func)
                else:
                    hidden_size = 8
                    for _ in windows_t0:
                        model_parameters = {"in_sz":data_cols, "h_sz":hidden_size, "n_lstm_lyrs":window_len}
                        lstm_layers = LSTM(**model_parameters)
                        crncy_encoders.append(lstm_layers)
                str_vals = "_".join([str(param) for param in model_parameters.values()])
                net_name = f"DQN_{func_type}_" + str_vals

            q_func = DQN_Q_FUNC(crncy_encoders, action_space)

            agent = DQN_AGENT_2(EPSILON, action_space, q_func, DEVICE, gamma=gamma)
            buffer = ReplayBuffer_DQN(int(4*episode_len), BATCH_SIZE, DEVICE, action_space)

            sum_rewards, avg_rewards = list(), list()

            n_steps = 0
            for n_episode in range(N_EPIODES):
                S_t = env.reset()
                D_t =  False
                episode_rewards = list()
                while not D_t:                            
                    A_t = agent.select_action(S_t, n_episode)
                    S_prime, R_t, D_t = env.step(A_t) 
                    transition = (deepcopy(S_t), A_t, R_t, D_t, deepcopy(S_prime)) 
                    buffer.add(transition)
                    episode_rewards.append(np.sum(R_t))
                    S_t = deepcopy(S_prime)        
                    if n_steps > WARM_START and n_steps % TRAINING_FREQUENCY == 0:
                        b_s, b_a, b_r, b_d, b_s_ = buffer.get_batch()
                        agent.train2(b_s, b_a, b_r, b_d, b_s_)
                    n_steps += 1
                sum_r = np.sum(episode_rewards)
                sum_rewards.append(sum_r)
                avg_r = np.mean(episode_rewards)
                avg_rewards.append(avg_r)
                print(f"{int(100*explore_frac)}_{int(100*gamma)}_Episode: {n_episode}, Timesteps: {n_steps}, sum reward: {sum_r}, avg reward: {avg_r}")

            model_path = os.path.join(__file__.replace("dqn_Training_binary.py", "Models"), f"DQN_{func_type}_{int(100*explore_frac)}_{int(100*gamma)}")
            os.mkdir(model_path)
            torch.save(agent.policy_net.state_dict(), f"{model_path}/{net_name}")

            sum_rewards_path = os.path.join(model_path, "sum_rewards")
            torch.save(torch.tensor(sum_rewards), sum_rewards_path)

            avg_rewards_path = os.path.join(model_path, "avg_rewards")
            torch.save(torch.tensor(avg_rewards), avg_rewards_path)
