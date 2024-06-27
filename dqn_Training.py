from data_interface import Interface
from environment import Environment
from RL_utils import DQN_AGENT, ReplayBuffer, CNN, LSTM, load_q_func
from Data_Fetcher.global_variables import EPSILON, TARGET_UPDATE_FREQUENCY, TRAINING_FREQUENCY, BATCH_SIZE, WARM_START, DQN_ACTIONS, DEVICE, N_EPIODES
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from copy import deepcopy

self_play = True # if true create two agents, one that performs the opposite action to the current agents but both actions come into the replay buffer

for q_func_params in [{"q_func_type":"CNN", "n_episodes":200}, {"q_func_type":"LSTM", "n_episodes":150}]:
    for explore_frac in [0.2, 0.4, 0.6]:
        for gamma in [0.99, 0.75, 0.5]:

            N_EPIODES = q_func_params["n_episodes"]
            EXPLORE_FRAC = explore_frac
            EPSILON = lambda i: 1 - 0.999999 * min(1, i/(N_EPIODES * EXPLORE_FRAC))

            intfc = Interface()
            env = Environment(intfc, interval="30m")

            episodes, episode_len, data_cols, window_len = env.episodes.shape

            action_space = len(DQN_ACTIONS)

            q_func_type = q_func_params["q_func_type"]
            model_q_func_name = None # "DQN_CNN_8_8_16_2_4_4_1_16_128_2_1"

            if model_q_func_name != None:
                q_func = load_q_func(model_q_func_name, eval=False, path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_CNN_8_8_16_2_4_4_1_16_128_2_1/self_play")
            else:
                if q_func_type == "CNN":
                    model_parameters = {"in_chnls":data_cols, "out_chnls":data_cols, "time_series_len":window_len, "final_layer_size":action_space, "n_cnn_layers":4, "kernel_size":4, 
                                        "kernel_div":1, "cnn_intermed_chnls":16, "mlp_intermed_size":128, "n_mlp_layers":2, "punctual_vals":1}
                    cnn_layers, mlp_layers = CNN.create_conv1d_layers(**model_parameters) 
                    q_func = CNN(cnn_layers, mlp_layers)
                else:
                    model_parameters = {"in_sz":data_cols, "h_sz":16, "n_lstm_lyrs":window_len, "final_layer_size":action_space, "n_mlp_lyrs":2, "mlp_intermed_size":128, "punctual_vals":1}
                    q_func = LSTM(**model_parameters)
                str_vals = "_".join([str(param) for param in model_parameters.values()])
                model_q_func_name = f"DQN_{q_func_type}_{str_vals}"

            agent = DQN_AGENT(EPSILON, action_space, q_func, DEVICE, gamma=gamma)
            buffer = ReplayBuffer(int(episodes*episode_len), BATCH_SIZE, DEVICE, action_space)

            sum_rewards, avg_rewards = list(), list()

            n_steps = 0
            for n_episode in range(N_EPIODES):
                S_t = env.reset()
                if self_play:
                    env_2, S_t_2 = deepcopy(env), deepcopy(S_t)
                D_t =  False
                episode_rewards = list()
                while not D_t:                            
                    A_t = agent.select_action(S_t, n_episode)
                    S_prime, R_t, D_t = env.step(A_t)
                    transition = (S_t, A_t, R_t, D_t, S_prime)
                    buffer.add(transition)
                    S_t = S_prime
                    if self_play:
                        A_t_2 = torch.abs(A_t-torch.tensor([1])).item()
                        S_prime_2, R_t_2, D_t_2 = env_2.step(A_t_2)
                        transition_2 = (S_t_2, A_t_2, R_t_2, D_t_2, S_prime_2)
                        buffer.add(transition_2)
                        S_t_2 = S_prime_2            
                    episode_rewards.append(R_t)
                    if n_steps > WARM_START and n_steps % TRAINING_FREQUENCY == 0:
                        b_s, b_a, b_r, b_d, b_s_ = buffer.get_batch()
                        agent.train(b_s, b_a, b_r, b_d, b_s_)
                    if n_steps % TARGET_UPDATE_FREQUENCY == 0:
                        agent.update_target_net()
                    n_steps += 1
                    break
                sum_r = np.sum(episode_rewards)
                sum_rewards.append(sum_r)
                avg_r = np.mean(episode_rewards)
                avg_rewards.append(avg_r)
                print(f"Episode: {n_episode}, Timesteps: {n_steps}, sum reward: {sum_r}, avg reward: {avg_r}")
                break

            model_dir = os.path.join(__file__.replace("/dqn_Training.py", ""), "Models", f"DQN_{q_func_type}_explore_{int(10*explore_frac)}_gamma_{int(100*gamma)}")
            os.mkdir(model_dir)
            
            model_path = os.path.join(model_dir, model_q_func_name)
            torch.save(agent.policy_net.state_dict(), model_path)

            #plt.plot(range(len(sum_rewards)), sum_rewards)
            #plt.xlabel("episode")
            #plt.ylabel("sum episode returns")
            #plt.show()

            sum_rewards_path = os.path.join(model_dir, "sum_rewards")
            torch.save(torch.tensor(sum_rewards), sum_rewards_path)

            #plt.plot(range(len(avg_rewards)), avg_rewards)
            #plt.xlabel("episode")
            #plt.ylabel("avg episode returns")
            #plt.show()

            avg_rewards_path = os.path.join(model_dir, "avg_rewards")
            torch.save(torch.tensor(avg_rewards), avg_rewards_path)