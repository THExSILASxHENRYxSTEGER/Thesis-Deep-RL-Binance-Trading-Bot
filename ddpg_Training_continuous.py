from data_interface import Interface
from environment import ENVIRONMENT_DDPG
from RL_utils import DDPG_AGENT, ReplayBuffer_DDPG, CNN2, LSTM, load_q_func, ACTOR, CRITIC
from Data_Fetcher.global_variables import EPSILON, TRAINING_FREQUENCY, BATCH_SIZE, WARM_START, DEVICE, N_EPIODES
from random_processes import OrnsteinUhlenbeckProcess
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from copy import deepcopy

intfc = Interface()

for q_func_params in [{"q_func_type":"CNN", "n_episodes":50}, {"q_func_type":"LSTM", "n_episodes":50}]:
    for explore_frac in reversed([0.2, 0.4, 0.6]):
        for gamma in [0.66, 0.33, 0.99]:

            N_EPIODES = q_func_params["n_episodes"]
            EXPLORE_FRAC = explore_frac
            EPSILON = lambda i: 1 - 0.999999 * min(1, i/(N_EPIODES * EXPLORE_FRAC))

            env = ENVIRONMENT_DDPG(intfc, interval="1h", n_root=3)
            windows_t0 = env.windows[0]
            episode_len = len(env.windows)
            data_cols, window_len = windows_t0[0].shape
            data_cols_gnrl, _ = windows_t0[len(windows_t0)-1].shape
            action_space = env.get_action_space() # for DDPG action space is # of currencies ie a weighting
            func_type = "CNN"
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
                    model_parameters = {"in_sz":data_cols, "h_sz":7, "n_lstm_lyrs":window_len, "final_layer_size":action_space, "n_mlp_lyrs":1, "mlp_intermed_size":128, "punctual_vals":1}
                    q_func = LSTM(**model_parameters)
                #str_vals = "_".join([str(param) for param in model_parameters.values()])

            actor = ACTOR(crncy_encoders, action_space)
            critic = CRITIC(crncy_encoders, action_space)

            random_process = OrnsteinUhlenbeckProcess(theta=0.1, mu=0.0, sigma=.15, size=action_space)

            agent = DDPG_AGENT(actor, critic, EPSILON, DEVICE, random_process, gamma=gamma)
            buffer = ReplayBuffer_DDPG(int(4*episode_len), BATCH_SIZE, DEVICE, action_space)

            sum_rewards, avg_rewards = list(), list()

            n_steps = 0
            for n_episode in range(N_EPIODES):
                S_t = env.reset()
                agent.random_process.reset_states()
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
                        agent.train(b_s, b_a, b_r, b_d, b_s_)
                    n_steps += 1
                sum_r = np.sum(episode_rewards)
                sum_rewards.append(sum_r)
                avg_r = np.mean(episode_rewards)
                avg_rewards.append(avg_r)
                print(f"Episode: {n_episode}, Timesteps: {n_steps}, sum reward: {sum_r}, avg reward: {avg_r}")

            model_path = os.path.join(__file__.replace("ddpg_Training_continuous.py", "Models"), f"DDPG_CNN_{int(100*explore_frac)}_{int(100*gamma)}")
            os.mkdir(model_path)
            torch.save(agent.actor.state_dict(), f"{model_path}/ACTOR")
            torch.save(agent.critic.state_dict(), f"{model_path}/CRITIC")

            sum_rewards_path = os.path.join(model_path, "sum_rewards")
            torch.save(torch.tensor(sum_rewards), sum_rewards_path)

            avg_rewards_path = os.path.join(model_path, "avg_rewards")
            torch.save(torch.tensor(avg_rewards), avg_rewards_path)
