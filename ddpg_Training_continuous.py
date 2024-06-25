from data_interface import Interface
from environment import ENVIRONMENT_DDPG
from RL_utils import DDPG_AGENT_WEIGHTING, ReplayBuffer, CNN, LSTM, load_q_func, ACTOR, CRITIC
from Data_Fetcher.global_variables import EPSILON, TRAINING_FREQUENCY, BATCH_SIZE, WARM_START, DQN_ACTIONS, DEVICE, N_EPIODES
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from copy import deepcopy

self_play = True # if true create two agents, one that performs the opposite action to the current agents but both actions come into the replay buffer

intfc = Interface()
env = ENVIRONMENT_DDPG(intfc, interval="30m")#, n_root=4)

episode_len, data_cols, window_len = env.episode.shape 

action_space = env.get_action_space() # for DDPG action space is # of currencies ie a weighting

func_type = "CNN"
model_q_func_name = None # "DQN_CNN_8_8_16_2_4_4_1_16_128_2_1"

if model_q_func_name != None:
    q_func = load_q_func(model_q_func_name, eval=False, path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_CNN_8_8_16_2_4_4_1_16_128_2_1/self_play")
else:
    if func_type == "CNN":
        base_model_parameters = {"in_chnls":data_cols, "out_chnls":data_cols*2, "time_series_len":window_len, "final_layer_size":action_space, "n_cnn_layers":4, "kernel_size":4, 
                            "kernel_div":1, "cnn_intermed_chnls":data_cols*2, "only_cnn":True}
        base_cnn_layers, final_cnn_layer_size = CNN.create_conv1d_layers(**base_model_parameters) 
        base_model = CNN(base_cnn_layers, None)
    else:
        model_parameters = {"in_sz":data_cols, "h_sz":7, "n_lstm_lyrs":window_len, "final_layer_size":action_space, "n_mlp_lyrs":1, "mlp_intermed_size":128, "punctual_vals":1}
        q_func = LSTM(**model_parameters)
    #str_vals = "_".join([str(param) for param in model_parameters.values()])

actor_mlp = nn.Sequential(*[nn.Dropout(), nn.Linear(final_cnn_layer_size+action_space, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.Dropout(), nn.LeakyReLU(), nn.Linear(128, action_space)])
actor = ACTOR(base_model, actor_mlp)

critic_mlp = nn.Sequential(*[nn.Dropout(), nn.Linear(final_cnn_layer_size+action_space*2, 256), nn.LeakyReLU(), nn.Linear(256, 128), nn.Dropout(), nn.LeakyReLU(), nn.Linear(128, action_space)]) # critic gets previous position (previous action) and action of this time step
critic = CRITIC(base_model, critic_mlp)

random_process_params = {"size":action_space, "mu":0, "sigma":1}

agent = DDPG_AGENT_WEIGHTING(actor, critic, EPSILON, DEVICE, random_process_params)
buffer = ReplayBuffer(int(action_space*episode_len), BATCH_SIZE, DEVICE, action_space)

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
            b_s, b_a, b_r, b_d, b_s_ = buffer.get_batch(one_hot_actions=False)
            agent.train(b_s, b_a, b_r, b_d, b_s_) 
        n_steps += 1
    sum_r = np.sum(episode_rewards)
    sum_rewards.append(sum_r)
    avg_r = np.mean(episode_rewards)
    avg_rewards.append(avg_r)
    print(f"Episode: {n_episode}, Timesteps: {n_steps}, sum reward: {sum_r}, avg reward: {avg_r}")

model_path = os.path.join(os.getcwd(), "DDPG_CNN")
torch.save(agent.actor.state_dict(), f"{model_path}_ACTOR")
torch.save(agent.critic.state_dict(), f"{model_path}_CRITIC")


plt.plot(range(len(sum_rewards)), sum_rewards)
plt.xlabel("episode")
plt.ylabel("sum episode returns")
plt.show()

sum_rewards_path = os.path.join(os.getcwd(), "sum_rewards")
torch.save(torch.tensor(sum_rewards), sum_rewards_path)

plt.plot(range(len(avg_rewards)), avg_rewards)
plt.xlabel("episode")
plt.ylabel("avg episode returns")
plt.show()

avg_rewards_path = os.path.join(os.getcwd(), "avg_rewards")
torch.save(torch.tensor(avg_rewards), avg_rewards_path)
