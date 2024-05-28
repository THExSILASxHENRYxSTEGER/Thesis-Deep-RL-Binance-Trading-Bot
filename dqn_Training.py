from data_interface import Interface
from environment import Environment
from RL_utils import DQN_AGENT, ReplayBuffer, CNN, LSTM
from Data_Fetcher.global_variables import EPSILON, TARGET_UPDATE_FREQUENCY, TRAINING_FREQUENCY, BATCH_SIZE, WARM_START, DQN_ACTIONS, DEVICE, N_EPIODES
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

intfc = Interface()
env = Environment(intfc, interval="30m")

episodes, episode_len, data_cols, window_len = env.episodes.shape 

action_space = len(DQN_ACTIONS)

network_type = "LSTM"

if network_type == "CNN":
    model_parameters = {"in_chnls":data_cols, "out_chnls":data_cols, "time_series_len":window_len, "action_space":action_space, "n_cnn_layers":4, "kernel_size":4, 
                        "kernel_div":1, "cnn_intermed_chnls":16, "mlp_intermed_size":128, "n_mlp_layers":2, "punctual_vals":1}
    cnn_layers, mlp_layers = CNN.create_conv1d_layers(**model_parameters) 
    network = CNN(cnn_layers, mlp_layers)
else:
    model_parameters = {"in_sz":data_cols, "h_sz":20, "n_lstm_lyrs":window_len, "action_space":action_space, "n_mlp_lyrs":2, "mlp_intermed_size":128, "punctual_vals":1}
    network = LSTM(**model_parameters)

str_vals = "_".join([str(param) for param in model_parameters.values()])
model_q_func_name = f"DQN_{network_type}_{str_vals}"

agent = DQN_AGENT(EPSILON, action_space, network, DEVICE)
buffer = ReplayBuffer(int(episodes*episode_len/3), BATCH_SIZE, DEVICE, action_space)

sum_rewards, avg_rewards = list(), list()

n_steps = 0
for n_episode in range(N_EPIODES):
    S_t = env.reset()
    D_t =  False
    episode_rewards = list()
    while not D_t:
        A_t = agent.take_action(S_t, n_episode)
        S_prime, R_t, D_t = env.step(A_t)
        transition = (S_t, A_t, R_t, D_t, S_prime)
        buffer.add(transition)
        S_t = S_prime
        episode_rewards.append(R_t)
        if n_steps > WARM_START and n_steps % TRAINING_FREQUENCY == 0:
            b_s, b_a, b_r, b_d, b_s_ = buffer.get_batch()
            agent.train(b_s, b_a, b_r, b_d, b_s_)
        if n_steps % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_net()
        n_steps += 1
    sum_r = np.sum(episode_rewards)
    sum_rewards.append(sum_r)
    avg_r = np.mean(episode_rewards)
    avg_rewards.append(avg_r)
    print(f"Episode: {n_episode}, Timesteps: {n_steps}, sum reward: {sum_r}, avg reward: {avg_r}")

PATH = os.path.join(os.getcwd(), model_q_func_name)
torch.save(agent.policy_net.state_dict(), PATH)


plt.plot(range(len(sum_rewards)), sum_rewards)
plt.xlabel("episode")
plt.ylabel("sum episode returns")
plt.show()

plt.plot(range(len(avg_rewards)), avg_rewards)
plt.xlabel("episode")
plt.ylabel("avg episode returns")
plt.show()
