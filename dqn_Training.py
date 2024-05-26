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

network_type = "CNN"

if network_type == "CNN":
    cnn_layers, mlp_layers = CNN.create_conv1d_layers(data_cols, data_cols, window_len, action_space, n_cnn_layers=4, n_mlp_layers=2) 
    network = CNN(cnn_layers, mlp_layers)
else:
    h_size = 20
    network = LSTM(data_cols, h_size, window_len, action_space)

agent = DQN_AGENT(EPSILON, action_space, network, DEVICE)
buffer = ReplayBuffer(int(episodes*episode_len/3), BATCH_SIZE, DEVICE, action_space)

rewards = list()

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
    r = np.sum(episode_rewards)
    rewards.append(r)
    print(f"Episode: {n_episode}, Timesteps: {n_steps}, avg reward: {r}")

PATH = os.path.join(os.getcwd(), 'qnet')
torch.save(agent.policy_net.state_dict(), PATH)

plt.show(range(len(episode_rewards)), episode_rewards)
plt.xlabel("Episode")
plt.ylabel("avg loss")
plt.show()
