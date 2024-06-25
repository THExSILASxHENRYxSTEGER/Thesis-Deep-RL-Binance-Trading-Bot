from data_interface import Interface
from environment import Environment
from RL_utils import DDPG_AGENT, ReplayBuffer, CNN, LSTM, load_q_func
from Data_Fetcher.global_variables import EPSILON, TARGET_UPDATE_FREQUENCY, TRAINING_FREQUENCY, BATCH_SIZE, WARM_START, DQN_ACTIONS, DEVICE, N_EPIODES
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from copy import deepcopy

self_play = True # if true create two agents, one that performs the opposite action to the current agents but both actions come into the replay buffer

intfc = Interface()
env = Environment(intfc, interval="30m")

episodes, episode_len, data_cols, window_len = env.episodes.shape 

action_space = 1 # for DDPG action space is continuous so only one-dimensional

func_type = "CNN"
model_q_func_name = None # "DQN_CNN_8_8_16_2_4_4_1_16_128_2_1"

if model_q_func_name != None:
    q_func = load_q_func(model_q_func_name, eval=False, path="/home/honta/Desktop/Thesis/Thesis-Deep-RL-Binance-Trading-Bot/Models/DQN_CNN_8_8_16_2_4_4_1_16_128_2_1/self_play")
else:
    if func_type == "CNN":
        actor_model_parameters = {"in_chnls":data_cols, "out_chnls":data_cols, "time_series_len":window_len, "final_layer_size":action_space, "n_cnn_layers":4, "kernel_size":4, 
                            "kernel_div":1, "cnn_intermed_chnls":16, "mlp_intermed_size":128, "n_mlp_layers":2, "punctual_vals":1}
        actor_cnn_layers, actor_mlp_layers = CNN.create_conv1d_layers(**actor_model_parameters) 
        actor = CNN(actor_cnn_layers, actor_mlp_layers, final_activation=torch.nn.Sigmoid())

        critic_model_parameters = actor_model_parameters
        critic_model_parameters["punctual_vals"] += 1 # the critic also takes as input the action chosen by the actor
        critic_cnn_layers, critic_mlp_layers = CNN.create_conv1d_layers(**critic_model_parameters)
        critic = CNN(critic_cnn_layers, critic_mlp_layers)
    else:
        model_parameters = {"in_sz":data_cols, "h_sz":7, "n_lstm_lyrs":window_len, "final_layer_size":action_space, "n_mlp_lyrs":1, "mlp_intermed_size":128, "punctual_vals":1}
        q_func = LSTM(**model_parameters)
    #str_vals = "_".join([str(param) for param in model_parameters.values()])

random_process_params = {"size":action_space, "theta":0.10, "mu":0.6, "sigma":0.12}

agent = DDPG_AGENT(actor, critic, EPSILON, DEVICE, random_process_params)
buffer = ReplayBuffer(int(episodes*episode_len), BATCH_SIZE, DEVICE, action_space)

sum_rewards, avg_rewards = list(), list()

n_steps = 0
for n_episode in range(N_EPIODES):
    S_t = env.reset(start_selling=False)
    if self_play:
        env_2, S_t_2 = deepcopy(env), deepcopy(S_t)
    D_t =  False
    episode_rewards = list()
    while not D_t:                            
        A_t = agent.select_action(S_t, n_episode)
        S_prime, R_t, D_t = env.step(1)
        if R_t >= 0:   
            R_t = (R_t)**(1/3.9207)
        else:
            R_t = -(np.abs(R_t)**(1/4))
        R_t *= A_t 
        transition = (deepcopy(S_t), A_t, R_t, D_t, deepcopy(S_prime)) 
        buffer.add(transition)
        episode_rewards.append(deepcopy(R_t))
        S_t = deepcopy(S_prime)
        if self_play:
            A_t_2 = np.array([1])-A_t
            S_prime_2, R_t_2, D_t_2 = env_2.step(1)
            if R_t_2 >= 0:   
                R_t_2 = (R_t_2)**(1/3.9207)
            else:
                R_t_2 = -(np.abs(R_t_2)**(1/4))
            R_t_2 *= A_t_2
            transition_2 = (deepcopy(S_t_2), A_t_2, R_t_2, D_t_2, deepcopy(S_prime_2))
            buffer.add(transition_2)
            S_t_2 = deepcopy(S_prime_2)          
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

"""########################## !!!!!!!!!!!!!!!!!!!!!!!!!!1 New idea :::::

make the output of the actor sigmoidal [0,1] and multiply the output with the reward 
as a weighting how much of the currency should be bought

(additionally during exploraton put multiple predetermined actions and therefore rewards for same input 
in buffer ie [0.0,0.1,0.2,...,0.9,1.0] for same input)



"""