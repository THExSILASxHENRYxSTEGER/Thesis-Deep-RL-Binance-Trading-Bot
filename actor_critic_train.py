from data_interface import Interface
from environment import Environment
from RL_utils import ACTOR_CRITIC_AGENT, Policy_Value, CNN, load_policy_value_func
from Data_Fetcher.global_variables import DQN_ACTIONS, DEVICE
import os
import torch
import matplotlib.pyplot as plt

intfc = Interface()
env = Environment(intfc, interval="30m")

episodes, episode_len, data_cols, window_len = env.episodes.shape 

action_space = len(DQN_ACTIONS)

q_func_type = "CNN"
actor_critic_func_name = None # "AC_2_100_CNN_8_8_16_100_4_4_1_16_128_2_1"

if actor_critic_func_name == None:
    model_parameters = {"in_chnls":data_cols, "out_chnls":data_cols, "time_series_len":window_len, "final_layer_size":100, "n_cnn_layers":4, 
                        "kernel_size":4, "kernel_div":1, "cnn_intermed_chnls":16, "mlp_intermed_size":128, "n_mlp_layers":2, "punctual_vals":1}
    cnn_layers, mlp_layers = CNN.create_conv1d_layers(**model_parameters) 
    base_model = CNN(cnn_layers, mlp_layers)
    final_layer_size = model_parameters["final_layer_size"]
    policy_value_func = Policy_Value(base_model, final_layer_size, action_space)
    str_vals = "_".join([str(param) for param in model_parameters.values()])
    actor_critic_func_name = f"AC_{action_space}_{final_layer_size}_{q_func_type}_{str_vals}"
else:
    policy_value_func = load_policy_value_func(actor_critic_func_name, eval=False)

agent = ACTOR_CRITIC_AGENT(policy_value_func, action_space, DEVICE)

n_episodes = 500
avg_rewards = list()
sum_rewards = list()

for n_episode in range(n_episodes):
    S_t = env.reset()
    D_t =  False
    ep_rewards = list() 
    while not D_t:
        A_t = agent.select_action(S_t)
        S_t, R_t, D_t = env.step(A_t)
        agent.add_reward(R_t)
    avg_ep_reward = agent.get_avg_reward()
    sum_ep_reward = agent.get_sum_reward()
    avg_rewards.append(avg_ep_reward)
    sum_rewards.append(sum_ep_reward)
    agent.train()
    print('Episode: {}, Avg reward: {:.6f}, Sum reward: {:.3f}'.format(n_episode, avg_ep_reward, sum_ep_reward))

model_path = os.path.join(os.getcwd(), actor_critic_func_name)
torch.save(agent.policy_value_func.state_dict(), model_path)

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