from data_interface import Interface
from environment import Environment
from dqn_utils import DQN_AGENT, ReplayBuffer, CNN
from Data_Fetcher.global_variables import EPSILON, TARGET_UPDATE_FREQUENCY, TRAINING_FREQUENCY, BATCH_SIZE, WARM_START, DQN_ACTIONS, DEVICE

intfc = Interface()
env = Environment(intfc, interval="30m")

episodes, episode_len, data_cols, window_len = env.episodes.shape 

action_space = len(DQN_ACTIONS)

cnn_layers, mlp_layers = CNN.create_conv1d_layers(data_cols, data_cols, window_len, action_space, n_cnn_layers=4, n_mlp_layers=2) 

network = CNN(cnn_layers, mlp_layers)
agent = DQN_AGENT(EPSILON, action_space, network, DEVICE)
buffer = ReplayBuffer(int(episodes*episode_len/3), BATCH_SIZE, DEVICE, action_space)

episode_loss, episode_reward = list(), list()

n_episodes, D_t = 100, False
n_steps = 0
for i in range(n_episodes):
    S_t = env.reset()
    print(f"Episode: {i}")
    while not D_t:
        A_t = agent.take_action(S_t, n_steps)
        S_prime, R_t, D_t = env.step(A_t)
        transition = (S_t, A_t, R_t, D_t, S_prime)
        buffer.add(transition)
        S_t = S_prime
        if n_steps > WARM_START and n_steps % TRAINING_FREQUENCY == 0:
            b_s, b_a, b_r, b_d, b_s_ = buffer.get_batch()
            agent.train(b_s, b_a, b_r, b_d, b_s_)
            print("trained")
        if n_steps % TARGET_UPDATE_FREQUENCY == 0:
            agent.update_target_net()
        n_steps += 1
