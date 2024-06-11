import numpy as np
import torch
from torch.optim import Adam
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import os
from Data_Fetcher.global_variables import DEVICE, BATCH_SIZE

torch.manual_seed(0)
np.random.seed(0)

############################### Deep-Q-Network Utilities #######################################

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size, device, action_space) -> None:
        self.device = device
        self.buffer_size = buffer_size
        self.buffer = list()
        self.batch_size = batch_size
        self.action_space = action_space

    def add(self, transition):
        idx = np.random.randint(-1, len(self.buffer))
        if len(self.buffer) < self.buffer_size:
            self.buffer.insert(idx, transition)
        else:
            self.buffer[idx] = transition

    def get_batch(self):
        b_wndws, b_pos, b_a, b_r, b_d, b_wndws_, b_pos_ = [list() for _ in range(7)]
        for idx in np.random.randint(0, len(self.buffer), (self.batch_size,)):
            s, a, r, d, s_ = self.buffer[idx]
            wndw, pos = s
            b_wndws.append(wndw)
            b_pos.append(pos)
            b_a.append(a) # implement entire loop only with torch ie optimize
            b_r.append(r)
            b_d.append(d)
            wndw, pos = s_
            b_wndws_.append(wndw)
            b_pos_.append(pos)
        b_wndws = torch.tensor(np.array(b_wndws)).float().to(self.device)
        b_pos = torch.tensor(b_pos).float().to(self.device)
        b_a = torch.eye(self.action_space)[b_a].to(self.device)
        b_r = torch.tensor(b_r).float().to(self.device)
        b_d = torch.tensor(b_d).float().to(self.device)
        b_wndws_ = torch.tensor(np.array(b_wndws_)).float().to(self.device)
        b_pos_ = torch.tensor(b_pos_).float().to(self.device)
        return (b_wndws, b_pos), b_a, b_r, b_d, (b_wndws_, b_pos_)

class DQN_AGENT:

    def __init__(self, eps, action_space, network, device, gamma=0.99, optimizer=Adam, loss=nn.MSELoss, training=True) -> None:
        self.eps = eps
        self.action_space = action_space
        self.device = device
        self.gamma = torch.tensor(gamma).to(self.device)
        self.training = training
        self.policy_net = network.float().to(self.device)
        self.target_net = deepcopy(network).float().to(self.device)
        self.update_target_net()
        if self.training:
            self.optimizer = optimizer(self.policy_net.parameters())
            self.loss = loss()

    def select_action(self, S_t, n_episode):
        if self.eps(n_episode) < np.random.rand() and self.training: #>
            return np.argmax(np.random.rand(self.action_space))
        else:
            S_t = self.state_to_device(S_t)
            A_t = torch.argmax(self.policy_net(S_t))
            torch.cuda.empty_cache()
            return A_t
        
    def state_to_device(self, S_t):
        window, position = S_t
        window = torch.tensor(window).float().to(self.device)
        r, c = window.shape
        window = window.reshape(1, r, c)
        position = torch.tensor(position).float().to(self.device)
        return (window, position)

    def train(self, b_s, b_a, b_r, b_d, b_s_):
        pred = torch.sum(self.policy_net(b_s) * b_a, dim=1)
        target = b_r + (torch.ones_like(b_d)-b_d) * self.gamma * torch.max(self.target_net(b_s_), dim=1)[0]
        loss = self.loss(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()
        return loss 

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

############################### Deep Learning Utilities ###########################################

class CNN(nn.Module):

    def __init__(self, cnn_layers, mlp_layers) -> None:
        super(CNN, self).__init__()
        conv_seq, mlp_seq = list(), list()
        final_layer = mlp_layers.pop(len(mlp_layers)-1)
        for layer in cnn_layers:
            conv_seq.append(nn.Conv1d(**layer))
            conv_seq.append(nn.LeakyReLU())
        for layer in mlp_layers:
            mlp_seq.append(nn.Linear(**layer))
            mlp_seq.append(nn.LeakyReLU())
        mlp_seq.append(nn.Linear(**final_layer))
        self.cnn = nn.Sequential(*conv_seq)
        self.mlp = nn.Sequential(*mlp_seq)

    def forward(self, S_t): # inputs have to be of type float
        window, position = S_t
        cnn_out = self.cnn(window)
        cnn_out = torch.flatten(cnn_out, start_dim=1)
        mlp_in = torch.concat((cnn_out, torch.atleast_2d(position).T), dim=1)
        return self.mlp(mlp_in)

    @staticmethod
    def create_conv1d_layers(in_chnls, out_chnls, time_series_len, final_layer_size, n_cnn_layers=2, kernel_size=4, 
                            kernel_div=1, cnn_intermed_chnls=16, mlp_intermed_size=128, n_mlp_layers=1, punctual_vals=1):
        cnn_layers = list()
        for i in range(n_cnn_layers):
            layer_dict = {
                "in_channels": cnn_intermed_chnls, 
                "out_channels":cnn_intermed_chnls, 
                "kernel_size": kernel_size
            }
            if i == 0:
                layer_dict["in_channels"] = in_chnls
            if i == n_cnn_layers-1:
                layer_dict["out_channels"] = out_chnls
            cnn_layers.append(layer_dict)
            time_series_len = time_series_len-kernel_size+1
            kernel_size = int(kernel_size/kernel_div)    
        mlp_layers = list()
        for i in range(n_mlp_layers):
            layer_dict = {
                "in_features":mlp_intermed_size, 
                "out_features":mlp_intermed_size, 
            }
            if i == 0:
                layer_dict["in_features"] = time_series_len*out_chnls+punctual_vals # change formula to update
            if i == n_mlp_layers-1:
                layer_dict["out_features"] = final_layer_size
            mlp_layers.append(layer_dict)
        return cnn_layers, mlp_layers
    
class LSTM(nn.Module):
    
    def __init__(self, in_sz, h_sz, n_lstm_lyrs, final_layer_size, n_mlp_lyrs=2, mlp_intermed_size=128, punctual_vals=1) -> None:
        super(LSTM, self).__init__()
        self.h_sz = h_sz
        self.lstm_cells = nn.Sequential(*[nn.LSTMCell(in_sz, h_sz) for _ in range(n_lstm_lyrs)])
        mlp_lyrs = list()
        for i in range(n_mlp_lyrs):
            in_features, out_features = mlp_intermed_size, mlp_intermed_size
            if i == 0:
                in_features = 2*h_sz + punctual_vals
            if i == n_mlp_lyrs-1:
                out_features = final_layer_size
                mlp_lyrs.append(nn.Linear(in_features, out_features))
                break
            mlp_lyrs.append(nn.Linear(in_features, out_features))
            mlp_lyrs.append(nn.LeakyReLU())
        self.mlp = nn.Sequential(*mlp_lyrs)

    def forward(self, S_t, device=DEVICE):
        window, position = S_t
        n_batches, in_size, n_cells = window.shape 
        window = window.reshape(n_cells, n_batches, in_size)
        hx, cx = torch.zeros((n_batches, self.h_sz)).to(device), torch.zeros((n_batches, self.h_sz)).to(device)
        for i in range(len(self.lstm_cells)):
            hx, cx = self.lstm_cells[i](window[i], (hx, cx))
        mlp_in = torch.concat((hx, cx, torch.atleast_2d(position).T), dim=1)
        return self.mlp(mlp_in)

def get_base_function(q_func, q_func_components):
    if q_func == "CNN":
        cnn_layers, mlp_layers = CNN.create_conv1d_layers(*[int(param) for param in q_func_components])
        q_func = CNN(cnn_layers, mlp_layers)
    elif q_func == "LSTM":
        q_func = LSTM(*[int(param) for param in q_func_components])
    return q_func

def load_q_func(q_func_name, path=f"{os.path.abspath('')}/Models", device=DEVICE, eval=True): # the model parameters are encoded in the name
    q_func_components = q_func_name.split("_")
    _ = q_func_components.pop(0)
    q_func = q_func_components.pop(0)
    q_func = get_base_function(q_func, q_func_components)
    q_func_state_dict = torch.load(os.path.join(path, q_func_name), map_location=device)
    q_func.load_state_dict(q_func_state_dict)
    if eval:
        q_func.eval()
    return q_func

def load_policy_value_func(actor_critic_func_name, path=f"{os.path.abspath('')}/Models", device=DEVICE, eval=True):
    ac_components = actor_critic_func_name.split("_")
    _ = ac_components.pop(0)
    action_space = int(ac_components.pop(0))
    final_layer_size = int(ac_components.pop(0))
    base_func = ac_components.pop(0)
    base_func = get_base_function(base_func, ac_components)
    policy_value_func = Policy_Value(base_func, final_layer_size, action_space)
    q_func_state_dict = torch.load(os.path.join(path, actor_critic_func_name), map_location=device)
    policy_value_func.load_state_dict(q_func_state_dict)
    if eval:
        policy_value_func.eval()
    return policy_value_func

############################### Actor-Critic utilities ###########################################

class Policy_Value(nn.Module):

    def __init__(self, base_model, final_layer_size, action_space) -> None:
        super(Policy_Value, self).__init__()
        if type(base_model) == str: ############## load model
            pass
        # base model for the policy and value function estimation 
        self.base_model = base_model
        # actor's final layers
        self.policy_layer = nn.Linear(final_layer_size, action_space)
        self.policy_activation = nn.Softmax(dim=1)
        # critic's final layer to estmate value of state
        self.value_layer = nn.Linear(final_layer_size, action_space)

    def forward(self, S_t):
        base_out = self.base_model(S_t)
        action_out = self.policy_layer(base_out)
        action_prob = self.policy_activation(action_out)
        state_values = self.value_layer(base_out)
        return action_prob, state_values
    
class Advantages(Dataset):
    
    def __init__(self, q_vals, actions, rewards, log_pi, q_vals_, pi_, device=DEVICE) -> None:
        self.q_vals = torch.stack(q_vals).float().to(device)
        self.actions = torch.stack(actions).float().to(device)
        self.rewards = torch.stack(rewards).float().to(device)
        self.log_pi = torch.stack(log_pi).float().to(device)
        self.q_vals_ = torch.stack(q_vals_).float().to(device)
        self.pi_ = torch.stack(pi_).float().to(device)

    def __len__(self):
        return len(self.q_vals)
    
    def __getitem__(self, idx):
        return self.q_vals[idx], self.actions[idx], self.rewards[idx], self.log_pi[idx], self.q_vals_[idx], self.pi_[idx]

class ACTOR_CRITIC_AGENT:

    def __init__(self, policy_value_func, action_space, device, gamma=0.99, optimizer=Adam) -> None:
        self.policy_value_func = policy_value_func.float().to(device)
        self.action_space = action_space
        self.device = device
        self.q_vals, self.actions, self.rewards, self.pi, self.log_pi = list(), list(), list(), list(), list()
        self.gamma = torch.tensor(gamma).float().to(self.device)
        self.q_value_loss = nn.MSELoss(reduction='sum')
        self.optimizer = optimizer(self.policy_value_func.parameters())
    
    def select_action(self, S_t):
        S_t = self.state_to_device(S_t)
        probs, q_values = self.policy_value_func(S_t)
        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)
        # and sample an action using the distribution
        action = m.sample()
        one_hot = np.zeros(probs.shape)
        one_hot[0][action.item()] = 1.0
        one_hot = torch.from_numpy(one_hot)
        self.actions.append(one_hot)
        self.q_vals.append(q_values) # Agent uses action values instead of state values
        self.pi.append(probs)
        self.log_pi.append(m.log_prob(action))
        return action.item()

    def state_to_device(self, S_t):
        window, position = S_t
        window = torch.from_numpy(window).float().to(self.device)
        r, c = window.shape
        window = window.reshape(1, r, c)
        position = torch.tensor(position).float().to(self.device)
        return (window, position)

    def add_reward(self, R_t):
        self.rewards.append(torch.tensor([R_t], requires_grad=True))

    def get_avg_reward(self):
        return torch.mean(torch.tensor(self.rewards))

    def get_sum_reward(self):
        return torch.sum(torch.tensor(self.rewards))

    def get_data_loader(self, batch_size):
        data_set = Advantages(self.q_vals, self.actions, self.rewards, self.log_pi, self.q_vals_, self.pi_)
        data_loader = DataLoader(data_set, batch_size, shuffle=True)
        return data_loader

    def train(self, batch_size=BATCH_SIZE):
        self.q_vals_ = self.q_vals[1:]                        # q-values of the next state, so remove the first q-values in the copy and
        self.q_vals_.append(torch.zeros_like(self.q_vals[0])) # add zero valued q-values at the end of the episode
        self.pi_ = self.pi[1:]                                # do the same as above with the policy probabilities 
        self.pi_.append(torch.zeros_like(self.pi[0])) 
        batch_size = len(self.pi_)
        data_loader = self.get_data_loader(batch_size)
        for b_q, b_a, b_r, b_log_pi, b_q_, b_pi_ in data_loader:
            pred = torch.sum(b_a * b_q, dim=2)
            target = b_r + torch.sum(b_q_*b_pi_, dim=2)
            advntgs = target-pred
            policy_loss = (-b_log_pi.T)@advntgs
            q_value_loss = torch.sum(torch.square(advntgs))
            total_loss = policy_loss+q_value_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
        torch.cuda.empty_cache()
        self.q_vals, self.actions, self.rewards, self.pi, self.log_pi = list(), list(), list(), list(), list()
