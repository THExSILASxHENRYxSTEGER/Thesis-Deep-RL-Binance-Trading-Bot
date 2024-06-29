import numpy as np
import torch
from torch.optim import Adam
from torch import nn
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
import os
from Data_Fetcher.global_variables import DEVICE, BATCH_SIZE
from random_processes import OrnsteinUhlenbeckProcess

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

    def get_batch(self, one_hot_actions=True):
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
        b_pos = torch.tensor(np.array(b_pos)).float().to(self.device)
        if one_hot_actions:
            b_a = torch.eye(self.action_space)[np.array(b_a)].float().to(self.device)
        else:
            b_a = torch.tensor(np.array(b_a)).float().to(self.device)
        b_r = torch.tensor(np.array(b_r)).float().to(self.device)
        b_d = torch.tensor(np.array(b_d)).float().to(self.device)
        b_wndws_ = torch.tensor(np.array(b_wndws_)).float().to(self.device)
        b_pos_ = torch.tensor(np.array(b_pos_)).float().to(self.device)
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

    def select_action(self, S_t, n_episode, ):
        if self.eps(n_episode) > np.random.rand() and self.training: #>
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

    def __init__(self, cnn_layers, mlp_layers, final_activation=None) -> None:
        super(CNN, self).__init__()
        conv_seq, mlp_seq = list(), list()
        for layer in cnn_layers:
            conv_seq.append(nn.Conv1d(**layer))
            conv_seq.append(nn.LeakyReLU())
        self.cnn = nn.Sequential(*conv_seq)
        if mlp_layers == None:
            self.mlp = None
            return
        final_layer = mlp_layers.pop(len(mlp_layers)-1)
        for layer in mlp_layers:
            mlp_seq.append(nn.Linear(**layer))
            mlp_seq.append(nn.LeakyReLU())
        mlp_seq.append(nn.Linear(**final_layer))
        if final_activation != None:
            mlp_seq.append(final_activation)
        self.mlp = nn.Sequential(*mlp_seq)

    def forward(self, S_t): # inputs have to be of type float
        window, position = S_t
        cnn_out = self.cnn(window)
        cnn_out = torch.flatten(cnn_out, start_dim=1)
        if self.mlp == None:
            return cnn_out
        mlp_in = torch.concat((cnn_out, torch.atleast_2d(position)), dim=1)
        return self.mlp(mlp_in)

    @staticmethod
    def create_conv1d_layers(in_chnls, out_chnls, time_series_len, final_layer_size, n_cnn_layers=2, kernel_size=4, 
                            kernel_div=1, cnn_intermed_chnls=16, mlp_intermed_size=128, n_mlp_layers=1, punctual_vals=1, only_cnn=False):
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
        if only_cnn:
            return cnn_layers, time_series_len*out_chnls
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
    if ac_components[0] == "AC": 
        _ = ac_components.pop(0)
        action_space = int(ac_components.pop(0))
        final_layer_size = int(ac_components.pop(0))
        base_func = ac_components.pop(0)
        base_func = get_base_function(base_func, ac_components)
        policy_value_func = Policy_Value(base_func, final_layer_size, action_space)
        q_func_state_dict = torch.load(os.path.join(path, actor_critic_func_name), map_location=device)
        policy_value_func.load_state_dict(q_func_state_dict)
    elif ac_components[0] == "DQN":
        base_func = load_q_func(actor_critic_func_name, path)
        final_layer_size = action_space = int(ac_components[5])
        policy_value_func = Policy_Value(base_func, final_layer_size, action_space)
    else:
        raise RuntimeError("Only DQN and AC are valid model descriptions")
    if eval:
        policy_value_func.eval()
    return policy_value_func

############################### Actor-Critic utilities ###########################################

class Policy_Value(nn.Module):

    def __init__(self, base_model, final_layer_size, action_space) -> None:
        super(Policy_Value, self).__init__()
        # base model for the policy and value function estimation 
        self.base_model = base_model
        # actor's final layers
        self.additional_top_layers = (final_layer_size != action_space)
        if self.additional_top_layers:
            # critic's final layer to estmate value of state      
            self.value_layer = nn.Linear(final_layer_size, action_space)  #### !!!!!!! oddly the probailities and weights of are out of touch the probability of one is very big though the state value is much lower than the other
            # actors final layer to estimate probabilities
            self.policy_layer = nn.Linear(final_layer_size, action_space) #### !!!!!!!! critic and policy func having two different layers might be a problem check that
        self.policy_activation = nn.Softmax(dim=1)        
        
    def forward(self, S_t):
        base_out = self.base_model(S_t)
        if self.additional_top_layers:
            state_values = self.value_layer(base_out)
            action_out = self.policy_layer(base_out)
            action_prob = self.policy_activation(action_out)
            return action_prob, state_values
        action_prob = self.policy_activation(base_out)
        return action_prob, base_out
    
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

######################## Deep Deterministic Policy Gradient utils ################################
    
#class DDPG_AGENT:
#
#    def __init__(self, actor, critic, eps, device, random_process_params:dict, gamma=0.99, optimizer=Adam, value_loss_fn=nn.MSELoss, training=True, tau=0.001) -> None:
#        # hyperparameters
#        self.eps = eps
#        self.device = device
#        self.gamma = torch.tensor(gamma).to(self.device)
#        self.training = training
#        self.random_process = OrnsteinUhlenbeckProcess(**random_process_params)
#        self.tau = tau
#        # actor model
#        self.actor = actor.float().to(self.device)
#        self.actor_target = deepcopy(self.actor).float().to(self.device)
#        self.actor_optim  = optimizer(self.actor.parameters())
#        # critic model
#        self.critic = critic.float().to(self.device)
#        self.critic_target = deepcopy(self.critic).float().to(self.device)
#        self.critic_optim  = optimizer(self.critic.parameters())     
#        self.value_loss_fn = value_loss_fn()
#
#    def select_action(self, S_t, n_episode):
#        S_t = self.state_to_device(S_t)
#        A_t = self.actor(S_t).detach().numpy()
#        torch.cuda.empty_cache()
#        noise = int(self.training)*self.eps(n_episode)*np.random.normal(loc=0, scale=0.4)#np.random.normal(loc=0, scale=0.3)#
#        A_t += noise
#        A_t = np.clip(A_t, 0., 1.)
#        return A_t
#
#    def state_to_device(self, S_t):
#        window, position = S_t
#        window = torch.tensor(window).float().to(self.device)
#        r, c = window.shape
#        window = window.reshape(1, r, c)
#        position = torch.tensor(position).float().to(self.device)
#        return (window, position)
#
#    @staticmethod
#    def to_binary_action(A_t):
#        return int(A_t >= 0.1)
#
#    def train(self, b_s, b_a, b_r, b_d, b_s_):
#        # critic update
#        windows, pos = b_s
#        b_s_val = (windows, torch.vstack((pos, b_a.flatten())))
#        q_batch = self.critic(b_s_val)
#        windows_, pos_ = b_s_
#        b_s_val_ = (windows_, torch.vstack((pos_, self.actor_target(b_s_).flatten())))
#        target_q_batch = b_r.flatten() + (torch.ones_like(b_d)-b_d)*self.gamma*self.critic_target(b_s_val_).flatten()
#        value_loss = self.value_loss_fn(q_batch.flatten(), target_q_batch)
#        self.critic_optim.zero_grad()
#        value_loss.backward()
#        self.critic_optim.step()
#        # actor update
#        b_s_policy = (windows, torch.vstack((pos, self.actor(b_s).flatten())))
#        policy_loss = -self.critic(b_s_policy)
#        policy_loss = policy_loss.mean()
#        self.actor_optim.zero_grad()
#        policy_loss.backward()
#        self.actor_optim.step()
#        torch.cuda.empty_cache()
#        # target weights update
#        self.soft_update(self.actor_target, self.actor, self.tau)
#        self.soft_update(self.critic_target, self.critic, self.tau)
#
#    def soft_update(self, target, source, tau):
#        for target_param, param in zip(target.parameters(), source.parameters()):
#            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

###################### DDPG of weighting ##############################

class ACTOR(nn.Module):
    
    def __init__(self, base_model, actor_mlp) -> None:
        super(ACTOR, self).__init__()
        self.base_model = base_model
        self.actor_mlp = actor_mlp
        self.final_activation = nn.Softmax(dim=1) 
        
######## maybe not softmax but n+1 times tanh and softmax is applied by environment to get weights for steps
######## use OU process for stability 
######## use n seperate cnns plus one for economic data

    def forward(self, S_t):
        _, pos = S_t
        cnn_out = self.base_model(S_t)
        mlp_in = torch.concat((cnn_out, torch.atleast_2d(pos)), dim=1)
        mlp_out = self.actor_mlp(mlp_in)
        return self.final_activation(mlp_out)

class CRITIC(nn.Module):
    
    def __init__(self, base_model, critic_mlp) -> None:
        super(CRITIC, self).__init__()
        self.base_model = base_model 
        self.critic_mlp = critic_mlp
    
    def forward(self, S_t, A_t):
        _, pos = S_t
        cnn_out = self.base_model(S_t)
        mlp_in = torch.concat((cnn_out, pos, A_t), dim=1)
        mlp_out = self.critic_mlp(mlp_in)
        return mlp_out
    
class DDPG_AGENT_WEIGHTING:

    def __init__(self, actor, critic, eps, device, rnd_prcs_prms:dict, gamma=0.99, optimizer=Adam, value_loss_fn=nn.MSELoss, training=True, tau=0.001) -> None:
        #hyperparameters
        self.eps = eps
        self.device = device
        self.gamma = torch.tensor(gamma).to(self.device)
        self.training = training
        self.rnd_prcs_prms = rnd_prcs_prms
        self.tau = tau
        # actor model
        self.actor = actor.float().to(self.device)
        self.actor_target = deepcopy(self.actor).float().to(self.device)
        self.actor_optim  = optimizer(self.actor.parameters())
        # critic model
        self.critic = critic.float().to(self.device)
        self.critic_target = deepcopy(self.critic).float().to(self.device)
        self.critic_optim  = optimizer(self.critic.parameters())     
        self.value_loss_fn = value_loss_fn()

    def select_action(self, S_t, n_episode):
        S_t = self.state_to_device(S_t)
        A_t = self.actor(S_t).flatten().cpu().detach().numpy()
        torch.cuda.empty_cache()
        noise = np.random.normal(self.rnd_prcs_prms["mu"], self.rnd_prcs_prms["sigma"], self.rnd_prcs_prms["size"]) 
        noise *= int(self.training)*self.eps(n_episode)
        A_t += noise
        A_t = np.exp(A_t)/np.sum(np.exp(A_t))
        return A_t

    def state_to_device(self, S_t):
        window, position = S_t
        window = torch.tensor(window).float().to(self.device)
        r, c = window.shape
        window = window.reshape(1, r, c)
        position = torch.tensor(position).float().to(self.device)
        return (window, position)
    
    def train(self, b_s, b_a, b_r, b_d, b_s_):
        # critic update
        q_batch = self.critic(b_s, b_a)
        _, n_actions = b_a.shape
        b_d = torch.tile((torch.ones_like(b_d)-b_d), (n_actions, 1)).T
        target_q_batch = b_r + b_d * self.gamma * self.critic_target(b_s_, self.actor_target(b_s_))
        value_loss = self.value_loss_fn(q_batch, target_q_batch)
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()
        # actor update
        policy_loss = -self.critic(b_s, self.actor(b_s))
        policy_loss = policy_loss.mean() 
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        torch.cuda.empty_cache()
        # target weights update
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)