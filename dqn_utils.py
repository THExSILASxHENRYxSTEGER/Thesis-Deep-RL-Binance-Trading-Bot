import numpy as np
import torch
from torch.optim import Adam
from torch import nn
from copy import deepcopy

torch.seed(0)

def state_to_device(S_t, device):
    window, position = S_t
    window = torch.tensor(window).float().to(device)
    r, c = window.shape
    window = window.reshape(1, r, c)
    position = torch.tensor(position).float().to(device)
    return (window, position)

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
        b_wndws = torch.tensor(b_wndws).float().to(self.device)
        b_pos = torch.tensor(b_pos).float().to(self.device)
        b_a = torch.eye(self.action_space)[b_a].to(self.device)
        b_r = torch.tensor(b_r).float().to(self.device)
        b_d = torch.tensor(b_d).float().to(self.device)
        b_wndws_ = torch.tensor(b_wndws_).float().to(self.device)
        b_pos_ = torch.tensor(b_pos_).float().to(self.device)
        return (b_wndws, b_pos), b_a, b_r, b_d, (b_wndws_, b_pos_)

class DQN_AGENT:

    def __init__(self, eps, action_space, network, device, gamma=0.99, optimizer=Adam, loss=nn.MSELoss) -> None:
        self.eps = eps
        self.action_space = action_space
        self.device = device
        self.gamma = gamma
        self.policy_net = network.float().to(self.device)
        self.target_net = deepcopy(network).float().to(self.device)
        self.update_target_net()
        self.optimizer = optimizer(self.policy_net.parameters())
        self.loss = loss()

    def take_action(self, S_t, t):
        if self.eps(t) > np.random.rand():
            return np.argmax(np.random.rand(self.action_space))
        else:
            S_t = state_to_device(S_t, self.device)
            A_t = torch.argmax(self.policy_net(S_t))
            torch.cuda.empty_cache()
            return A_t

    def train(self, b_s, b_a, b_r, b_d, b_s_): # at end of function torch.cuda.empty_cache()
        target = b_r + (torch.ones_like(b_d)-b_d) * self.gamma * torch.max(self.target_net(b_s_), dim=1)[0]
        pred = torch.sum(self.policy_net(b_s) * b_a, dim=1)
        loss = self.loss(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()
        return loss 

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

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
        #mlp_seq.append(nn.Softmax())
        self.cnn = nn.Sequential(*conv_seq)
        self.mlp = nn.Sequential(*mlp_seq)

    def forward(self, S_t): # inputs have to be of type float
        window, position = S_t
        cnn_out = self.cnn(window)
        cnn_out = torch.flatten(cnn_out, start_dim=1)
        mlp_in = torch.concat((cnn_out, torch.atleast_2d(position).T), dim=1)
        return self.mlp(mlp_in)

    @staticmethod
    def create_conv1d_layers(in_chnls, out_chnls, time_series_len, action_space, n_cnn_layers=2, kernel_size=4, 
                            kernel_div=1, cnn_intermed_chnls=16, mlp_intermed_size=128, n_mlp_layers=1, punctual_vals=1):
        cnn_layers = list()
        for i in range(n_cnn_layers):
            layer_dict = {
                "in_channels":cnn_intermed_chnls, 
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
                layer_dict["out_features"] = action_space
            mlp_layers.append(layer_dict)
        return cnn_layers, mlp_layers
    
class LSTM(nn.Module):
    
    def __init__(self) -> None:
        super(CNN, self).__init__()