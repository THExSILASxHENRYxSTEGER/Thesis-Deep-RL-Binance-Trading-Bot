from data_interface import Interface
from Data_Fetcher.global_variables import WINDOW_SIZES, TREASURY_INTEREST_API_CODES, DQN_ACTIONS, BINANCE_TRANSACTION_COST
import numpy as np

class Environment:
    
    def __init__(self, intfc, set_type="train", interval="1h", prcs_included=["open","high","low","close"], other_cols=["volume"], make_rtrns=True, normalize=True) -> None:
        ds = intfc.get_set_type_dataset(set_type, interval)
        gnrl_data, spcfc_train_data = intfc.get_overall_data_and_ticker_dicts(ds)
        self.episodes = self.get_sliding_windows(gnrl_data, spcfc_train_data, interval, prcs_included, other_cols, make_rtrns, normalize) #  self.states are the sliding windows of ticker specific and unspecific data
        self.action_space = len(DQN_ACTIONS.keys())

    def get_sliding_windows(self, gnrl_data, spcfc_train_data, interval="1h", prcs_included=["open","high","low","close"], other_cols=["volume"], make_rtrns=True, normalize=True):
        self.trnsctn_costs = list()
        gnrl_data = np.array([gnrl_data[f"{trsry}_Treasury_Yield"] for trsry in TREASURY_INTEREST_API_CODES.keys()])
        if make_rtrns: gnrl_data = gnrl_data[:,1:]
        if normalize: gnrl_data = Interface.norm_srs(gnrl_data)
        sliding_windows = list()
        for tckr in spcfc_train_data.keys():
            prc_srs = np.array([spcfc_train_data[tckr][prc_key] for prc_key in prcs_included]) # prices of the ticker
            non_price_data = np.array([spcfc_train_data[tckr][other_col] for other_col in other_cols]) #  the other cols included are also normalized with respect to their own max
            trnsctn_cost = BINANCE_TRANSACTION_COST
            if make_rtrns:
                prc_srs = np.array([Interface.make_prices_to_returns(srs) for srs in prc_srs]) # make the raw prices to returns
                non_price_data = non_price_data[:,1:]
            if normalize:
                trnsctn_cost /= np.max(np.abs(prc_srs[0]))
                prc_srs = Interface.norm_srs(prc_srs)
                non_price_data = Interface.norm_srs(non_price_data)
            crncy_vals = np.concatenate([prc_srs, non_price_data, gnrl_data], axis=0)
            crncy_sliding_windows = Environment.__sliding_windows(crncy_vals, WINDOW_SIZES[interval])
            sliding_windows.append(crncy_sliding_windows)
            self.trnsctn_costs.append(trnsctn_cost)
        return np.array(sliding_windows)
    
    @staticmethod
    def __sliding_windows(crncy_vals, window_size):
        _, T = crncy_vals.shape
        sliding_windows = list()
        for i in range(T-window_size):
            sliding_windows.append(crncy_vals[:, i:i+window_size])
        return np.array(sliding_windows)

    def reset(self):
        self.position = DQN_ACTIONS["SELL"] # the position we are in at the moment of the episode (at the beginning of any episode we assume no investment into the currency yet)
        n, _, _, _ = self.episodes.shape
        episode_nr = np.random.randint(0,n)
        self.episode = self.episodes[episode_nr] # the current episode is the historic data of one ticker over the current set type
        self.episode_idx = 0                # this is the index of the current state within the current episode
        S_t = [self.episode[self.episode_idx], self.position] # the state is the sliding window of the economic data and the current investment position BUY or SELL
        self.episode_trnsct_cost = self.trnsctn_costs[episode_nr]
        return S_t

    def step(self, A_t):
        self.episode_idx += 1
        S_window = self.episode[self.episode_idx]
        R = S_window[0][-1]
        if self.position == DQN_ACTIONS["BUY"] and A_t == DQN_ACTIONS["SELL"]: 
            R = -R-self.episode_trnsct_cost
        elif self.position == DQN_ACTIONS["SELL"] and A_t == DQN_ACTIONS["BUY"]: 
            R = R-self.episode_trnsct_cost
        elif self.position == DQN_ACTIONS["SELL"] and A_t == DQN_ACTIONS["SELL"]: 
            R = -R 
        self.position = A_t
        S_prime = [S_window, self.position]
        D = True if self.episode_idx+1 == self.episode.shape[0] else False
        return S_prime, R, D
