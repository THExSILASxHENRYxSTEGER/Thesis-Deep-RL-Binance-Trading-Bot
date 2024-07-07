from data_interface import Interface
from Data_Fetcher.global_variables import WINDOW_SIZES, TREASURY_INTEREST_API_CODES, DQN_ACTIONS, BINANCE_TRANSACTION_COST, TICKERS
import numpy as np
from copy import deepcopy

class Environment:
    
    def __init__(self, intfc, set_type="train", interval="1h", prcs_included=["open","high","low","close"], other_cols=["volume"], make_rtrns=True, normalize=True) -> None:
        self.action_space = len(DQN_ACTIONS.keys())
        ds = intfc.get_set_type_dataset(set_type=set_type, interval=interval)
        gnrl_data, spcfc_data = intfc.get_overall_data_and_ticker_dicts(ds)
        # if any of these conditions are fulfilled the sliding windows can just be taken (maybe normalized) 
        if set_type == "train": #or not make_rtrns or not normalize: 
            # self.states are the sliding windows of ticker specific and unspecific data
            self.episodes = self.__get_sliding_window_episodes(gnrl_data, spcfc_data, interval, prcs_included, other_cols, make_rtrns, normalize) 
        # the test and validation set data needs to be normalized with the values in the training set so we need to extract the normmalization values and then normalize the windows with those values
        else: 
            norm_cols = None
            if normalize:
                train_ds = intfc.get_set_type_dataset(set_type="train", interval=interval)
                train_gnrl_data, spcfc_train_data = intfc.get_overall_data_and_ticker_dicts(train_ds)
                norm_cols = self.__get_sliding_window_episodes(train_gnrl_data, spcfc_train_data, interval, prcs_included, other_cols, make_rtrns, normalize, get_norm_cols=True) 
            ds = intfc.get_set_type_dataset(set_type, interval)
            gnrl_data, spcfc_data = intfc.get_overall_data_and_ticker_dicts(ds)
            self.episodes = self.__get_sliding_window_episodes(gnrl_data, spcfc_data, interval, prcs_included, other_cols, make_rtrns, normalize, norm_cols=norm_cols)

    def __get_sliding_window_episodes(self, gnrl_data, spcfc_data, interval="1h", prcs_included=["open","high","low","close"], other_cols=["volume"], make_rtrns=True, normalize=True, get_norm_cols=False, norm_cols=None):
        self.trnsctn_costs = list()
        gnrl_data = np.array([gnrl_data[f"{trsry}_Treasury_Yield"] for trsry in TREASURY_INTEREST_API_CODES.keys()])
        if make_rtrns: 
            gnrl_data = gnrl_data[:,1:]
        if normalize and norm_cols == None: 
            gnrl_data = Interface.norm_srs(gnrl_data, get_norm_cols)
        if get_norm_cols: # get the columns normalized from Interface.norm_srs where get_max_axis is True
            self.future_norm_cols = list()
        if norm_cols != None:
            gnrl_data_norm, spcfc_train_data_norm = norm_cols
            gnrl_data = (gnrl_data.T/gnrl_data_norm).T
        sliding_windows = list()
        for i, tckr in enumerate(spcfc_data.keys()):
            prc_srs = np.array([spcfc_data[tckr][prc_key] for prc_key in prcs_included]) # prices of the ticker
            non_price_data = np.array([spcfc_data[tckr][other_col] for other_col in other_cols]) #  the other cols included are also normalized with respect to their own max
            trnsctn_cost = BINANCE_TRANSACTION_COST
            if make_rtrns:
                prc_srs = np.array([Interface.make_prices_to_returns(srs) for srs in prc_srs]) # make the raw prices to returns
                non_price_data = non_price_data[:,1:]
            if normalize:
                trnsctn_cost /= np.max(np.abs(prc_srs[0]))
                if norm_cols == None:
                    prc_srs = Interface.norm_srs(prc_srs, get_norm_cols) # if get_norm_cols is True then we get the normalization values of all rows instead of th normlized price series
                    non_price_data = Interface.norm_srs(non_price_data, get_norm_cols) # same as with the price series in the line above this line 
                else:
                    prc_srs_norm, non_price_data_norm = spcfc_train_data_norm[i]
                    prc_srs =  (prc_srs.T/prc_srs_norm).T
                    non_price_data = (non_price_data.T/non_price_data_norm).T
            if get_norm_cols:
                self.future_norm_cols.append((prc_srs, non_price_data))
                continue
            crncy_vals = np.concatenate([prc_srs, non_price_data, gnrl_data], axis=0)
            crncy_sliding_windows = Environment.__sliding_windows(crncy_vals, WINDOW_SIZES[interval])
            sliding_windows.append(crncy_sliding_windows)
            self.trnsctn_costs.append(trnsctn_cost)
        if get_norm_cols: 
            return (gnrl_data, self.future_norm_cols)
        return np.array(sliding_windows)

    @staticmethod
    def __sliding_windows(crncy_vals, window_size):
        _, T = crncy_vals.shape
        sliding_windows = list()
        for i in range(T-window_size):
            sliding_windows.append(crncy_vals[:, i:i+window_size])
        return np.array(sliding_windows)

    def get_episode_windows(self):
        return self.episodes

    def reset(self, start_selling=True, episode_nr=None):
        self.position = DQN_ACTIONS["SELL"] if start_selling else DQN_ACTIONS["BUY"] # the position we are in at the moment of the episode (at the beginning of any episode we assume no investment into the currency yet, unless in a continuos setting where we only care about the returns )
        n, _, _, _ = self.episodes.shape
        if episode_nr == None:
            self.episode_nr = np.random.randint(0,n) 
        else:
            self.episode_nr = episode_nr
        self.episode = self.episodes[self.episode_nr] # the current episode is the historic data of one ticker over the current set type
        self.episode_idx = 0                     # this is the index of the current state within the current episode
        S_t = [self.episode[self.episode_idx], np.array(self.position)] # the state is the sliding window of the economic data and the current investment position BUY or SELL
        self.episode_trnsct_cost = self.trnsctn_costs[self.episode_nr]
        return S_t

    def step(self, A_t):
        self.episode_idx += 1
        S_window = self.episode[self.episode_idx]
        R = S_window[3][-1]     #########################!!!!!!!!!!!!!!!!!! check whether should be open or close possibly close
        if self.position == DQN_ACTIONS["BUY"] and A_t == DQN_ACTIONS["SELL"]:  # depending on the previous position anegative return can either be a positive reward if we sold or vice versa
            R = -R-self.episode_trnsct_cost
        elif self.position == DQN_ACTIONS["SELL"] and A_t == DQN_ACTIONS["BUY"]: 
            R = R-self.episode_trnsct_cost
        elif self.position == DQN_ACTIONS["SELL"] and A_t == DQN_ACTIONS["SELL"]: 
            R = -R 
        self.position = A_t
        S_prime = [S_window, np.array(deepcopy(self.position))]
        D = True if self.episode_idx+1 == self.episode.shape[0] else False
        return S_prime, deepcopy(R), D
    
class Simple_Env(Environment):
    
    def reset(self, start_selling=True, episode_nr=None):
        n, _, _, _ = self.episodes.shape
        if episode_nr == None:
            self.episode_nr = np.random.randint(0,n) 
        else:
            self.episode_nr = episode_nr
        self.episode = self.episodes[self.episode_nr] # the current episode is the historic data of one ticker over the current set type
        self.episode_idx = 0                     # this is the index of the current state within the current episode
        S_t = self.episode[self.episode_idx] # the state is the sliding window of the economic data and the current investment position BUY or SELL
        return S_t

    def step(self, A_t):
        self.episode_idx += 1
        S_prime = self.episode[self.episode_idx]
        R = S_prime[3][-1]     #########################!!!!!!!!!!!!!!!!!! check whether should be open or close possibly close
        if A_t == DQN_ACTIONS["SELL"]:  # depending on the previous position anegative return can either be a positive reward if we sold or vice versa
            R = -R
        D = True if self.episode_idx+1 == self.episode.shape[0] else False
        return S_prime, deepcopy(R), D


class ENVIRONMENT_DDPG(Environment):

    def __init__(self, intfc, set_type="train", interval="1h", prcs_included=["open", "high", "low", "close"], other_cols=["volume"], make_rtrns=True, normalize=False, n_root=4) -> None:
        super().__init__(intfc, set_type, interval, prcs_included, other_cols, make_rtrns, normalize)
        self.make_rtrns = make_rtrns
        self.n_common_vars = len(prcs_included)+len(other_cols) # variables that are not macroeconomic
        self.n_crncs, self.n_steps, _, _ = self.episodes.shape
        self.windows = list()
        n_root = n_root if type(n_root)==int else 1
        self.reward_manipulation = np.vectorize(lambda x: x**(1/n_root) if x >= 0 else -np.abs(x)**(1/n_root))
        train_ds = intfc.get_set_type_dataset(set_type="train", interval=interval)
        _, spcfc_train_data = intfc.get_overall_data_and_ticker_dicts(train_ds)
        self.volume_max = np.array([np.max(spcfc_train_data[tckr]["volume"]) for tckr in TICKERS])
        for i in range(self.n_steps):
            window = self.__rescale_rtrns_and_trnsctn_csts(self.episodes[:,i,:,:])
            self.windows.append(window)
        del self.episodes

    def __rescale_rtrns_and_trnsctn_csts(self, windows): # take root of all rtrns and transaction costs 
        economic_data = deepcopy(windows[0][self.n_common_vars:])
        un_common_vars, econ_filler_cols = economic_data.shape
        econ_filler_rows = np.abs(self.n_common_vars-un_common_vars)
        fillers = np.zeros((econ_filler_rows, econ_filler_cols))
        economic_data = np.vstack([economic_data, fillers])
        time_series = list()
        for i, window in enumerate(windows):
            window_ = window[:self.n_common_vars]
            window_[4] /= self.volume_max[i]
            window_ = self.reward_manipulation(window_)
            time_series.append(window_)
        time_series.append(economic_data)
        return time_series
 
    def reset(self):
        self.episode_idx = 0
        windows = self.windows[self.episode_idx]
        S_t = windows
        return S_t
    
    def step(self, A_ts): # A_ts is here a vector of new weights
        self.episode_idx += 1
        S_prime = self.windows[self.episode_idx]
        if self.make_rtrns:
            rwrds = np.array([S_prime[i][3][-1] for i in range(len(TICKERS))])
        else:
            rwrds = np.array([S_prime[i][3][-1]-S_prime[i][3][-2] for i in range(len(TICKERS))])
        R_t = rwrds*A_ts
        D = True if self.episode_idx+1 == len(self.windows) else False
        return S_prime, R_t, D
    
    def get_action_space(self):
        return self.n_crncs


#class ENVIRONMENT_DDPG(Environment):
#
#    def __init__(self, intfc, set_type="train", interval="1h", prcs_included=["open", "high", "low", "close"], other_cols=["volume"], make_rtrns=True, normalize=True, n_root=None, consider_trnsctn_cost=True) -> None:
#        super().__init__(intfc, set_type, interval, prcs_included, other_cols, make_rtrns, normalize)
#        self.n_common_vars = len(prcs_included)+len(other_cols) # variables that are not macroeconomic
#        self.n_root = n_root
#        self.consider_trnsctn_cost = consider_trnsctn_cost
#        if n_root != None:
#            self.reward_manipulation = np.vectorize(lambda x: x**(1/n_root) if x >= 0 else -np.abs(x)**(1/n_root))
#            self.trnsctn_costs = self.reward_manipulation(np.array(self.trnsctn_costs)).tolist()
#        self.trnsctn_costs = np.hstack((self.trnsctn_costs, np.mean(self.trnsctn_costs)))
#        self.n_crncs, self.n_steps, _, _ = self.episodes.shape 
#        self.windows = list()
#        for i in range(self.n_steps):
#            window = self.__rescale_rtrns_and_trnsctn_csts(self.episodes[:,i,:,:])
#            self.windows.append(window)
#        del self.episodes
#
#    def __rescale_rtrns_and_trnsctn_csts(self, windows): # take root of all rtrns and transaction costs 
#        economic_data = deepcopy(windows[0][self.n_common_vars:])
#        time_series = list()
#        for window in windows:
#            window_ = window[:self.n_common_vars]
#            if self.n_root != None:
#                window_ = self.reward_manipulation(window_)
#            time_series.append(window_)
#        time_series.append(economic_data)
#        return time_series
# 
#    def reset(self):
#        self.episode_idx = 0
#        windows = self.windows[self.episode_idx]
#        self.pf_weights = np.zeros(self.n_crncs+1) 
#        self.pf_weights[self.n_crncs] = 1.0
#        S_t = (windows, deepcopy(self.pf_weights))
#        return S_t
#    
#    def step(self, A_ts): # A_ts is here a vector of new weights
#        self.episode_idx += 1
#        window = self.episode[self.episode_idx]
#        rtrns =  window[np.array(range(self.n_crncs))*self.n_common_vars][:,-1]
#        neg_rtrn_indcs = np.where(rtrns<0)[0]
#        if len(neg_rtrn_indcs) == 0 or len(neg_rtrn_indcs) == self.n_crncs:
#            hold_rtrn = -np.sum(rtrns)
#        else:
#            hold_rtrn = np.mean(rtrns[neg_rtrn_indcs])
#        rtrns = np.hstack((rtrns, hold_rtrn))
#        R_t = rtrns*A_ts
#        weight_change = A_ts-self.pf_weights
#        if self.consider_trnsctn_cost:
#            neg_change_indcs = np.where(weight_change <= 0)[0]
#            pos_trnsctn_cost = deepcopy(self.trnsctn_costs)
#            pos_trnsctn_cost[neg_change_indcs] = 0
#            step_trnsctn_csts = pos_trnsctn_cost*weight_change
#            R_t = R_t-step_trnsctn_csts
#        self.pf_weights = A_ts
#        S_t = (window, self.pf_weights)
#        D = True if self.episode_idx+1 == self.episode.shape[0] else False
#        return S_t, R_t, D
#    
#    def get_action_space(self):
#        return self.n_crncs+1


class ENVIRONMENT_DQN(ENVIRONMENT_DDPG):

    def __init__(self, intfc, set_type="train", interval="1h", prcs_included=["open", "high", "low", "close"], other_cols=["volume"], make_rtrns=True, normalize=False, n_root=4) -> None:
            super().__init__(intfc, set_type, interval, prcs_included, other_cols, make_rtrns, normalize, n_root=n_root)

    def step(self, A_t):
        self.episode_idx += 1
        S_prime = self.windows[self.episode_idx]
        if self.make_rtrns:
            rwrds = np.array([S_prime[i][3][-1] for i in range(len(TICKERS))])
        else:
            rwrds = np.array([S_prime[i][3][-1]-S_prime[i][3][-2] for i in range(len(TICKERS))])
        if A_t == 0:
            pos_rwrds_indcs = np.where(rwrds>0)[0]
            if len(pos_rwrds_indcs) == 0 or len(pos_rwrds_indcs) == self.n_crncs:
                R_t = -np.sum(rwrds)
            else:
                R_t = -np.sum(rwrds[pos_rwrds_indcs])
        else:
            binary_A_t = "{0:b}".format(A_t)
            #buy_frac = 1/np.sum([int(n) for n in binary_A_t])
            R_t = np.sum([int(n)*rwrd for rwrd, n in zip(rwrds, reversed(binary_A_t))])
        D = True if self.episode_idx+1 == len(self.windows) else False
        return S_prime, R_t, D
    
    def get_action_space(self):
        return 2**self.n_crncs