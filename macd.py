import numpy as np

class MACD:
    
    def __init__(self, S=12, L=26, p_timescale=63, q_timescale=252) -> None:
        self.S = S
        self.L = L
        self.p_timescale = p_timescale
        self.q_timescale = q_timescale


