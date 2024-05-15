from baselines import BuyAndHold

# for the buy and hold strategy the time interval is not important, since the overall return 
# is only dependent on the first and the last price in the entire interval to be measured
# ie. overall return = (p_T-p_0)/p_0 
# where p_0 is the first price and p_T the last price in any price series      

buy_hold = BuyAndHold()
buy_hold.print_set_avg_rtrns()
buy_hold.plot_rtrns()
buy_hold.plot_rtrns(False)