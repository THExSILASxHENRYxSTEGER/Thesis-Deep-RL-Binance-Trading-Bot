########### script that gets called by systemd every 30 min to make a trade ###########
import sys
sys.path.append("../Data_Fetcher")
from Data_Fetcher.global_variables import TICKERS
from Binance_Trading.trading_utils import get_client, get_prev_30_min_OHLC

client = get_client()

ohcl_data = [get_prev_30_min_OHLC(client, tickr) for tickr in TICKERS]

g = 0

# check current positions and 