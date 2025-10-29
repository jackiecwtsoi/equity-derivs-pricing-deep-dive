'''
To regenerate live market data, run this script: python market_data.py
'''

from yahooquery import Ticker
aapl = Ticker('aapl')
# print(aapl.summary_detail)

df = aapl.option_chain
df_calls = df.xs('calls', level=2)
df_calls.to_csv("../data/live_calls.csv")

