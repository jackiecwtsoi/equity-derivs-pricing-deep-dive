from yahooquery import Ticker

class MarketDataLoader():
    def load_call_option_chain(self, ticker):
        df = self.__load_option_chain(ticker).xs("calls", level=2)
        df.to_csv("../data/live_calls.csv")

    def load_put_option_chain(self, ticker):
        df = self.__load_option_chain(ticker).xs("puts", level=2)
        df.to_csv("../data/live_puts.csv")

    def __load_option_chain(self, ticker):
        return Ticker(ticker).option_chain