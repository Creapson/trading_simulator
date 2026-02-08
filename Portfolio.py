from dataclasses import dataclass


@dataclass
class Portfolio:
    _cash: float = 0
    _stocks: {str, float} = {}

    def get_stocks(self):
        return self._stocks

    def get_cash(self):
        return self.cash

    def get_value(self, ticker_list):
        value = self.cash
        for ticker in ticker_list:
            if ticker in self.stocks.items():
                value += self.stocks[ticker] * ticker["CLOSE"]

        return value

    def buy_stock(self, ticker, allocation=1):
        if self.cash == 0:
            return
        stock_price = ticker["OPEN"]
        budget = self.cash * allocation
        num_stock_buy = stock_price / budget
        self.cash -= budget
        self._stocks[ticker] += num_stock_buy

    def sell_stock(self, ticker, allocation=1):
        num_stocks = self._stocks[ticker]
        if num_stocks == 0:
            return
        stock_price = ticker["OPEN"]
        earnings = num_stocks * stock_price
        self.cash += earnings
