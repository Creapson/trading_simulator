class Portfolio:
    def __init__(self, cash=0, stocks=None):
        self._cash = cash
        self._stocks = stocks or {}
        self.trade_history = []

    def get_stocks(self):
        return self._stocks

    def get_cash(self):
        return self._cash

    def get_value(self, ticker, close_price):
        value = self._cash

        if ticker in self._stocks:
            value += self._stocks[ticker] * close_price

        return value

    def buy_stock(self, ticker, price, allocation=1):
        if self._cash <= 0:
            return

        budget = self._cash * allocation
        num_stocks = budget / price

        self._cash -= budget
        self._stocks[ticker] = self._stocks.get(ticker, 0) + num_stocks

    def sell_stock(self, ticker, price, allocation=1):
        num_stocks = self._stocks.get(ticker, 0)

        if num_stocks <= 0:
            return

        earnings = num_stocks * price

        self._cash += earnings
        self._stocks[ticker] = 0

    def add_trade(self, ticker, date, is_long):
        pass
