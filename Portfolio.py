class Portfolio:
    def __init__(self, cash=0, stocks=None):
        self._cash = cash
        self._stocks = stocks or {}
        self.trade_history = []

    def get_stocks(self):
        return self._stocks

    def get_cash(self):
        return self._cash

    def get_value(self, ticker, date):
        value = self._cash
        symbol = ticker.name

        if symbol in self._stocks:
            close_price = ticker.get_dataframe().loc[date, "CLOSE"]
            value += self._stocks[symbol] * close_price

        return value

    def buy_stock(self, ticker, date, allocation=1):
        if self._cash <= 0:
            return

        symbol = ticker.name
        open_price = ticker.get_dataframe().loc[date, "CLOSE"]

        budget = self._cash * allocation
        num_stocks = budget / open_price  # ← fixed math

        self._cash -= budget
        self._stocks[symbol] = self._stocks.get(symbol, 0) + num_stocks

    def sell_stock(self, ticker, date, allocation=1):
        symbol = ticker.name
        num_stocks = self._stocks.get(symbol, 0)

        if num_stocks <= 0:
            return

        open_price = ticker.get_dataframe().loc[date, "CLOSE"]
        earnings = num_stocks * open_price

        self._cash += earnings
        self._stocks[symbol] = 0

    def add_trade(self, ticker, date, is_long):
        pass
