import pandas as pd


class Portfolio:
    def __init__(self, cash=0):
        self._cash = cash
        self._stocks = {}
        self.trade_history = []

    def get_stocks(self):
        return self._stocks

    def get_cash(self):
        return self._cash

    def add_stock(self, ticker, price, num_stocks):
        num_stocks_old, avg_price_old = self._stocks.get(ticker, (0, 0))
        num_stocks_new = num_stocks_old + num_stocks
        avg_price_new = (
            num_stocks_old * avg_price_old + num_stocks * price
        ) / num_stocks_new

        self._stocks[ticker] = (num_stocks_new, avg_price_new)
        pass

    def remove_stock(self, ticker, num_stocks):
        num_stocks_old, avg_buy_price = self._stocks.get(ticker, (0, 0))

        if num_stocks_old < num_stocks:
            return 0

        self._stocks[ticker] = (num_stocks_old - num_stocks), avg_buy_price
        return avg_buy_price

    def get_value(self, ticker, close_price):
        value = self._cash

        if ticker in self._stocks:
            num_stocks, _ = self._stocks.get(ticker, (0, 0))
            value += num_stocks * close_price

        return value

    def buy_stock(self, ticker, price, date, allocation=1):
        num_stocks = (self._cash * allocation) / price
        self.add_trade(num_stocks, ticker, date, True, price)

    def sell_stock(self, ticker, price, date, allocation=1):
        num_stocks_current, _ = self._stocks.get(ticker, (0, 0))
        num_stocks = num_stocks_current * allocation
        self.add_trade(num_stocks, ticker, date, False, price)

    def add_trade(self, num_stocks, ticker, date, is_long, price):
        # Buy
        if is_long:
            if self._cash <= 0:
                return

            self._cash -= num_stocks * price
            self.add_stock(ticker, price, num_stocks)
            profit = 0

        # Sell
        else:
            num_stocks, _ = self._stocks.get(ticker, (0, 0))

            avg_price = self.remove_stock(ticker, num_stocks)
            earnings = price * num_stocks
            self._cash += earnings

            profit = price / avg_price

        trade_type = "Buy" if is_long else "Sell"
        stocks_remaining, _ = self._stocks.get(ticker, (0, 0))

        self.trade_history.append(
            {
                "Date": date,
                "Type": trade_type,
                "Ticker": ticker,
                "Num Stocks": num_stocks,
                "Remaining": stocks_remaining,
                "Price": price,
                "Profit": profit,
            }
        )

    def print_trade_history(self):
        df = pd.DataFrame(self.trade_history)
        print(df)

    def get_trade_history(self):
        return self.trade_history
