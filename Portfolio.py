
import pandas as pd

class Portfolio:
    def __init__(self, cash=0):
        self._cash = cash
        self._stocks = {}  # Ticker: [Anzahl, Durchschnittspreis]
        self.trade_history = []

    def add_cash(self, amount):
        self._cash  += amount

    def add_dividend(self, ticker, payout):
        num_stocks, _ = self._stocks.get(ticker, (0, 0))
        self.add_cash(num_stocks * payout)

    def add_stock_split(self, ticker, factor):
        num_stocks, avg_buy_price = self._stocks.get(ticker, (0, 0))
        num_stocks_new = num_stocks * factor
        avg_buy_price_new = avg_buy_price / factor
        self._stocks[ticker] = (num_stocks_new, avg_buy_price_new)

    def get_value(self, ticker, close_price):
        value = self._cash

        if ticker in self._stocks:
            num_stocks, _ = self._stocks.get(ticker, (0, 0))
            value += num_stocks * close_price

        return value

    def long_stock(self, ticker, price, date, allocation=1.0):
        if self._cash <= 0: return
        
        amount_to_spend = self._cash * allocation
        num_stocks = amount_to_spend / price
        
        self._cash -= amount_to_spend
        self._add_stock_to_inventory(ticker, price, num_stocks)
        self._record_trade(date, "Long", ticker, num_stocks, price)

    def short_stock(self, ticker, price, date, allocation=1.0):
        notional_value = self._cash * allocation
        num_stocks = notional_value / price
        
        self._cash += (num_stocks * price)
        self._add_stock_to_inventory(ticker, price, -num_stocks)
        self._record_trade(date, "Short", ticker, num_stocks, price)

    def close_position(self, ticker, price, date):
        num_stocks, avg_buy_price = self._stocks.get(ticker, (0, 0))
        if num_stocks == 0: return

        if num_stocks > 0:
            performance = (price / avg_buy_price) - 1
            self._cash += (num_stocks * price)
        else:
            performance = (avg_buy_price - price) / avg_buy_price
            self._cash += (num_stocks * price) 

        self._stocks[ticker] = (0, 0)
        self._record_trade(date, "Close", ticker, abs(num_stocks), price, performance)

    def _add_stock_to_inventory(self, ticker, price, num_stocks):
        num_old, avg_old = self._stocks.get(ticker, (0, 0))
        num_new = num_old + num_stocks
        
        if num_new == 0:
            avg_new = 0
        else:
            avg_new = (abs(num_old) * avg_old + abs(num_stocks) * price) / abs(num_new)
        
        self._stocks[ticker] = (num_new, avg_new)

    def _record_trade(self, date, t_type, ticker, num, price, perf=0):
        self.trade_history.append({
            "Date": date,
            "Type": t_type,
            "Ticker": ticker,
            "Shares": round(num, 4),
            "Price": round(price, 2),
            "Performance": f"{perf:.2%}",
            "Portfolio_Cash": round(self._cash, 2)
        })

    def print_trade_history(self):
        print(pd.DataFrame(self.trade_history))
