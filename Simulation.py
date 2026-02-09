import matplotlib.pyplot as plt
import pandas as pd

from Portfolio import Portfolio
from Strategy import Strategy
from Ticker import Ticker


class Simulation:
    def __init__(
        self,
        strategys: [Strategy] = [],
        ticker: Ticker = None,
        start_date: str = None,
        end_date: str = None,
    ):
        self.strategys = strategys
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.portfolio = None

        self.results = []
        self.indicator_list = []

        self.calc_indicators()

    def calc_indicators(self):
        indicators = set()
        for strategy in self.strategys:
            indicators.update(strategy.get_dependencies())
        self.indicator_list = list(indicators)
        self.ticker.add_indicators(self.indicator_list)

    def start(self):
        self.results = []
        for strategy in self.strategys:
            print("Started Simulator for: ", strategy.name)
            result, strat_name = self.simulate(strategy)
            self.results.append((result, strat_name))

    def simulate(self, strategy):
        df = self.ticker.get_dataframe()
        self.portfolio = Portfolio(cash=df["CLOSE"].iloc[0])
        sell_signal = False
        buy_signal = False
        results = []
        for date, row in df.iterrows():
            # handle signals
            if buy_signal:
                buy_signal = False
                self.portfolio.buy_stock(self.ticker, date)

            if sell_signal:
                sell_signal = False
                self.portfolio.sell_stock(self.ticker, date)

            # calculate signal for next day
            buy_signal, sell_signal = strategy.evaluate(row)
            results.append(
                {
                    "Date": date,
                    strategy.name: self.portfolio.get_value(self.ticker, date),
                }
            )

        return pd.DataFrame(results).set_index("Date"), strategy.name

    def plot_results(self, show_indicators=False):
        df = self.ticker.get_dataframe()
        plt.figure(figsize=(14, 7))

        # plot simulation results
        for result, strat_name in self.results:
            plt.plot(
                result.index,
                result[strat_name],
                label=strat_name,
                linewidth=1,
            )

        # plot indicator
        if show_indicators:
            # plot indicators
            for indicator in self.indicator_list:
                plt.plot(
                    df.index,
                    df[indicator],
                    label=indicator,
                    linewidth=1,
                )

        plt.title("Simulation results")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
