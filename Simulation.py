import matplotlib.pyplot as plt
import numpy as np
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
        self.ticker_loaded = self.ticker.add_indicators(self.indicator_list)
        # self.ticker.set_timespan(start_time="1929-01-01 00:00")

    def start(self):
        self.results = []
        if not self.ticker_loaded:
            print("Ticker not loaded properly. Cant start Simulation!")
            return
        for strategy in self.strategys:
            print("Started Simulator for: ", strategy.name)
            result, strat_name = self.simulate(strategy)
            self.results.append((result, strat_name))

    def simulate(self, strategy):
        df = self.ticker.get_dataframe()
        df = df.dropna()
        self.portfolio = Portfolio(cash=df["CLOSE"].iloc[0])
        sell_signal = False
        buy_signal = False
        results = []
        for date, row in df.iterrows():
            buy_signal, sell_signal = strategy.evaluate(row)
            # handle signals
            if buy_signal:
                buy_signal = False
                self.portfolio.buy_stock(self.ticker, date)

            if sell_signal:
                sell_signal = False
                self.portfolio.sell_stock(self.ticker, date)

            # calculate signal for next day
            results.append(
                {
                    "Date": date,
                    strategy.name: self.portfolio.get_value(self.ticker, date),
                }
            )

        return pd.DataFrame(results).set_index("Date"), strategy.name

    def plot_results(self, show_indicators=False, log_scale=False):
        df = self.ticker.get_dataframe()
        df.dropna()
        plt.figure(figsize=(14, 7))
        if not log_scale:
            plt.plot(
                df.index,
                df["CLOSE"],
                label="Price",
                linewidth=1,
            )
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
                        label="Indicator: " + indicator,
                        linewidth=1,
                    )
        else:
            plt.plot(
                df.index,
                np.log10(df["CLOSE"]),
                label="Price",
                linewidth=1,
            )
            # plot simulation results
            for result, strat_name in self.results:
                plt.plot(
                    result.index,
                    np.log10(result[strat_name]),
                    label=strat_name,
                    linewidth=1,
                )

            # plot indicator
            if show_indicators:
                # plot indicators
                for indicator in self.indicator_list:
                    plt.plot(
                        df.index,
                        np.log10(df[indicator]),
                        label="Indicator: " + indicator,
                        linewidth=1,
                    )

        plt.title("Simulation results")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
