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
        indicator_list = list(indicators)
        self.ticker_loaded = self.ticker.add_indicators(indicator_list)
        # self.ticker.set_timespan(start_time="2024-01-01 00:00")

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
        df = self.ticker.get_dataframe().dropna()

        signals = strategy.evaluate(df)

        signals = signals.reindex(df.index).fillna(0).shift(1)

        self.portfolio = Portfolio(cash=df["CLOSE"].iloc[0])

        results = []

        for date in df.index:
            signal = signals.loc[date]

            if signal == 1:
                price = df.loc[date, "OPEN"]
                self.portfolio.buy_stock(self.ticker.ticker, price)

            elif signal == -1:
                price = df.loc[date, "OPEN"]
                self.portfolio.sell_stock(self.ticker.ticker, price)

            close_price = df.loc[date, "CLOSE"]
            portfolio_value = self.portfolio.get_value(self.ticker.ticker, close_price)

            results.append({"Date": date, strategy.name: portfolio_value})

        return pd.DataFrame(results).set_index("Date"), strategy.name

    def plot_results(self, show_indicators=False, log_scale=False):
        df = self.ticker.get_dataframe()
        df.dropna()
        plt.figure(figsize=(14, 7))

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
            for indicator in self.ticker.get_used_indicators():
                plt.plot(
                    df.index,
                    df[indicator],
                    label="Indicator: " + indicator,
                    linewidth=1,
                )

        if log_scale:
            plt.yscale("log")
        plt.title("Simulation results")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
