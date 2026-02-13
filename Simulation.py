import matplotlib.pyplot as plt
import pandas as pd

from Portfolio import Portfolio
from Strategy import Strategy
from Ticker import Ticker


class Simulation:
    def __init__(
        self,
        strategys: [Strategy] = [],
        tickers: [Ticker] = [],
        start_date: str = None,
        end_date: str = None,
    ):
        self.strategys = strategys
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

        self.results = []
        self.indicator_list = []

    def calc_indicators(self, ticker):
        indicators = set()
        for strategy in self.strategys:
            indicators.update(strategy.get_dependencies())
        indicator_list = list(indicators)
        return ticker.add_indicators(indicator_list)

    def set_timespan(self, start, end):
        self.ticker.set_timespan(start_time=start)

    def start(self):
        self.results = []

        for ticker in self.tickers:
            loaded = self.calc_indicators(ticker)
            if not loaded:
                print("Ticker not loaded properly. Cant start Simulation!")

            for strategy in self.strategys:
                print("Started Simulator for: ", strategy.name)
                result, strat_name = self.simulate(strategy, ticker)
                self.results.append((result, strat_name, ticker))

    def simulate(self, strategy, ticker):
        df = ticker.get_dataframe().dropna()

        signals = strategy.evaluate(df)

        signals = signals.reindex(df.index).fillna(0).shift(1)

        portfolio = Portfolio(cash=df["CLOSE"].iloc[0])

        results = []

        for date in df.index:
            signal = signals.loc[date]

            if signal == 1:
                price = df.loc[date, "OPEN"]
                portfolio.buy_stock(ticker.ticker, price)

            elif signal == -1:
                price = df.loc[date, "OPEN"]
                portfolio.sell_stock(ticker.ticker, price)

            close_price = df.loc[date, "CLOSE"]
            portfolio_value = portfolio.get_value(ticker.ticker, close_price)
            results.append({"Date": date, strategy.name: portfolio_value})

        return pd.DataFrame(results).set_index("Date"), strategy.name

    def quick_summary(self):
        summary_data = {}

        for result, name, ticker in self.results:
            if name not in summary_data:
                summary_data[name] = {"Strategy": name}

            performance = (
                result[name].iloc[-1] / ticker.get_dataframe()["CLOSE"].iloc[-1]
            )

            summary_data[name][ticker.ticker] = performance

        df = pd.DataFrame(list(summary_data.values())).set_index("Strategy")
        df["mean"] = df.mean(axis=1)
        df = df.sort_values("mean")

        print("\n\n=====QUICK-SUMMARY=====\n\n")
        print(df.to_string())
        return df

    def plot_results(self, show_indicators=False, log_scale=False):
        df = self.tickers[0].get_dataframe()
        used_indicators = self.tickers[0].get_used_indicators()

        # Determine if we need a secondary plot
        # Logic: If show_indicators is True and there's at least one non-overlay indicator
        from Ticker import CHART_OVERLAYS

        secondary_inds = [
            ind for ind in used_indicators if ind.split(":")[0] not in CHART_OVERLAYS
        ]
        has_secondary = show_indicators and len(secondary_inds) > 0

        # Create subplots
        n_rows = 2 if has_secondary else 1
        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=1,
            sharex=True,
            figsize=(14, 10 if has_secondary else 7),
            gridspec_kw={"height_ratios": [3, 1] if has_secondary else [1]},
        )

        # Ensure axes is always a list/array for easy indexing
        if n_rows == 1:
            ax1 = axes
            ax2 = (
                axes  # Fallback: secondary logic will be skipped by 'if has_secondary'
            )
        else:
            ax1, ax2 = axes

        # --- PRIMARY PLOT (ax1) ---
        ax1.plot(df.index, df["CLOSE"], label="Price", linewidth=1.5, color="black")

        for result, strat_name, ticker in self.results:
            ax1.plot(
                result.index,
                result[strat_name],
                label=f"Portfolio: {strat_name}",
                linewidth=1,
            )

        if show_indicators:
            for indicator in used_indicators:
                prefix = indicator.split(":")[0]
                if prefix in CHART_OVERLAYS:
                    ax1.plot(
                        df.index, df[indicator], label=f"Ind: {indicator}", alpha=0.7
                    )

        # --- SECONDARY PLOT (ax2) ---
        if has_secondary:
            for indicator in secondary_inds:
                ax2.plot(df.index, df[indicator], label=f"Oscillator: {indicator}")
            # Helpful for MACD/MOM
            ax2.axhline(0, color="black", lw=0.5, ls="--")
            ax2.set_ylabel("Value")
            ax2.legend(loc="upper left")
            ax2.grid(True, alpha=0.3)

        from matplotlib.widgets import MultiCursor

        cursor = MultiCursor(
            fig.canvas, (ax1, ax2), color="r", lw=0.5, horizOn=True, vertOn=True
        )
        # Formatting
        ax1.set_title(f"Simulation: {self.tickers[0].ticker}")
        ax1.set_ylabel("Price (USD)")
        if log_scale:
            ax1.set_yscale("log")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
