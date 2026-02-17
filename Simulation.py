import time

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
        self.summary_data = {}

    def calc_indicators(self, ticker):
        indicators = set()
        for strategy in self.strategys:
            indicators.update(strategy.get_dependencies())
        indicator_list = list(indicators)
        success = ticker.add_indicators(indicator_list)
        ticker.dropna()
        return success

    def set_timespan(self, start, end):
        self.ticker.set_timespan(start_time=start)

    def start(self, show_progress=True):
        def printProgressBar(
            iteration,
            total,
            start_time,  # Added start_time parameter
            prefix="",
            suffix="",
            decimals=1,
            length=100,
            fill="█",
            printEnd="\r",
        ):
            percent = ("{0:." + str(decimals) + "f}").format(
                100 * (iteration / float(total))
            )
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + "-" * (length - filledLength)

            # --- Time Calculation ---
            elapsed_time = time.time() - start_time
            avg_time_per_iteration = elapsed_time / iteration
            remaining_iterations = total - iteration
            eta_seconds = int(remaining_iterations * avg_time_per_iteration)

            # Format seconds into M:S or H:M:S
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
            # ------------------------

            print(
                f"\r{prefix} |{bar}| {percent}% {suffix} | ETA: {eta_str}", end=printEnd
            )

            if iteration == total:
                print()

        self.results = []
        start_time = time.time()  # Capture start time

        # precalc all indicators for the tickers
        for ticker in self.tickers:
            print(f"Calculating indicators for: {ticker.ticker}")
            loaded = self.calc_indicators(ticker)
            if not loaded:
                print(f"\nTicker {ticker} not loaded properly. Skipping...")
                return

        # counter
        i = 0
        for strategy in self.strategys:
            results = []
            for ticker in self.tickers:
                if not show_progress:
                    print("Started Simulator for: ", strategy.name)
                result, strat_name = self.simulate(strategy, ticker)
                results.append((result, strat_name, ticker))
                if len(self.tickers) == 1:
                    self.results.append((result, strat_name, ticker))
                i += 1

                if show_progress and i % 25 == 0:
                    printProgressBar(
                        i,
                        len(self.strategys) * len(self.tickers),
                        start_time,  # Pass start time here
                        prefix="Progress:",
                        suffix="Complete",
                        length=50,
                    )

            # interpret all the results for this ticker
            avg_annual_return = 0
            avg_annual_sharp_ratio = 0
            avg_max_drawdown = 0
            for result, name, _ in results:
                # Annual return
                avg_annual_return += self.get_annual_return(result)
                # Annual Sharp Ratio
                avg_annual_sharp_ratio += self.get_annual_sharpe_ratio(result)
                # Drawdown
                avg_max_drawdown += self.get_max_drawdown(result)

            num_results = len(results)
            avg_annual_return = avg_annual_return / num_results
            avg_annual_sharp_ratio = avg_annual_sharp_ratio / num_results
            avg_max_drawdown = avg_max_drawdown / num_results

            self.add_summary(
                name, avg_annual_return, avg_annual_sharp_ratio, avg_max_drawdown
            )

    def simulate(self, strategy, ticker):
        df = ticker.get_dataframe()
        if df.empty:
            return pd.Series(dtype="float64"), strategy.name

        signals = strategy.evaluate(df)
        signals = signals.reindex(df.index).fillna(0).shift(1).to_numpy()

        open_prices = df["OPEN"].to_numpy()
        close_prices = df["CLOSE"].to_numpy()
        dates = df.index

        portfolio = Portfolio(cash=close_prices[0])

        values = []

        for i in range(len(df)):
            signal = signals[i]

            if signal == 1:
                portfolio.buy_stock(ticker.ticker, open_prices[i])
            elif signal == -1:
                portfolio.sell_stock(ticker.ticker, open_prices[i])

            portfolio_value = portfolio.get_value(ticker.ticker, close_prices[i])
            values.append(portfolio_value)

        result = pd.DataFrame({"Value": values}, index=dates)
        return result, strategy.name

    def get_annual_return(
        self,
        result,
    ):
        if result.empty or len(result) < 2:
            return 0
        total_days = (result.index.max() - result.index.min()).days
        num_years = total_days / 365.25

        start_price = result["Value"].iloc[0]
        end_price = result["Value"].iloc[-1]

        annual_return = (end_price / start_price) ** (1 / num_years) - 1

        return annual_return

    def get_annual_sharpe_ratio(
        self,
        result,
        risk_free_rate: float = 0.0,
        trading_days: int = 252,
    ):
        if result.empty or len(result) < 2:
            return 0

        daily_returns = result["Value"].pct_change().dropna()

        if daily_returns.std() == 0:
            return 0

        daily_rf = (1 + risk_free_rate) ** (1 / trading_days) - 1

        excess_returns = daily_returns - daily_rf

        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * (
            trading_days**0.5
        )

        return sharpe_ratio

    def get_max_drawdown(self, result):
        if result.empty:
            return 0.0

        rolling_max = result["Value"].cummax()
        drawdowns = (result["Value"] / rolling_max) - 1.0
        max_drawdown = drawdowns.min()

        return max_drawdown

    def add_summary(
        self,
        name,
        avg_annual_return=None,
        avg_annual_sharp_ratio=None,
        max_drawdown=None,
    ):
        if name not in self.summary_data:
            self.summary_data[name] = {"Strategy": name}

        self.summary_data[name]["Average Annual Return"] = avg_annual_return
        self.summary_data[name]["Average Sharp Ratio"] = avg_annual_sharp_ratio
        self.summary_data[name]["Max Drawdown"] = max_drawdown

    def get_quick_summary(self):
        df = pd.DataFrame(list(self.summary_data.values())).set_index("Strategy")
        df = df.sort_values("Average Annual Return")

        print("\n\n=====QUICK-SUMMARY=====")
        print(df.to_string())
        return df

    def plot_results(self, show_indicators=False, log_scale=False, show_volume=False):
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

        for i, (result, strat_name, ticker) in enumerate(self.results):
            ax1.plot(
                result.index,
                result,
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
