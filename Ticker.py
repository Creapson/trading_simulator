from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

# Define which indicators are "Overlays" (Price-scale)
# Everything else will default to "Oscillator" (Secondary-scale)
CHART_OVERLAYS = {
    "SMA_CLOSE",
    "SMA_HIGH",
    "SMA_LOW",
    "SMA_OPEN",
    "EMA",
    "WMA",
    "DEMA",
    "TEMA",
    "TRIMA",
    "KAMA_20",
    "T3_5",
    "BB_UPPER",
    "BB_LOWER",
    "BB_MIDDLE",
    "SAR",
}

INDICATOR_DEPENDENCIES = {
    # MACD family
    "MACD": ["EMA:12", "EMA:26"],
    "MACD_SIGNAL": ["MACD"],
    "MACD_HIST": ["MACD", "MACD_SIGNAL"],
    # Stochastic
    "STOCH_FASTD": ["STOCH_FASTK"],
    "STOCH_SLOWK": ["STOCH_FASTK"],
    "STOCH_SLOWD": ["STOCH_SLOWK"],
    # Stochastic RSI
    "STOCHRSI_FASTK": ["RSI:14"],
    "STOCHRSI_FASTD": ["STOCHRSI_FASTK"],
    # PPO / APO
    "PPO": ["EMA:12", "EMA:26"],
    "APO": ["EMA:12", "EMA:26"],
    # Volume indicators
    "ADOSC": ["AD"],
    # Volatility / price-derived
    "ATR_14": ["TRANGE"],
    "NATR_14": ["ATR_14"],
    # Bollinger Bands
    "BB_UPPER": ["BB_MIDDLE"],
    "BB_LOWER": ["BB_MIDDLE"],
    # Directional Movement (building blocks)
    "PLUS_DI_14": ["PLUS_DM", "TRANGE"],
    "MINUS_DI_14": ["MINUS_DM", "TRANGE"],
    # ADX
    "ADX_14": ["PLUS_DI_14", "MINUS_DI_14"],
    # Aroon
    "AROON_OSC": ["AROON_UP", "AROON_DOWN"],
}


@dataclass
class Ticker:
    ticker: str
    name: str = None
    desc: str = None

    def __init__(self, ticker, name=None, desc=None):
        self.yf_ticker = yf.Ticker(ticker)
        self.ticker = ticker
        self.name = name
        self.desc = desc
        self.df = None

        self.is_loaded = False
        self.can_load_ticker = True
        self.indicator_list = []

        self.load()

    def load(self):
        self.load_history()
        # self.load_financials()

    def load_history(self):
        # try loading_from_file
        self.is_loaded = self.history_from_file()

        if not self.is_loaded:
            self.is_loaded = self.history_from_yf()

        if not self.is_loaded:
            print("Failed to get Data for this Ticker!")
            self.can_load_ticker = False
        else:
            self.can_load_ticker = True

    def load_financials(self):
        print(self.yf_ticker.get_financials(freq="quarterly"))
        print(self.yf_ticker.get_fast_info())
        print(self.yf_ticker.get_earnings_history())

    def get_dataframe(self):
        return self.df

    def get_used_indicators(self):
        return self.indicator_list

    def set_timespan(self, start_time=None, end_time=None):
        try:
            if start_time is None and end_time is None:
                return
            elif end_time is None:
                self.df = self.df.loc[start_time:]
            else:
                self.df = self.df.loc[:end_time]
        except Exception:
            print("Failed to slice Ticker!")

    def history_from_file(self):
        try:
            self.df = pd.read_csv(
                "data/ticker/history/" + self.ticker + ".csv",
                parse_dates=["Date"],
                index_col="Date",
            )
            print(f"Loaded Ticker {self.ticker} from file!")
            return True
        except Exception:
            print(f"Failed to read {self.ticker} from file!")
            return False

    def history_from_yf(self):
        print("Downloading Ticker from Yahoo-Finance!")
        try:
            self.df = yf.download(self.ticker, period="max", interval="1d")

            # Download failed or returned no data
            if self.df is None or self.df.empty:
                print("Download failed: no data returned.")
                return False

            # 1️⃣ Flatten columns
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = self.df.columns.get_level_values(0)

            self.df.columns = self.df.columns.str.upper()

            self.df.index = pd.to_datetime(self.df.index)
            self.df = self.df.sort_index()

            self.df.to_csv("data/ticker/history/" + self.ticker + ".csv")
            print(f"Saved {self.ticker}.csv")
            return True

        except Exception as e:
            print(f"Download failed with error: {e}")
            return False

    def print_info(self):
        # Display basic info
        print(f"Dataset shape: {self.df.shape}")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
        print(f"\nFirst few rows:\n{self.df}")

        print(self.df.columns)

    def add_indicator(self, indicator: str, force: bool = False):
        if not self.can_load_ticker:
            return False
        if not self.is_loaded:
            self.load_history()

        indicator = indicator.upper()

        build_order = self._resolve_dependencies(indicator)

        for ind in build_order:
            if ind in self.df.columns and not force:
                continue

            try:
                self.calc_indicator(ind)
                # print("Calcualted Indicator: ", ind)
                self.indicator_list.append(ind)
            except KeyError:
                raise ValueError(f"No build rule registered for {ind}")
                return False

        return True

    def add_indicators(self, indicators: list[str], force: bool = False):
        for ind in indicators:
            succes = self.add_indicator(ind, force=force)
            if not succes:
                return False
        return True

    def dropna(self):
        self.df.dropna()

    def _indicator_demux(self, indicator):
        pass

    def calc_indicator(self, indicator_name: str):
        ind_split = indicator_name.split(":")
        ind_type = ind_split[0]

        if len(ind_split) > 1:
            ind_params = [int(ind_param) for ind_param in ind_split[1].split("_")]

        match ind_type:
            # Simple Moving Averages
            case "SMA_CLOSE":
                return self.add_sma_close(ind_params[0])
            case "SMA_HIGH":
                return self.add_sma_high(ind_params[0])
            case "SMA_LOW":
                return self.add_sma_low(ind_params[0])
            case "SMA_OPEN":
                return self.add_sma_open(ind_params[0])
            case "SMA_SLOPE":
                return self.add_sma_slope(ind_params[0], ind_params[1])

            # Exponential Moving Averages
            case "EMA":
                return self.add_ema(ind_params[0])
            case "EMA_SLOPE":
                return self.add_ema_slope(ind_params[0], ind_params[1])
            # Weighted & Double/Triple EMA
            case "WMA":
                return self.add_wma(ind_params[0])
            case "DEMA":
                return self.add_dema(ind_params[0])
            case "TEMA":
                return self.add_tema(ind_params[0])
            case "TRIMA":
                return self.add_trima(ind_params[0])
            case "KAMA_20":
                return self.add_kama_20()
            case "T3_5":
                return self.add_t3_5()

            # Momentum & RSI
            case "RSI":
                return self.add_rsi(ind_params[0])
            case "MOM":
                return self.add_mom(ind_params[0])
            case "ROC":
                return self.add_roc(ind_params[0])

            # MACD
            case "MACD":
                return self.add_macd()
            case "MACD_SIGNAL":
                return self.add_macd_signal()
            case "MACD_HIST":
                return self.add_macd_hist()

            # Stochastics
            case "STOCH_FASTK":
                return self.add_stoch_fastk()
            case "STOCH_FASTD":
                return self.add_stoch_fastd()
            case "STOCH_SLOWK":
                return self.add_stoch_slowk()
            case "STOCH_SLOWD":
                return self.add_stoch_slowd()
            case "STOCHRSI_FASTK":
                return self.add_stochrsi_fastk()
            case "STOCHRSI_FASTD":
                return self.add_stochrsi_fastd()

            # Commodity & Volatility
            case "CCI_14":
                return self.add_cci_14()
            case "CCI_20":
                return self.add_cci_20()
            case "CMO_14":
                return self.add_cmo_14()
            case "WILLR_14":
                return self.add_willr_14()
            case "PPO":
                return self.add_ppo()
            case "APO":
                return self.add_apo()
            case "BOP":
                return self.add_bop()
            case "ULTOSC":
                return self.add_ultosc()

            # Volume Indicators
            case "AD":
                return self.add_ad()
            case "ADOSC":
                return self.add_adosc()
            case "OBV":
                return self.add_obv()
            case "MFI_14":
                return self.add_mfi_14()

            # Price & Volatility (ATR / Bollinger)
            case "TYPPRICE":
                return self.add_typprice()
            case "TRANGE":
                return self.add_trange()
            case "ATR_14":
                return self.add_atr_14()
            case "NATR_14":
                return self.add_natr_14()
            case "BB_MIDDLE":
                return self.add_bb_middle()
            case "BB_UPPER":
                return self.add_bb_upper()
            case "BB_LOWER":
                return self.add_bb_lower()

            # Trend Indicators (ADX / Aroon / SAR)
            case "PLUS_DM":
                return self.add_plus_dm()
            case "MINUS_DM":
                return self.add_minus_dm()
            case "PLUS_DI_14":
                return self.add_plus_di_14()
            case "MINUS_DI_14":
                return self.add_minus_di_14()
            case "ADX_14":
                return self.add_adx_14()
            case "AROON_UP":
                return self.add_aroon_up()
            case "AROON_DOWN":
                return self.add_aroon_down()
            case "AROON_OSC":
                return self.add_aroon_osc()
            case "SAR":
                return self.add_sar()

            # Fallback
            case _:
                raise ValueError(f"Indicator {indicator_name} not recognized.")

    def _resolve_dependencies(self, indicator: str, resolved=None, seen=None):
        if resolved is None:
            resolved = []
        if seen is None:
            seen = set()

        if indicator in seen:
            raise ValueError(f"Circular dependency detected: {indicator}")

        seen.add(indicator)

        deps = INDICATOR_DEPENDENCIES.get(indicator, [])
        for dep in deps:
            if dep not in resolved:
                self._resolve_dependencies(dep, resolved, seen)

        if indicator not in resolved:
            resolved.append(indicator)

        return resolved

    ##############
    # INDICATORS #
    ##############

    # MOVING AVERAGES

    # Simple Moving Averages
    def add_sma_close(self, days):
        self.df["SMA_CLOSE:" + str(days)] = self.df["CLOSE"].rolling(window=days).mean()

    def add_sma_high(self, days):
        self.df["SMA_HIGH:" + str(days)] = self.df["HIGH"].rolling(window=days).mean()

    def add_sma_low(self, days):
        self.df["SMA_LOW:" + str(days)] = self.df["LOW"].rolling(window=days).mean()

    def add_sma_open(self, days):
        self.df["SMA_OPEN:" + str(days)] = self.df["OPEN"].rolling(window=days).mean()

    def add_sma_slope(self, days, shift):
        sma = "SMA_CLOSE:" + str(days)
        self.add_indicator(sma)
        sma_slope = "SMA_SLOPE:" + str(days) + "_" + str(shift)
        self.df[sma_slope] = ((self.df[sma] / self.df[sma].shift(shift)) - 1) * days

    # Exponential Moving Averages
    def add_ema(self, days):
        self.df["EMA:" + str(days)] = (
            self.df["CLOSE"].ewm(span=days, adjust=False).mean()
        )

    def add_ema_slope(self, days, shift):
        ema = "EMA:" + str(days)
        self.add_indicator(ema)
        ema_slope = "EMA_SLOPE:" + str(days) + "_" + str(shift)
        self.df[ema_slope] = ((self.df[ema] / self.df[ema].shift(shift)) - 1) * days

    # Weighted Moving Averages
    def add_wma(self, period):
        weights = np.arange(1, period + 1)
        self.df["WMA:" + str(period)] = (
            self.df["CLOSE"]
            .rolling(period)
            .apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        )

    # Double Exponential Moving Average
    def add_dema(self, period):
        ema = self.df["CLOSE"].ewm(span=period, adjust=False).mean()
        ema_ema = ema.ewm(span=period, adjust=False).mean()
        self.df["DEMA:" + str(period)] = 2 * ema - ema_ema

    # Triple Exponential Moving Average

    def add_tema(self, period):
        ema1 = self.df["CLOSE"].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        self.df["TEMA:" + str(period)] = 3 * (ema1 - ema2) + ema3

    # Triangular Moving Average
    def add_trima(self, period):
        self.df["TRIMA:" + str(period)] = (
            self.df["CLOSE"].rolling(window=period).mean().rolling(window=period).mean()
        )

    # Kaufman Adaptive Moving Average
    def add_kama_20(self, fast=2, slow=30):
        close = self.df["CLOSE"]

        change = close.diff(20).abs()
        volatility = close.diff().abs().rolling(20).sum()
        er = change / volatility

        fast_sc = 2 / (fast + 1)
        slow_sc = 2 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        kama = np.zeros(len(close))
        kama[:20] = close.iloc[:20]

        for i in range(20, len(close)):
            kama[i] = kama[i - 1] + sc.iloc[i] * (close.iloc[i] - kama[i - 1])

        self.df["KAMA_20"] = kama

    # T3 Moving Average
    def add_t3_5(self, v=0.7):
        close = self.df["CLOSE"]

        e1 = close.ewm(span=5, adjust=False).mean()
        e2 = e1.ewm(span=5, adjust=False).mean()
        e3 = e2.ewm(span=5, adjust=False).mean()
        e4 = e3.ewm(span=5, adjust=False).mean()
        e5 = e4.ewm(span=5, adjust=False).mean()
        e6 = e5.ewm(span=5, adjust=False).mean()

        c1 = -(v**3)
        c2 = 3 * v**2 + 3 * v**3
        c3 = -6 * v**2 - 3 * v - 3 * v**3
        c4 = 1 + 3 * v + v**2 + v**3

        self.df["T3_5"] = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    # MOMENTUM INDICATORS

    # Relative Strength Index

    def add_rsi(self, period):
        delta = self.df["CLOSE"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss
        self.df["RSI:" + str(period)] = 1 - (1 / (1 + rs))

    # Momentum
    def add_mom(self, past):
        self.df["MOM:" + str(past)] = self.df["CLOSE"].diff(past)

    # Rate of Change
    def add_roc(self, past):
        self.df["ROC:" + str(past)] = self.df["CLOSE"].pct_change(past)

    # Moving Average Convergence Divergence
    def add_macd(self):
        self.df["MACD"] = self.df["EMA:12"] - self.df["EMA:26"]

    def add_macd_signal(self):
        self.df["MACD_SIGNAL"] = self.df["MACD"].ewm(span=9, adjust=False).mean()

    def add_macd_hist(self):
        self.df["MACD_HIST"] = self.df["MACD"] - self.df["MACD_SIGNAL"]

    # Stochastik Socillator
    def add_stoch_fastk(self, period=14):
        low = self.df["LOW"].rolling(period).min()
        high = self.df["HIGH"].rolling(period).max()
        self.df["STOCH_FASTK"] = 1 * (self.df["CLOSE"] - low) / (high - low)

    def add_stoch_fastd(self):
        self.df["STOCH_FASTD"] = self.df["STOCH_FASTK"].rolling(3).mean()

    def add_stoch_slowk(self):
        self.df["STOCH_SLOWK"] = self.df["STOCH_FASTK"].rolling(3).mean()

    def add_stoch_slowd(self):
        self.df["STOCH_SLOWD"] = self.df["STOCH_SLOWK"].rolling(3).mean()

    # Stochastik RSI
    def add_stochrsi_fastk(self, period=14):
        rsi = self.df["RSI:14"]
        min_rsi = rsi.rolling(period).min()
        max_rsi = rsi.rolling(period).max()
        self.df["STOCHRSI_FASTK"] = 1 * (rsi - min_rsi) / (max_rsi - min_rsi)

    def add_stochrsi_fastd(self):
        self.df["STOCHRSI_FASTD"] = self.df["STOCHRSI_FASTK"].rolling(3).mean()

    # Commodity Channel Index
    def _cci(self, period):
        tp = (self.df["HIGH"] + self.df["LOW"] + self.df["CLOSE"]) / 3
        sma = tp.rolling(period).mean()
        mad = (tp - sma).abs().rolling(period).mean()
        return (tp - sma) / (0.015 * mad)

    def add_cci_14(self):
        self.df["CCI_14"] = self._cci(14)

    def add_cci_20(self):
        self.df["CCI_20"] = self._cci(20)

    # Chande Momentum Oscillator
    def add_cmo_14(self):
        delta = self.df["CLOSE"].diff()
        gain = delta.clip(lower=0).rolling(14).sum()
        loss = -delta.clip(upper=0).rolling(14).sum()
        self.df["CMO_14"] = 1 * (gain - loss) / (gain + loss)

    def add_willr_14(self):
        high = self.df["HIGH"].rolling(14).max()
        low = self.df["LOW"].rolling(14).min()
        self.df["WILLR_14"] = -1 * (high - self.df["CLOSE"]) / (high - low)

    # Percentage Price Oscillator
    def add_ppo(self):
        self.df["PPO"] = 1 * (self.df["EMA:12"] - self.df["EMA:26"]) / self.df["EMA:26"]

    # Absolute Price Oscillator
    def add_apo(self):
        self.df["APO"] = self.df["EMA:12"] - self.df["EMA:26"]

    # Balance of Power
    def add_bop(self):
        self.df["BOP"] = (self.df["CLOSE"] - self.df["OPEN"]) / (
            self.df["HIGH"] - self.df["LOW"]
        )

    # Ultimate Oscillator
    def add_ultosc(self):
        close = self.df["CLOSE"]
        low = self.df["LOW"]
        high = self.df["HIGH"]

        bp = close - np.minimum(low, close.shift(1))
        tr = np.maximum(high, close.shift(1)) - np.minimum(low, close.shift(1))

        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()

        self.df["ULTOSC"] = 1 * (4 * avg7 + 2 * avg14 + avg28) / 7

    # VOLUME INDUCATOR

    # Accumulation/Distribution
    def add_ad(self):
        high = self.df["HIGH"]
        low = self.df["LOW"]
        close = self.df["CLOSE"]
        volume = self.df["VOLUME"]

        mfm = ((close - low) - (high - close)) / (high - low)
        mfm = mfm.replace([np.inf, -np.inf], 0).fillna(0)

        mfv = mfm * volume
        self.df["AD"] = mfv.cumsum()

    # Chaikin A/D Oscillator
    def add_adosc(self, fast=3, slow=10):
        ad = self.df["AD"]
        ema_fast = ad.ewm(span=fast, adjust=False).mean()
        ema_slow = ad.ewm(span=slow, adjust=False).mean()
        self.df["ADOSC"] = ema_fast - ema_slow

    # On-Balance Volume
    def add_obv(self):
        direction = np.sign(self.df["CLOSE"].diff()).fillna(0)
        self.df["OBV"] = (direction * self.df["VOLUME"]).cumsum()

    # Money Flow Index
    def add_mfi_14(self):
        high = self.df["HIGH"]
        low = self.df["LOW"]
        close = self.df["CLOSE"]
        volume = self.df["VOLUME"]

        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume

        direction = typical_price.diff()
        positive_flow = money_flow.where(direction > 0, 0.0)
        negative_flow = money_flow.where(direction < 0, 0.0)

        pos_sum = positive_flow.rolling(14).sum()
        neg_sum = negative_flow.rolling(14).sum()

        mfr = pos_sum / neg_sum
        self.df["MFI_14"] = 1 - (1 / (1 + mfr))

    # VOLATILITY INDICATORS
    # Typical Price
    def add_typprice(self):
        self.df["TYPPRICE"] = (self.df["HIGH"] + self.df["LOW"] + self.df["CLOSE"]) / 3

    # True Range
    def add_trange(self):
        high = self.df["HIGH"]
        low = self.df["LOW"]
        prev_close = self.df["CLOSE"].shift(1)

        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        self.df["TRANGE"] = tr

    # Average True Range
    def add_atr_14(self):
        tr = self.df["TRANGE"]
        self.df["ATR_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    # Normalized ATR
    def add_natr_14(self):
        self.df["NATR_14"] = 1 * self.df["ATR_14"] / self.df["CLOSE"]

    # Bollinger Bands
    def add_bb_middle(self):
        self.df["BB_MIDDLE"] = self.df["CLOSE"].rolling(20).mean()

    def add_bb_upper(self, std_mult=2):
        std = self.df["CLOSE"].rolling(20).std()
        self.df["BB_UPPER"] = self.df["BB_MIDDLE"] + std_mult * std

    def add_bb_lower(self, std_mult=2):
        std = self.df["CLOSE"].rolling(20).std()
        self.df["BB_LOWER"] = self.df["BB_MIDDLE"] - std_mult * std

    # TREND INDICATOR

    # Directional Indicators
    def add_plus_dm(self):
        up_move = self.df["HIGH"].diff()
        down_move = -self.df["LOW"].diff()

        self.df["PLUS_DM"] = np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0.0,
        )

    def add_minus_dm(self):
        up_move = self.df["HIGH"].diff()
        down_move = -self.df["LOW"].diff()

        self.df["MINUS_DM"] = np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0.0,
        )

    def add_plus_di_14(self):
        tr = self.df["TRANGE"]
        plus_dm = self.df["PLUS_DM"]

        tr_smooth = tr.ewm(alpha=1 / 14, adjust=False).mean()
        dm_smooth = plus_dm.ewm(alpha=1 / 14, adjust=False).mean()

        self.df["PLUS_DI_14"] = 1 * dm_smooth / tr_smooth

    def add_minus_di_14(self):
        tr = self.df["TRANGE"]
        minus_dm = self.df["MINUS_DM"]

        tr_smooth = tr.ewm(alpha=1 / 14, adjust=False).mean()
        dm_smooth = minus_dm.ewm(alpha=1 / 14, adjust=False).mean()

        self.df["MINUS_DI_14"] = 1 * dm_smooth / tr_smooth

    # Average Directional Index
    def add_adx_14(self):
        plus_di = self.df["PLUS_DI_14"]
        minus_di = self.df["MINUS_DI_14"]

        dx = 1 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        self.df["ADX_14"] = dx.ewm(alpha=1 / 14, adjust=False).mean()

    # Aroon Indicator
    def add_aroon_up(self, period=14):
        rolling_high_idx = (
            self.df["HIGH"]
            .rolling(period)
            .apply(
                lambda x: period - 1 - np.argmax(x),
                raw=True,
            )
        )
        self.df["AROON_UP"] = 1 * (period - rolling_high_idx) / period

    def add_aroon_down(self, period=14):
        rolling_low_idx = (
            self.df["LOW"]
            .rolling(period)
            .apply(
                lambda x: period - 1 - np.argmin(x),
                raw=True,
            )
        )
        self.df["AROON_DOWN"] = 1 * (period - rolling_low_idx) / period

    def add_aroon_osc(self):
        self.df["AROON_OSC"] = self.df["AROON_UP"] - self.df["AROON_DOWN"]

    # Parabolic SAR
    def add_sar(self, af_step=0.02, af_max=0.2):
        high = self.df["HIGH"].values
        low = self.df["LOW"].values

        sar = np.zeros(len(self.df))
        trend = 1  # 1 = uptrend, -1 = downtrend
        af = af_step
        ep = low[0]

        sar[0] = low[0]

        for i in range(1, len(self.df)):
            sar[i] = sar[i - 1] + af * (ep - sar[i - 1])

            if trend == 1:
                sar[i] = min(sar[i], low[i - 1], low[i])
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
                if low[i] < sar[i]:
                    trend = -1
                    sar[i] = ep
                    ep = low[i]
                    af = af_step
            else:
                sar[i] = max(sar[i], high[i - 1], high[i])
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
                if high[i] > sar[i]:
                    trend = 1
                    sar[i] = ep
                    ep = high[i]
                    af = af_step

        self.df["SAR"] = sar
