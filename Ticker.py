from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

INDICATOR_DEPENDENCIES = {
    # leaf indicators
    "SMA_5": [],
    "SMA_10": [],
    "SMA_20": [],
    "SMA_50": [],
    "SMA_100": [],
    "SMA_200": [],
    "EMA_5": [],
    "EMA_10": [],
    "EMA_12": [],
    "EMA_20": [],
    "EMA_26": [],
    "EMA_50": [],
    "WMA_10": [],
    "WMA_20": [],
    "DEMA_10": [],
    "DEMA_20": [],
    "TEMA_10": [],
    "TEMA_20": [],
    "TRIMA_20": [],
    "KAMA_20": [],
    "T3_5": [],
    # RSI
    "RSI_7": [],
    "RSI_14": [],
    "RSI_21": [],
    # MACD family
    "MACD": ["EMA_12", "EMA_26"],
    "MACD_SIGNAL": ["MACD"],
    "MACD_HIST": ["MACD", "MACD_SIGNAL"],
    # Stochastic
    "STOCH_FASTK": [],
    "STOCH_FASTD": ["STOCH_FASTK"],
    "STOCH_SLOWK": ["STOCH_FASTK"],
    "STOCH_SLOWD": ["STOCH_SLOWK"],
    # Stochastic RSI
    "STOCHRSI_FASTK": ["RSI_14"],
    "STOCHRSI_FASTD": ["STOCHRSI_FASTK"],
    # CCI
    "CCI_14": [],
    "CCI_20": [],
    # CMO
    "CMO_14": [],
    # Momentum
    "MOM_10": [],
    # Rate of Change
    "ROC_10": [],
    # Williams %R
    "WILLR_14": [],
    # PPO / APO
    "PPO": ["EMA_12", "EMA_26"],
    "APO": ["EMA_12", "EMA_26"],
    # Balance of Power
    "BOP": [],
    # Ultimate Oscillator
    "ULTOSC": [],
    # Volume indicators
    "AD": [],
    "ADOSC": ["AD"],
    "OBV": [],
    "MFI_14": [],
    # Volatility / price-derived
    "TYPPRICE": [],
    "TRANGE": [],
    "ATR_14": ["TRANGE"],
    "NATR_14": ["ATR_14"],
    # Bollinger Bands
    "BB_MIDDLE": [],
    "BB_UPPER": ["BB_MIDDLE"],
    "BB_LOWER": ["BB_MIDDLE"],
    # Directional Movement (building blocks)
    "PLUS_DM": [],
    "MINUS_DM": [],
    "PLUS_DI_14": ["PLUS_DM", "TRANGE"],
    "MINUS_DI_14": ["MINUS_DM", "TRANGE"],
    # ADX
    "ADX_14": ["PLUS_DI_14", "MINUS_DI_14"],
    # Aroon
    "AROON_UP": [],
    "AROON_DOWN": [],
    "AROON_OSC": ["AROON_UP", "AROON_DOWN"],
    # Parabolic SAR
    "SAR": [],
}


@dataclass
class Ticker:
    ticker: str
    name: str = None
    desc: str = None

    def __init__(self, ticker, name=None, desc=None):
        self.ticker = ticker
        self.name = name
        self.desc = desc
        self.df = None

        self.is_loaded = False
        self.can_load_ticker = True

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

    def get_dataframe(self):
        return self.df

    def get_available_indicator(self):
        indicator = self._indicator_registry()
        return indicator.keys()

    def history_from_file(self):
        try:
            self.df = pd.read_csv("data/ticker/history/" + self.ticker + ".csv")
            print("Loaded Ticker {self.ticker} from file!")
            return True
        except Exception:
            print(f"Filed to read {self.ticker} from file!")
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
            self.save_to_file()
            return True

        except Exception as e:
            print(f"Download failed with error: {e}")
            return False

    def save_to_file(self):
        self.df.to_csv("data/ticker/history/" + self.ticker + ".csv")
        print(f"Saved {self.ticker}.csv")
        pass

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
        registry = self._indicator_registry()

        if indicator not in registry:
            raise ValueError(f"Unknown indicator: {indicator}")

        build_order = self._resolve_dependencies(indicator)

        for ind in build_order:
            if ind in self.df.columns and not force:
                continue

            try:
                registry[ind]()
                print("Calcualted Indicator: ", ind)
                return True
            except KeyError:
                raise ValueError(f"No build rule registered for {ind}")
                return False
        return False

    def add_indicators(self, indicators: list[str], force: bool = False):
        for ind in indicators:
            succes = self.add_indicator(ind, force=force)
            if not succes:
                return False
        return True

    def _indicator_registry(self):
        return {
            "SMA_5": self.add_sma_5,
            "SMA_10": self.add_sma_10,
            "SMA_20": self.add_sma_20,
            "SMA_50": self.add_sma_50,
            "SMA_100": self.add_sma_100,
            "SMA_200": self.add_sma_200,
            "EMA_5": self.add_ema_5,
            "EMA_10": self.add_ema_10,
            "EMA_12": self.add_ema_12,
            "EMA_20": self.add_ema_20,
            "EMA_26": self.add_ema_26,
            "EMA_50": self.add_ema_50,
            "WMA_10": self.add_wma_10,
            "WMA_20": self.add_wma_20,
            "DEMA_10": self.add_dema_10,
            "DEMA_20": self.add_dema_20,
            "TEMA_10": self.add_tema_10,
            "TEMA_20": self.add_tema_20,
            "TRIMA_20": self.add_trima_20,
            "KAMA_20": self.add_kama_20,
            "T3_5": self.add_t3_5,
            "RSI_7": self.add_rsi_7,
            "RSI_14": self.add_rsi_14,
            "RSI_21": self.add_rsi_21,
            "MACD": self.add_macd,
            "MACD_SIGNAL": self.add_macd_signal,
            "MACD_HIST": self.add_macd_hist,
            "STOCH_FASTK": self.add_stoch_fastk,
            "STOCH_FASTD": self.add_stoch_fastd,
            "STOCH_SLOWK": self.add_stoch_slowk,
            "STOCH_SLOWD": self.add_stoch_slowd,
            "STOCHRSI_FASTK": self.add_stochrsi_fastk,
            "STOCHRSI_FASTD": self.add_stochrsi_fastd,
            "CCI_14": self.add_cci_14,
            "CCI_20": self.add_cci_20,
            "CMO_14": self.add_cmo_14,
            "MOM_10": self.add_mom_10,
            "ROC_10": self.add_roc_10,
            "WILLR_14": self.add_willr_14,
            "PPO": self.add_ppo,
            "APO": self.add_apo,
            "BOP": self.add_bop,
            "ULTOSC": self.add_ultosc,
            "AD": self.add_ad,
            "ADOSC": self.add_adosc,
            "OBV": self.add_obv,
            "MFI_14": self.add_mfi_14,
            "TYPPRICE": self.add_typprice,
            "TRANGE": self.add_trange,
            "ATR_14": self.add_atr_14,
            "NATR_14": self.add_natr_14,
            "BB_MIDDLE": self.add_bb_middle,
            "BB_UPPER": self.add_bb_upper,
            "BB_LOWER": self.add_bb_lower,
            "PLUS_DM": self.add_plus_dm,
            "MINUS_DM": self.add_minus_dm,
            "PLUS_DI_14": self.add_plus_di_14,
            "MINUS_DI_14": self.add_minus_di_14,
            "ADX_14": self.add_adx_14,
            "AROON_UP": self.add_aroon_up,
            "AROON_DOWN": self.add_aroon_down,
            "AROON_OSC": self.add_aroon_osc,
            "SAR": self.add_sar,
        }

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
    def add_sma_5(self):
        self.df["SMA_5"] = self.df["CLOSE"].rolling(window=5).mean()

    def add_sma_10(self):
        self.df["SMA_10"] = self.df["CLOSE"].rolling(window=10).mean()

    def add_sma_20(self):
        self.df["SMA_20"] = self.df["CLOSE"].rolling(window=20).mean()

    def add_sma_50(self):
        self.df["SMA_50"] = self.df["CLOSE"].rolling(window=50).mean()

    def add_sma_100(self):
        self.df["SMA_100"] = self.df["CLOSE"].rolling(window=100).mean()

    def add_sma_200(self):
        self.df["SMA_200"] = self.df["CLOSE"].rolling(window=200).mean()

    # Exponential Moving Averages
    def add_ema_5(self):
        self.df["EMA_5"] = self.df["CLOSE"].ewm(span=5, adjust=False).mean()

    def add_ema_10(self):
        self.df["EMA_10"] = self.df["CLOSE"].ewm(span=10, adjust=False).mean()

    def add_ema_12(self):
        self.df["EMA_12"] = self.df["CLOSE"].ewm(span=12, adjust=False).mean()

    def add_ema_20(self):
        self.df["EMA_20"] = self.df["CLOSE"].ewm(span=20, adjust=False).mean()

    def add_ema_26(self):
        self.df["EMA_26"] = self.df["CLOSE"].ewm(span=26, adjust=False).mean()

    def add_ema_50(self):
        self.df["EMA_50"] = self.df["CLOSE"].ewm(span=50, adjust=False).mean()

    # Weighted Moving Averages
    def _wma(self, period):
        weights = np.arange(1, period + 1)
        return (
            self.df["CLOSE"]
            .rolling(period)
            .apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        )

    def add_wma_10(self):
        self.df["WMA_10"] = self._wma(10)

    def add_wma_20(self):
        self.df["WMA_20"] = self._wma(20)

    # Double Exponential Moving Average
    def _dema(self, period):
        ema = self.df["CLOSE"].ewm(span=period, adjust=False).mean()
        ema_ema = ema.ewm(span=period, adjust=False).mean()
        return 2 * ema - ema_ema

    def add_dema_10(self):
        self.df["DEMA_10"] = self._dema(10)

    def add_dema_20(self):
        self.df["DEMA_20"] = self._dema(20)

    # Triple Exponential Moving Average
    def _tema(self, period):
        ema1 = self.df["CLOSE"].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return 3 * (ema1 - ema2) + ema3

    def add_tema_10(self):
        self.df["TEMA_10"] = self._tema(10)

    def add_tema_20(self):
        self.df["TEMA_20"] = self._tema(20)

    # Triangular Moving Average
    def add_trima_20(self):
        self.df["TRIMA_20"] = (
            self.df["CLOSE"].rolling(window=20).mean().rolling(window=20).mean()
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

    def _rsi(self, period):
        delta = self.df["CLOSE"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def add_rsi_7(self):
        self.df["RSI_7"] = self._rsi(7)

    def add_rsi_14(self):
        self.df["RSI_14"] = self._rsi(14)

    def add_rsi_21(self):
        self.df["RSI_21"] = self._rsi(21)

    # Moving Average Convergence Divergence
    def add_macd(self):
        self.df["MACD"] = self.df["EMA_12"] - self.df["EMA_26"]

    def add_macd_signal(self):
        self.df["MACD_SIGNAL"] = self.df["MACD"].ewm(span=9, adjust=False).mean()

    def add_macd_hist(self):
        self.df["MACD_HIST"] = self.df["MACD"] - self.df["MACD_SIGNAL"]

    # Stochastik Socillator
    def add_stoch_fastk(self, period=14):
        low = self.df["LOW"].rolling(period).min()
        high = self.df["HIGH"].rolling(period).max()
        self.df["STOCH_FASTK"] = 100 * (self.df["CLOSE"] - low) / (high - low)

    def add_stoch_fastd(self):
        self.df["STOCH_FASTD"] = self.df["STOCH_FASTK"].rolling(3).mean()

    def add_stoch_slowk(self):
        self.df["STOCH_SLOWK"] = self.df["STOCH_FASTK"].rolling(3).mean()

    def add_stoch_slowd(self):
        self.df["STOCH_SLOWD"] = self.df["STOCH_SLOWK"].rolling(3).mean()

    # Stochastik RSI
    def add_stochrsi_fastk(self, period=14):
        rsi = self.df["RSI_14"]
        min_rsi = rsi.rolling(period).min()
        max_rsi = rsi.rolling(period).max()
        self.df["STOCHRSI_FASTK"] = 100 * (rsi - min_rsi) / (max_rsi - min_rsi)

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
        self.df["CMO_14"] = 100 * (gain - loss) / (gain + loss)

    # Momentum
    def add_mom_10(self):
        self.df["MOM_10"] = self.df["CLOSE"].diff(10)

    # Rate of Change
    def add_roc_10(self):
        self.df["ROC_10"] = self.df["CLOSE"].pct_change(10) * 100

    def add_willr_14(self):
        high = self.df["HIGH"].rolling(14).max()
        low = self.df["LOW"].rolling(14).min()
        self.df["WILLR_14"] = -100 * (high - self.df["CLOSE"]) / (high - low)

    # Percentage Price Oscillator
    def add_ppo(self):
        self.df["PPO"] = (
            100 * (self.df["EMA_12"] - self.df["EMA_26"]) / self.df["EMA_26"]
        )

    # Absolute Price Oscillator
    def add_apo(self):
        self.df["APO"] = self.df["EMA_12"] - self.df["EMA_26"]

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

        self.df["ULTOSC"] = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7

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
        self.df["MFI_14"] = 100 - (100 / (1 + mfr))

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
        self.df["NATR_14"] = 100 * self.df["ATR_14"] / self.df["CLOSE"]

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

        self.df["PLUS_DI_14"] = 100 * dm_smooth / tr_smooth

    def add_minus_di_14(self):
        tr = self.df["TRANGE"]
        minus_dm = self.df["MINUS_DM"]

        tr_smooth = tr.ewm(alpha=1 / 14, adjust=False).mean()
        dm_smooth = minus_dm.ewm(alpha=1 / 14, adjust=False).mean()

        self.df["MINUS_DI_14"] = 100 * dm_smooth / tr_smooth

    # Average Directional Index
    def add_adx_14(self):
        plus_di = self.df["PLUS_DI_14"]
        minus_di = self.df["MINUS_DI_14"]

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
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
        self.df["AROON_UP"] = 100 * (period - rolling_high_idx) / period

    def add_aroon_down(self, period=14):
        rolling_low_idx = (
            self.df["LOW"]
            .rolling(period)
            .apply(
                lambda x: period - 1 - np.argmin(x),
                raw=True,
            )
        )
        self.df["AROON_DOWN"] = 100 * (period - rolling_low_idx) / period

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
