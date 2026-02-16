import numpy as np
import pandas as pd


class Strategy:
    _DEPENDENCIES = []

    def __init__(self):
        self.name = None

    def get_dependencies(self):
        return self._DEPENDENCIES

    def evaluate(self, df):
        # Check if every dependency is available
        for dependency in self._DEPENDENCIES:
            if df.get(dependency) is None:
                print(f"Missing data for: {dependency}")
                return False, False

        mask_buy = self.is_buy_signal(df)
        mask_sell = self.is_sell_signal(df)
        signals_df = df[mask_buy | mask_sell].copy()

        conditions = [mask_buy[mask_buy | mask_sell], mask_sell[mask_buy | mask_sell]]
        choices = [1, -1]

        signals_df["signal"] = np.select(conditions, choices)

        return signals_df["signal"]

    def is_buy_signal(self):
        return None

    def is_sell_signal(self):
        return None


class BuyAndHold(Strategy):
    def __init__(self):
        self.name = "Buy and Hold"

    def is_buy_signal(self, df):
        return pd.Series(True, index=df.index)

    def is_sell_signal(self, df):
        return pd.Series(False, index=df.index)


class SMA_Cross(Strategy):
    def __init__(self, days_short=50, days_long=200):
        if days_short > days_long:
            tmp = days_long
            days_long = days_short
            days_short = tmp

        self.name = "SMA_CROSS S:" + str(days_short) + " L:" + str(days_long)

        self.sma_s = "SMA_CLOSE:" + str(days_short)
        self.sma_l = "SMA_CLOSE:" + str(days_long)
        self._DEPENDENCIES.append(self.sma_s)
        self._DEPENDENCIES.append(self.sma_l)

    def is_buy_signal(self, df):
        return (df[self.sma_s] > df[self.sma_l]) & (
            df[self.sma_s].shift(1) <= df[self.sma_l].shift(1)
        )

    def is_sell_signal(self, df):
        return (df[self.sma_s] < df[self.sma_l]) & (
            df[self.sma_s].shift(1) >= df[self.sma_l].shift(1)
        )


class EMA_Cross(Strategy):
    def __init__(self, short, long):
        if short > long:
            tmp = long
            long = short
            short = tmp

        self.name = "EMA_CROSS S:" + str(short) + " L:" + str(long)

        self.ema_s = "EMA:" + str(short)
        self.ema_l = "EMA:" + str(long)
        self._DEPENDENCIES.append(self.ema_s)
        self._DEPENDENCIES.append(self.ema_l)

    def is_buy_signal(self, df):
        return (df[self.ema_s] > df[self.ema_l]) & (
            df[self.ema_s].shift(1) <= df[self.ema_l].shift(1)
        )

    def is_sell_signal(self, df):
        return (df[self.ema_s] < df[self.ema_l]) & (
            df[self.ema_s].shift(1) >= df[self.ema_l].shift(1)
        )


# https://commodity.com/technical-analysis/momentum/


class MOM_ZeroCrossing(Strategy):
    _DEPENDENCIES = []

    def __init__(self, window):
        self.name = "MOM_ZeroCrossing W:" + str(window)
        self.mom = "MOM:" + str(window)
        self._DEPENDENCIES.append(self.mom)

    def is_buy_signal(self, df):
        return (df[self.mom] > 0) & (df[self.mom].shift(1) <= 0)

    def is_sell_signal(self, df):
        return (df[self.mom] < 0) & (df[self.mom].shift(1) >= 0)


class ADOSC_ZeroCrossing(Strategy):
    _DEPENDENCIES = []

    def __init__(self):
        self.name = "ADOSC_ZeroCrossing"
        self.ind = "ADOSC"
        self._DEPENDENCIES.append(self.ind)

    def is_buy_signal(self, df):
        return (df[self.ind] > 0) & (df[self.ind].shift(1) <= 0)

    def is_sell_signal(self, df):
        return (df[self.ind] < 0) & (df[self.ind].shift(1) >= 0)


class SMA_SLOPE_CHANGE(Strategy):
    _DEPENDENCIES = []

    def __init__(self, window=20, shift=3, threshold=0.1):
        self.name = (
            "SMA_SLOPE_CHANGE W:"
            + str(window)
            + " S:"
            + str(shift)
            + " TH:"
            + str(threshold)
        )
        self.mom_slope = "SMA_SLOPE:" + str(window) + "_" + str(shift)
        self.th = threshold
        self._DEPENDENCIES.append(self.mom_slope)

    def is_buy_signal(self, df):
        return (df[self.mom_slope] > self.th) & (df[self.mom_slope].shift(1) <= self.th)

    def is_sell_signal(self, df):
        return (df[self.mom_slope] < self.th) & (df[self.mom_slope].shift(1) >= self.th)


class EMA_SLOPE_CHANGE(Strategy):
    _DEPENDENCIES = []

    def __init__(self, window=20, shift=3, threshold=0.1):
        self.name = (
            "EMA_SLOPE_CHANGE W:"
            + str(window)
            + " S:"
            + str(shift)
            + " TH:"
            + str(threshold)
        )
        self.ema_slope = "EMA_SLOPE:" + str(window) + "_" + str(shift)
        self.th = threshold
        self._DEPENDENCIES.append(self.ema_slope)

    def is_buy_signal(self, df):
        return (df[self.ema_slope] > self.th) & (df[self.ema_slope].shift(1) <= self.th)

    def is_sell_signal(self, df):
        return (df[self.ema_slope] < self.th) & (df[self.ema_slope].shift(1) >= self.th)


"""
class (Strategy):
    _DEPENDENCIES = []

    def __init__(self):
        self.name = ""    

    def is_buy_signal(self, df):
        return False

    def is_sell_signal(self, df):
        return False
"""
