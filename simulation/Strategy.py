import numpy as np
import pandas as pd


class Strategy:

    def __init__(self):
        self.name = None
        self._DEPENDENCIES = []

    def get_dependencies(self):
        return self._DEPENDENCIES

    def evaluate(self, df):
        # Check if every dependency is available
        for dependency in self._DEPENDENCIES:
            if df.get(dependency) is None:
                print(f"Missing data for: {dependency}")
                print(df)
                return False

        mask_buy = self.is_long_signal(df)
        mask_sell = self.is_short_signal(df)

        mask_buy = self.is_long_signal(df)
        mask_sell = self.is_short_signal(df)

        signals = pd.Series(0, index=df.index)
        signals.loc[mask_buy] = 1
        signals.loc[mask_sell] = -1

        return signals

    def is_long_signal(self):
        pass

    def is_short_signal(self):
        pass


class HoldAndReinvest(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "HoldAndReinvest"

    def is_long_signal(self, df):
        return pd.Series(True, index=df.index)

    def is_short_signal(self, df):
        return pd.Series(False, index=df.index)


class SMA_Cross(Strategy):
    def __init__(self, days_short=50, days_long=200):
        super().__init__()
        if days_short > days_long:
            tmp = days_long
            days_long = days_short
            days_short = tmp

        self.name = "SMA_CROSS S:" + str(days_short) + " L:" + str(days_long)

        self.sma_s = "SMA_CLOSE:" + str(days_short)
        self.sma_l = "SMA_CLOSE:" + str(days_long)
        self._DEPENDENCIES.append(self.sma_s)
        self._DEPENDENCIES.append(self.sma_l)

    def is_long_signal(self, df):
        return (df[self.sma_s] > df[self.sma_l]) & (
            df[self.sma_s].shift(1) <= df[self.sma_l].shift(1)
        )

    def is_short_signal(self, df):
        return (df[self.sma_s] < df[self.sma_l]) & (
            df[self.sma_s].shift(1) >= df[self.sma_l].shift(1)
        )


class EMA_Cross(Strategy):
    def __init__(self, short, long):
        super().__init__()
        if short > long:
            tmp = long
            long = short
            short = tmp

        self.name = "EMA_CROSS S:" + str(short) + " L:" + str(long)

        self.ema_s = "EMA:" + str(short)
        self.ema_l = "EMA:" + str(long)
        self._DEPENDENCIES.append(self.ema_s)
        self._DEPENDENCIES.append(self.ema_l)

    def is_long_signal(self, df):
        return (df[self.ema_s] > df[self.ema_l]) & (
            df[self.ema_s].shift(1) <= df[self.ema_l].shift(1)
        )

    def is_short_signal(self, df):
        return (df[self.ema_s] < df[self.ema_l]) & (
            df[self.ema_s].shift(1) >= df[self.ema_l].shift(1)
        )


# https://commodity.com/technical-analysis/momentum/


class MOM_ZeroCrossing(Strategy):
    def __init__(self, window):
        super().__init__()
        self.name = "MOM_ZeroCrossing W:" + str(window)
        self.mom = "MOM:" + str(window)
        self._DEPENDENCIES.append(self.mom)

    def is_long_signal(self, df):
        return (df[self.mom] > 0) & (df[self.mom].shift(1) <= 0)

    def is_short_signal(self, df):
        return (df[self.mom] < 0) & (df[self.mom].shift(1) >= 0)


class ADOSC_ZeroCrossing(Strategy):
    def __init__(self):
        super().__init__()
        self.name = "ADOSC_ZeroCrossing"
        self.ind = "ADOSC"
        self._DEPENDENCIES.append(self.ind)

    def is_long_signal(self, df):
        return (df[self.ind] > 0) & (df[self.ind].shift(1) <= 0)

    def is_short_signal(self, df):
        return (df[self.ind] < 0) & (df[self.ind].shift(1) >= 0)


class SMA_SLOPE_CHANGE(Strategy):
    def __init__(self, window=20, shift=3, threshold=0.1):
        super().__init__()
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

    def is_long_signal(self, df):
        return (df[self.mom_slope] > self.th) & (df[self.mom_slope].shift(1) <= self.th)

    def is_short_signal(self, df):
        return (df[self.mom_slope] < self.th) & (df[self.mom_slope].shift(1) >= self.th)


class EMA_SLOPE_CHANGE(Strategy):
    def __init__(self, window=20, shift=3, threshold=0.1):
        super().__init__()
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

    def is_long_signal(self, df):
        return (df[self.ema_slope] > self.th) & (df[self.ema_slope].shift(1) <= self.th)

    def is_short_signal(self, df):
        return (df[self.ema_slope] < self.th) & (df[self.ema_slope].shift(1) >= self.th)


class RSI_Breakout(Strategy):
    def __init__(self, rsi=14, break_top=0.7, break_bottom=0.3):
        super().__init__()
        self.name = (
            "RSI_Breakout RSI:"
            + str(rsi)
            + " "
            + str(break_bottom)
            + "-"
            + str(break_top)
        )
        self.top = break_top
        self.bot = break_bottom
        self.rsi = "RSI:" + str(rsi)
        self._DEPENDENCIES.append(self.rsi)

    def is_long_signal(self, df):
        return (df[self.rsi] > self.bot) & (df[self.rsi].shift(1) <= self.bot)

    def is_short_signal(self, df):
        return (df[self.rsi] < self.top) & (df[self.rsi].shift(1) >= self.top)


class SMA_Cross_StopLoss(Strategy):
    def __init__(self, days_short=50, days_long=200, window=5, max_drawdown=0.9):
        super().__init__()
        if days_short > days_long:
            tmp = days_long
            days_long = days_short
            days_short = tmp

        self.name = (
            "SMA_CROSS S:"
            + str(days_short)
            + " L:"
            + str(days_long)
            + " W:"
            + str(window)
            + " Dd:"
            + str(max_drawdown)
        )

        self.sma_s = "SMA_CLOSE:" + str(days_short)
        self.sma_l = "SMA_CLOSE:" + str(days_long)
        self.max = "MAX:" + str(window)
        self.max_dd = max_drawdown
        self._DEPENDENCIES.append(self.sma_s)
        self._DEPENDENCIES.append(self.sma_l)
        self._DEPENDENCIES.append(self.max)

    def is_long_signal(self, df):
        return (df[self.sma_s] > df[self.sma_l]) & (
            df[self.sma_s].shift(1) <= df[self.sma_l].shift(1)
        )

    def is_short_signal(self, df):
        return (df[self.sma_s] < df[self.sma_l]) & (
            df[self.sma_s].shift(1) >= df[self.sma_l].shift(1)
        ) | (df[self.max] * (1 - self.max_dd) > df["CLOSE"])


"""
class (Strategy):

    def __init__(self):
        super().__init__()
        self.name = ""    

    def is_long_signal(self, df):
        return False

    def is_short_signal(self, df):
        return False
"""
