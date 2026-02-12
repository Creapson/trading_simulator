import numpy as np


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


class SMA_Cross(Strategy):
    def __init__(self, days_short, days_long):
        if days_short > days_long:
            tmp = days_long
            days_long = days_short
            days_short = tmp

        self.name = "SMA_CROSS S:" + str(days_short) + " L:" + str(days_long)

        self.sma_s = "SMA:" + str(days_short)
        self.sma_l = "SMA:" + str(days_long)
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
