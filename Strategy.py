from dataclasses import dataclass


@dataclass
class Strategy:
    _DEPENDENCIES: [str] = []

    def get_dependencies(self):
        return self._DEPENDENCIES

    def evaluate(self):
        should_buy = self.is_buy_signal()
        should_sell = self.is_sell_signal()
        return should_buy, should_sell

    def is_buy_signal(self):
        return None

    def is_sell_signal(self):
        return None


@dataclass
class PriceCrossover_50(Strategy):
    _DEPENDENCIES: [str] = ["SMA_50"]
    _CLOSE: float
    _SMA: float

    def evaluate(self):
        return self.is_buy_signal(), self.is_sell_signal()

    def is_buy_signal(self):
        if self.CLOSE is None or self.SMA is None:
            return False
        else:
            return self.CLOSE > self.SMA

    def is_sell_signal(self):
        if self.CLOSE is None or self.SMA is None:
            return False
        else:
            return self.CLOSE < self.SMA
        return None


"""
class (Strategy):
    _DEPENDENCIES: [str] = []

    def is_buy_signal(self):
        return False

    def is_sell_signal(self):
        return false
"""
