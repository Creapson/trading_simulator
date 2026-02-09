class Strategy:
    _DEPENDENCIES = []

    def __init__(self):
        self.name = None

    def get_dependencies(self):
        return self._DEPENDENCIES

    def evaluate(self, ind):
        return self.is_buy_signal(ind), self.is_sell_signal(ind)

    def is_buy_signal(self):
        return None

    def is_sell_signal(self):
        return None


class PriceCrossover_50(Strategy):
    _DEPENDENCIES = ["SMA_50"]

    def __init__(self):
        self.name = "PriceCrossover_50"

    def is_buy_signal(self, ind):
        if ind["CLOSE"] is None or ind["SMA_50"] is None:
            return False
        else:
            return ind["CLOSE"] > ind["SMA_50"]

    def is_sell_signal(self, ind):
        if ind["CLOSE"] is None or ind["SMA_50"] is None:
            return False
        else:
            return ind["CLOSE"] < ind["SMA_50"]
        return None


class BuyAndHold(Strategy):
    _DEPENDENCIES: [] = []

    def __init__(self):
        self.name = "Buy and Hold"

    def is_buy_signal(self, ind):
        return True

    def is_sell_signal(self, ind):
        return False


"""
class (Strategy):
    _DEPENDENCIES: [] = []

    def is_buy_signal(self, ind):
        return False

    def is_sell_signal(self, ind):
        return False
"""
