class Strategy:
    _DEPENDENCIES = []

    def __init__(self):
        self.name = None

    def get_dependencies(self):
        return self._DEPENDENCIES

    def evaluate(self, ind):
        # Check if every dependency is available
        for dependency in self._DEPENDENCIES:
            if ind.get(dependency) is None:
                print(f"Missing data for: {dependency}")
                return False, False

        return self.is_buy_signal(ind), self.is_sell_signal(ind)

    def is_buy_signal(self):
        return None

    def is_sell_signal(self):
        return None


class PriceCrossoverSMA50(Strategy):
    _DEPENDENCIES = ["SMA_50"]

    def __init__(self):
        self.name = "PriceCrossover with SMA_50"

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


class PriceCrossoverEMA50(Strategy):
    _DEPENDENCIES = ["EMA_50"]

    def __init__(self):
        self.name = "PriceCrossover with EMA_50"

    def is_buy_signal(self, ind):
        if ind["CLOSE"] is None or ind["EMA_50"] is None:
            return False
        else:
            return ind["CLOSE"] > ind["EMA_50"]

    def is_sell_signal(self, ind):
        if ind["CLOSE"] is None or ind["EMA_50"] is None:
            return False
        else:
            return ind["CLOSE"] < ind["EMA_50"]
        return None


class BuyAndHold(Strategy):
    _DEPENDENCIES: [] = []

    def __init__(self):
        self.name = "Buy and Hold"

    def is_buy_signal(self, ind):
        return True

    def is_sell_signal(self, ind):
        return False


class GoldenCross(Strategy):
    _DEPENDENCIES: [] = ["SMA_50", "SMA_200"]

    def __init__(self):
        self.name = "GoldenCross"

    def is_buy_signal(self, ind):
        return ind["SMA_50"] > ind["SMA_200"]

    def is_sell_signal(self, ind):
        return ind["SMA_50"] < ind["SMA_200"]


"""
class (Strategy):
    _DEPENDENCIES: [] = []

    def __init__(self):
        self.name = ""    

    def is_buy_signal(self, ind):
        return False

    def is_sell_signal(self, ind):
        return False
"""
