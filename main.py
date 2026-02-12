from Simulation import Simulation
from Strategy import *
from Ticker import Ticker

strats = []
strats.append(SMA_Cross(50, 200))
# strats.append(SMA_Cross(100, 300))

for window in range(1, 30, 1):
    strats.append(MOM_ZeroCrossing(window))

ticker = Ticker("AFX.DE")
ticker.add_indicator("MACD")
# ticker.add_indicator("MOM:10")

# for short in range(10, 100, 20):
# strats.append(EMA_Cross(12, 26))
strats.append(MOM_ZeroCrossing(12))

print("Number of Strats: ", len(strats))

# sim = Simulation(ticker=ticker, strategys=strats)
sim = Simulation(ticker=ticker, strategys=strats)
sim.start()
sim.plot_results(show_indicators=False, log_scale=True)
