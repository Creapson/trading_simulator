from Simulation import Simulation
from Strategy import *
from Ticker import Ticker

strats = []
strats.append(SMA_Cross(50, 200))
strats.append(MOM_ZeroCrossing(12))

ticker = Ticker("AAPL")
# ticker.add_indicator("ROC:10")
# ticker.add_indicator("MOM:10")

# for short in range(10, 100, 20):
#     strats.append(SMA_Cross(short, 20))

print("Number of Strats: ", len(strats))

# sim = Simulation(ticker=ticker, strategys=strats)
sim = Simulation(ticker=ticker, strategys=strats)
sim.start()
sim.plot_results(show_indicators=False, log_scale=True)
