from Simulation import Simulation
from Strategy import *
from Ticker import Ticker

strats = []
# strats.append(BuyAndHold())
# strats.append(PriceCrossoverSMA50())
# strats.append(PriceCrossoverSMA50())
# strats.append(Hyterese())
strats.append(GoldenCross200_50())
strats.append(GoldenCrossSMA100_Dropout())
strats.append(GoldenCrossSMA200_Dropout())
sim = Simulation(ticker=Ticker("^SPX"), strategys=strats)
sim.start()
sim.plot_results(show_indicators=True, log_scale=False)
