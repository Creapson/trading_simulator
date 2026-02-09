from Simulation import Simulation
from Strategy import BuyAndHold, GoldenCross, PriceCrossoverEMA50, PriceCrossoverSMA50
from Ticker import Ticker

strats = []
strats.append(BuyAndHold())
strats.append(PriceCrossoverSMA50())
strats.append(PriceCrossoverEMA50())
strats.append(GoldenCross())
sim = Simulation(ticker=Ticker("PLTR"), strategys=strats)
sim.start()
sim.plot_results(show_indicators=True)
