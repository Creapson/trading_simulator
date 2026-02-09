from Simulation import Simulation
from Strategy import (
    BuyAndHold,
    GoldenCross200_20,
    GoldenCross200_50,
    PriceCrossoverEMA50,
    PriceCrossoverSMA50,
)
from Ticker import Ticker

strats = []
strats.append(BuyAndHold())
strats.append(PriceCrossoverSMA50())
strats.append(PriceCrossoverEMA50())
strats.append(GoldenCross200_50())
strats.append(GoldenCross200_20())
sim = Simulation(ticker=Ticker("APPL"), strategys=strats)
sim.start()
sim.plot_results(show_indicators=True)
