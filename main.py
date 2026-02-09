from Simulation import Simluation
from Strategy import BuyAndHold, PriceCrossover_50
from Ticker import Ticker

strats = (BuyAndHold(), PriceCrossover_50())
sim = Simluation(ticker=Ticker("TSLA"), strategys=strats)
sim.start()
sim.plot_results()

print("finished simulating")
