from Simulation import Simulation
from Strategy import *
from Ticker import Ticker

strats = []
tickers = []

# tickers.append(Ticker("^SPX"))
# tickers.append(Ticker("BRK-B"))

tickers.append(Ticker("MSFT"))
tickers.append(Ticker("AAPL"))
tickers.append(Ticker("TSLA"))
tickers.append(Ticker("MOH"))
tickers.append(Ticker("PAYC"))
tickers.append(Ticker("MTCH"))
tickers.append(Ticker("LW"))

for short in range(10, 110, 20):
    for long in range(100, 300, 25):
        strats.append(SMA_Cross(short, long))

# strats.append(SMA_Cross(10, 250))
# strats.append(EMA_SLOPE_CHANGE(125, 2, 0.25))

print("Number of Strats: ", len(strats))

# sim = Simulation(ticker=ticker, strategys=strats)
sim = Simulation(tickers=tickers, strategys=strats)

# sim.set_timespan(start="2000-01-01 00:00")
sim.start()
df = sim.quick_summary()
df.to_csv("results.csv")
# sim.plot_results(show_indicators=True, log_scale=True)
