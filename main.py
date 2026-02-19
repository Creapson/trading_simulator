from Simulation import Simulation
from Strategy import *
from Ticker import Ticker


def load_tickers_from_file(filename, max_num=99999):
    ticker_list = []
    i = 0
    with open("data/ticker/" + filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if i > max_num:
                return ticker_list
            ticker_list.append(Ticker(line.strip()))
            i += 1
    return ticker_list


strats = []
tickers = []


# tickers.append(Ticker("^SPX"))
tckr = Ticker("BTC-USD")
tickers.append(tckr)
# tickers = load_tickers_from_file("smp_500_stocks.txt")

# strats.append(BuyAndHold())
# for short in range(10, 110, 10):
#     for long in range(100, 310, 10):
#         strats.append(SMA_Cross(short, long))
strats.append(SMA_Cross(10, 110))
strats.append(SMA_Cross())

# strats.append(SMA_Cross(10, 110))
# for bot in range(10, 55, 5):
#     for top in range(55, 100, 5):
#         strats.append(RSI_Breakout(break_bottom=bot / 100.0, break_top=top / 100.0))

# for window in range(10, 100, 10):
#     for shift in range(0, 4, 1):
#         for thresh in range(1, 6, 1):
#             strats.append(EMA_SLOPE_CHANGE(window, shift, thresh / 10))

# for window in range(0, 100, 1):
#     strats.append(MOM_ZeroCrossing(window))

# strats.append(ADOSC_ZeroCrossing())

print("Number of Strats: ", len(strats))

# sim = Simulation(ticker=ticker, strategys=strats)
sim = Simulation(tickers=tickers, strategys=strats)

# sim.set_timespan(start="2000-01-01 00:00")
sim.start(show_progress=True)
df = sim.get_quick_summary()
df.to_csv("results.csv")
sim.plot_results(show_indicators=True, log_scale=True, show_volume=False)
