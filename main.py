from gui.windows import PriceChartWindow
from simulation.Simulation import Simulation
from simulation.Analysis import Analysis
from simulation.Strategy import *
from ticker.Ticker import Ticker

from gui.windows.PriceChartWindow import PriceChartWindow

import dearpygui.dearpygui as dpg


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

msft = Ticker("SIMULATION")
msft.add_indicators(["RSI:21", "EMA:24"])
tickers.append(msft)
# tickers = load_tickers_from_file("smp_500_stocks.txt")
# strats.append(RSI_Breakout())

strats.append(HoldAndReinvest())
strats.append(RSI_Breakout(14, 0.5, 0.95))
strats.append(MOM_ZeroCrossing(1))
strats.append(SMA_Cross(10, 110))

print("Number of Strats: ", len(strats))

# sim = Simulation(ticker=ticker, strategys=strats)
sim = Simulation(tickers=tickers, strategys=strats)
# analysis = Analysis()
# corr_mat = analysis.correlation(tickers, "ROC:1", "pearson", 1)
# corr_pairs = analysis.correlated_ticker(corr_mat, 0.75)
print("Highly Correlated Stock Pairs (> 0.6):")
# print(corr_pairs.sort_values(ascending=False).to_string())

# sim.set_timespan(start="2000-01-01 00:00")
sim.start(show_progress=False)
df = sim.get_quick_summary()
df.to_csv("results.csv")

# 3. Setup Dear PyGui
dpg.create_context()

chartWindow = PriceChartWindow()
chartWindow.setup()
sim.plot_results(chartWindow=chartWindow, show_indicators=True, log_scale=False, show_volume=False)
# 4. Render the UI
dpg.create_viewport(title='YFinance + Dear PyGui', width=850, height=550)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window(chartWindow.id, True)
dpg.start_dearpygui()
dpg.destroy_context()
