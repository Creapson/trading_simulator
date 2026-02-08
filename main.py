import matplotlib.pyplot as plt
import pandas as pd

from Strategy import PriceCrossover
from Ticker import Ticker


def plot_strategy(df, sma_list):
    # Plot Bitcoin price with moving averages
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df["CLOSE"], label="BTC Price", linewidth=1)
    plt.plot(df.index, df["SMA_50"], label="SMA 50", linewidth=1.5)
    # plot each sma
    for sma in sma_list:
        plt.plot(
            df.index,
            df["NETWORTH" + sma],
            label=("PriceCrossover Strategy" + sma),
            linewidth=1,
        )
    plt.title("Bitcoin Price with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    1514761200


def simulate_strategy(dataframe, sma):
    print("starting strategy calculation with ", sma)
    number_of_btc = 0
    cash = dataframe.iloc[0]["OPEN"]
    print(cash)
    buy_signal = False
    sell_signal = False

    networth = []

    num_buys = 0
    num_sells = 0
    for date, row in dataframe.iterrows():
        open = row["OPEN"]
        close = row["CLOSE"]
        # handle signals
        if buy_signal:
            buy_signal = False
            if cash > 0:
                number_of_btc += cash / open
                cash = 0
                num_buys += 1

        if sell_signal:
            sell_signal = False
            if number_of_btc > 0:
                cash += number_of_btc * open
                number_of_btc = 0
                num_sells += 1

        # calculate signal for next day
        strategy = PriceCrossover(CLOSE=close, SMA=row[sma])
        buy_signal, sell_signal = strategy.evaluate()

        # add networth entry
        current_networth = cash + (number_of_btc * close)
        networth.append({"Date": date, "NETWORTH" + sma: current_networth})
    print(num_buys)
    print(num_sells)
    # return pd.DataFrame(networth)
    return pd.DataFrame(networth).set_index("Date")


def print_df_info(df_info):
    # Display basic info
    print(f"Dataset shape: {df_info.shape}")
    print(f"Date range: {df_info.index.min()} to {df_info.index.max()}")
    print(f"\nFirst few rows:\n{df_info}")

    print(df_info.columns)


def get_stock_data(ticker, interval="1d"):
    import yfinance as yf

    df = yf.download(ticker, start="2017-01-01", end="2026-01-01", interval="1d")

    # 1️⃣ Flatten columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2️⃣ Normalize names
    df.columns = df.columns.str.upper()
    df["SMA_50"] = df["CLOSE"].rolling(window=50).mean()
    df["SMA_20"] = df["CLOSE"].rolling(window=20).mean()
    df["SMA_200"] = df["CLOSE"].rolling(window=200).mean()
    df["SMA_10"] = df["CLOSE"].rolling(window=10).mean()

    df = df[df.index >= "2018-01-01"]
    df["UNIX_TIMESTAMP"] = df.index.astype("int64") // 10**9
    return df


"""
# Load the dataset
df = pd.read_csv("data/bitcoin-hourly-technical-indicators.csv")

# use data after 2018
df_filtered = df[df["UNIX_TIMESTAMP"] > 1514761200]
"""
print("Get Stock Data From yFinance")
df_stock = get_stock_data(ticker="DTE.DE", interval="1d")
print_df_info(df_stock)

df_sma_50 = simulate_strategy(df_stock, sma="SMA_50")
df_sma_200 = simulate_strategy(df_stock, sma="SMA_200")
df_sma_20 = simulate_strategy(df_stock, sma="SMA_20")
df_sma_10 = simulate_strategy(df_stock, sma="SMA_10")

df_merged = df_stock.join([df_sma_10, df_sma_20, df_sma_50, df_sma_200])
msft = Ticker("MSFT")
msft.load_history()
msft.add_indicator("SMA_20")
msft.print_info()
print("finished simulating")

plot_strategy(df_merged, ("SMA_10", "SMA_20", "SMA_50", "SMA_200"))
