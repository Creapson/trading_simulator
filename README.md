# Backtesting Simulation Engine

A Python-based framework for testing financial trading strategies using historical market data. This project features a modular architecture that separates data handling, strategy logic, portfolio management, and simulation execution.

---

## Core Architecture



* **Ticker Management (`Ticker.py`)**: Handles data acquisition from Yahoo Finance or local CSV files and manages a library of over 40 technical indicators.
* **Strategy Framework (`Strategy.py`)**: Provides a base class for defining buy/sell logic and dependency requirements for indicators like SMA, EMA, and Momentum.
* **Portfolio Tracking (`Portfolio.py`)**: Maintains a record of cash and stock holdings, calculating the total portfolio value throughout the simulation.
* **Simulation Engine (`Simulation.py`)**: Orchestrates the backtest by processing price data, generating signals via strategies, and executing trades.

---

## Technical Features

### Indicator Dependency Resolution
The engine includes a sophisticated resolution system that automatically calculates parent indicators. For example, requesting `MACD_HIST` will trigger the calculation of `MACD`, `MACD_SIGNAL`, `EMA:12`, and `EMA:26` in the correct order.

### Simulation Workflow
1.  **Initialization**: Load historical data for a specific ticker.
2.  **Indicator Calculation**: Strategies request specific indicators (e.g., `SMA:50`) which are appended to the dataset.
3.  **Signal Generation**: The simulation iterates through the time series, checking for buy (1) or sell (-1) signals.
4.  **Trade Execution**: Trades are executed on the "OPEN" price of the day following a signal to avoid look-ahead bias.
5.  **Visualization**: Results are plotted against the underlying asset price, with support for logarithmic scaling and indicator overlays.

---

## Quick Start Example

```python
from Simulation import Simulation
from Strategy import SMA_Cross
from Ticker import Ticker

# Setup ticker and strategy
ticker = Ticker("AAPL")
strategy = SMA_Cross(days_short=50, days_long=200)

# Run simulation
sim = Simulation(ticker=ticker, strategys=[strategy])
sim.start()

# Plot the performance
sim.plot_results(log_scale=True)

## Supported Indicators
The system supports a wide array of technical tools including:

- Moving Averages: SMA, EMA, WMA, DEMA, TEMA, KAMA.
- Momentum: RSI, MACD, Stochastic Oscillator, CCI, CMO, ROC.
- Volatility: ATR, Bollinger Bands, True Range.
- Trend/Volume: ADX, Aroon, OBV, Chaikin Oscillator.
