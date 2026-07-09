import pandas as pd
import numpy as np

from typing import List, Tuple
from Ticker import Ticker

from datetime import datetime

class Analysis:

    def correlation(self, tickers: List[Ticker], column: str, method: str, lookback_years: int = 2) -> pd.DataFrame:
        cutoff_date = datetime.now() - pd.DateOffset(years=lookback_years)

        df_list: List[pd.DataFrame] = []
        for ticker in tickers:
            df = ticker.df
            df = df.filter([column])
            df.columns = [ticker.ticker]
            df = df[df.index > cutoff_date]
            df_list.append(df)

        combined_df = pd.concat(df_list, axis=1, join='outer')
        correlation_matrix = combined_df.corr()
        return correlation_matrix

    def correlated_ticker(self, correlation_matrix: pd.DataFrame, corr_value: float = 0.6):
        mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)

        high_corr_list = correlation_matrix.where(mask).unstack().dropna()
        high_corr_list = high_corr_list[high_corr_list > corr_value]
        return high_corr_list
