# Data

### 1) stock_data.csv

Daily closing prices for the tickers of the S&P500 (large cap), S&P400 (mid cap) and S&P600 (small cap) indices.

Time window: 2022-2024

Data retreived through Yahoo Finance Python API. The tickers are retreive by webscrapping the Wikipedia pages.

### 2) fundamentals.csv

Retreiving fundamental financial data with Yahoo Finance (P/E Ratio, Beta, Market Cap, ROE).

### 3) raw_dataset.csv

With the stock_data.csv dataset, we calculate the following financial and statistical metrics: momentum, annualized volatility, max drawdown, skewness and kurtosis of the returns distribution.

Then we merge the dataset with the fundamentals dataset in order to obtain the raw_dataset.csv.


