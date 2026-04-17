# Notebooks

### 1) eda.ipynb

This notebook constructs and presents the dataset used throughout the project and includes an exploratory data analysis to better understand its characteristics. The analysis covers:

- **General Information and Statistics:** Using `info()` and `describe()` to examine the structure and summary statistics of the dataset.  
- **Risk vs. Reward Analysis:** Plotting *Volatility* against *Momentum*, scaled by market capitalization, to identify patterns in the 2022–2024 period.  
- **Correlation Analysis:** Checking relationships between features and assessing multicollinearity.  
- **Outlier Detection and Feature Distributions:** Identifying extreme values and distribution shapes to determine the preprocessing steps required for clustering.

### 2) markowitz.ipynb

This notebook focuses on constructing the optimal portfolio for 10 US stocks using Modern Portfolio Theory (Markowitz) by maximizing the Sharpe ratio. The goal is to understand the classical approach to portfolio optimization and allocation, emphasizing the balance between expected returns and risk (volatility).

The method uses only two features: annualized volatility and returns. To get familiar with the concepts, a preliminary test has been carried out on a subset of 10 stocks from the S&P500. 

### 3) k-means.ipynb

This notebook implements the K-Means algorithm as a benchmark machine learning method. It uses the 9-feature dataset prepared in `eda.ipynb` and applies the necessary preprocessing steps to ensure the data is suitable for clustering.  

The goal is to classify stocks into distinct market regimes based on both risk and fundamental metrics. By analyzing the resulting clusters, we can identify patterns such as groups of low-volatility, high-momentum large caps versus high-risk, high-variance small caps. These clusters will later be compared with portfolios constructed using alternative techniques, like the genetic algorithm, to assess their effectiveness in capturing market structure.

### 4) hierarchical-clustering.ipynb

This notebook introduces the Hierarchical Risk Parity (HRP) approach as an advanced portfolio allocation technique designed to overcome the limitations of traditional Markowitz optimization. Built on the previously prepared 9-feature dataset of over 1,200 stocks, it first applies essential preprocessing steps, such as outlier clipping and feature standardization, to ensure that financial metrics with different scales contribute equally to the model.

By applying Ward's agglomerative clustering, the algorithm maps the true hierarchical correlation structure of the market. The HRP algorithm then distributes portfolio weights based on these cluster variances to maximize diversification. Finally, the strategy's out-of-sample performance is backtested against a standard Equally Weighted (EW) benchmark, evaluating its effectiveness through key risk metrics.

### 5) genetic-algorithm.ipynb

This notebook implements a Genetic Algorithm (GA) to handle both asset selection and weight optimization simultaneously. Rather than just grouping similar assets, this evolutionary approach simulates natural selection across multiple generations to find an optimal portfolio composition. Each candidate portfolio is evaluated using a fitness function based on an institutional insurer's mandate, which prioritizes low volatility, controlled beta, and reasonable returns.

The algorithm is trained on historical data from 2022 to 2023 to determine the best stock subset and their respective allocations. The resulting portfolio is then tested on out-of-sample data from 2024 to assess its practical performance. The results are analyzed using standard risk-adjusted metrics, including the Sharpe, Sortino along with maximum drawdown.

