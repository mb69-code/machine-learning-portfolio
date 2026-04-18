
# ML-Driven Equity Portfolio Construction

> Stock selection and portfolio optimization using unsupervised learning and evolutionary algorithms — applied to S&P 500/400/600 universe on behalf of a US Life Insurance mandate.

## Overview

This project builds a full ML pipeline for equity portfolio construction, from feature engineering to out-of-sample backtesting. Three methods are implemented and compared:

| Method | Role | Key output |
|---|---|---|
| K-Means Clustering | Benchmark — risk/return segmentation | Cluster assignments |
| Hierarchical Clustering + HRP | Advanced diversification | Risk-parity weights |
| Genetic Algorithm | Weight & asset optimization | Optimal portfolio |

**Client mandate:** US Life Insurance company — low volatility, controlled beta ($\beta^*$), high liquidity, quarterly rebalancing.

## Methodology

### Feature engineering (9 features across 3 dimensions)

**Risk & return** — Realized volatility, Beta ($\beta$), 12-1 Momentum, Max Drawdown  
**Fundamentals** — P/E Ratio, ROE, Market Cap  
**Distribution** — Skewness, Kurtosis

### Models

**K-Means** minimizes within-cluster variance:
$$J = \sum_{j=1}^{k} \sum_{x_i \in C_j} ||x_i - \mu_j||^2$$
Optimal $k$ selected via Elbow method and Silhouette analysis.

**Hierarchical Clustering** uses a correlation-based distance metric:
$$d_{i,j} = \sqrt{\frac{1 - \rho_{i,j}}{2}}$$
Weights allocated via **Hierarchical Risk Parity (HRP)** — no matrix inversion required.

**Genetic Algorithm** evolves portfolios $w = (w_1,\dots,w_n)$ with $\sum w_i = 1$ using a custom fitness function:
$$\text{Score} = w_R \cdot R_p - w_\sigma \cdot \sigma_p - w_\beta \cdot |\beta_p - \beta^*| - w_{DD} \cdot |DD_{\max}| + w_F \cdot S_F$$

## Repository Structure

```
machine-learning-portfolio/
├── notebooks/
│   ├── eda.ipynb                     # Dataset construction & exploratory analysis
│   ├── markowitz.ipynb               # Classical MPT baseline (10-stock subset)
│   ├── k-means.ipynb                 # K-Means clustering benchmark
│   ├── hierarchical-clustering.ipynb # HRP portfolio construction & backtest
│   └── genetic-algorithm.ipynb       # GA optimization & out-of-sample test
├── data/
│   ├── stock_data.csv                # Daily prices — S&P 500/400/600 (2022–2024)
│   ├── fundamentals.csv              # P/E, Beta, Market Cap, ROE via Yahoo Finance
│   └── raw_dataset.csv               # Engineered features (merged dataset)
└── README.md
```

## Data

**Universe:** S&P 500 (large-cap), S&P 400 (mid-cap), S&P 600 (small-cap) — ~1,200 stocks  
**Period:** 2022-01-01 → 2025-01-01 (bear + transition + bull market)  
**Source:** Yahoo Finance API · Wikipedia scraping for tickers

## Getting Started

```bash
pip install numpy pandas scipy matplotlib scikit-learn yfinance
jupyter notebook notebooks/eda.ipynb
```

> Run notebooks in order: `eda` → `markowitz` → `k-means` → `hierarchical-clustering` → `genetic-algorithm`

## References

- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
- Lopez de Prado, M. (2016). *Building Diversified Portfolios that Outperform Out-of-Sample*. Journal of Portfolio Management.
- Holland, J. (1975). *Adaptation in Natural and Artificial Systems*. MIT Press.
  
