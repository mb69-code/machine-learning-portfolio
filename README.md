# **Equity Portfolio Construction & Stock Selection with ML**

This project focuses on applying Machine Learning techniques to the finance sector, specifically for **Equity Portfolio Construction and Optimization**. By leveraging data-driven strategies, we aim to automate and enhance stock selection processes among a given list of stocks (e.g., S&P 500). Implementation of clustering algorithms (Unsupervised Learning) such as *K-Means* or *Hierarchical Clustering* to group stocks with similar characteristics. Usage of clusters to select uncorrelated assets for better portfolio diversification.

# **Context**

As asset managers for US Life Insurance companies, we seek to optimize the equity allocation of our fund. By leveraging Machine Learning, we aim to select stocks that maximize returns while strictly adhering to the client's low-risk and high-liquidity mandates.

| Client | US Life Insurance Company |
| :--- | :--- |
| **Profile** | Long-term horizon, low risk tolerance, low need for immediate income. |
| **Key Constraints** | **High liquidity requirements** (to cover sudden claims/payouts). |

### **Strategy**

US Life Insurance companies struggle to fund their liabilities: very long-term obligations (10–30 years) often linked to annuity or life insurance contracts, combined with high liquidity needs for claims. Traditional solutions available to insurers (primarily bond portfolios or standardized mixed funds) have shown their limits: they either fail to generate sufficient yield to cover long-term liabilities or take on too much market or liquidity risk, potentially compromising liability stability.

Our fund aims to address this specific need by offering a simple and transparent portfolio that combines **liquidity**, **yield**, and **risk control**. We propose constructing a liquid portfolio, predominantly fixed-income, supplemented by equities (20–25%) to enhance expected returns. This includes quantified yield and volatility targets (rather than guaranteed promises) and quarterly rebalancing to limit risk. This approach would allow the insurer to meet its liquidity and solvency requirements while optimizing the portfolio's long-term return, unlike the standardized solutions currently on the market.

#### Asset Allocation

- **Bonds (70%):** Diversification between Government bonds (e.g., US Treasury 10Y) and Investment Grade Corporate bonds, with a duration focus adapted to liabilities.
- **Equities (30%):** Selection of high-quality US stocks to improve expected yield while limiting volatility (e.g., JPM, 3M, PG).

Quantitative Objectives: Target yield, maximum volatility threshold, minimum liquidity level, with quarterly monitoring to adjust the portfolio according to market evolution.

### 1. K-Means Clustering (Benchmark)
* **Goal:** Establish a baseline grouping of assets.
* **Method:** Partition stocks into $k$ distinct clusters based on risk/return profiles.
* **Selection:** Identification of similar assets to avoid concentration.

### 2. Hierarchical Clustering
* **Goal:** Advanced structure analysis for risk diversification.
* **Method:** Build a dendrogram to visualize asset relationships and implement **Hierarchical Risk Parity (HRP)**.
* **Advantage:** Unlike K-means, this does not force a pre-defined number of clusters and captures nested correlations.

### 3. Genetic Algorithm (Optimization)
* **Goal:** Portfolio selection and weight optimization.
* **Method:** Use evolutionary principles (selection, crossover, mutation) to find the optimal combination of stocks.
* **Fitness Function:** Maximizing the Sharpe Ratio or minimizing Volatility under specific constraints.


# **Data**

We retrieve financial data $X^{\intercal}=[x^{(1)}, x^{(2)}, \dots, x^{(N)}]$ for constituents of the S&P 500 (large-cap), S&P 400 (mid-cap), and S&P 600 (small-cap) indices.

**Time Window:** `2022-01-01` to `2025-01-01`

This three-year period is chosen to capture a diverse, representative range of recent market conditions without being distorted by the extreme volatility of the 2020 COVID-19 crash. Specifically, it encompasses:
* **2022 (Bear Market):** Rising inflation, monetary tightening, and a growth/tech sell-off.
* **2023 (Transition):** Market stabilization and gradual normalization.
* **2024 (Bull Market):** A renewed rally driven by technology and AI stocks.

Spanning these distinct macroeconomic shocks and sector rotations provides the variance needed to clearly discriminate between equity styles, such as growth vs. value, high-beta vs. defensive, and momentum trends, which is essential for meaningful clustering.


#### **Feature Engineering & Selection**

We engineered features across three dimensions to capture a comprehensive profile of asset behavior:

**1. Risk & Return Metrics**
Quantifies an asset's price dynamics, trend persistence, and historical downside.
* **Realized Volatility:** Annualized standard deviation of daily returns.
* **Beta ($\beta$):** Sensitivity to broader market movements (e.g., S&P 500).
* **12-1 Momentum:** Trailing 12-month return excluding the most recent month. (This standard financial metric captures persistent trends while filtering out short-term, 1-month reversal noise).
* **Max Drawdown:** Maximum percentage drop from a historical peak (measures worst-case historical loss).

**2. Fundamental Metrics**
Clusters stocks based on core financial health, valuation, and scale.
* **Valuation:** Price-to-Earnings (P/E) Ratio.
* **Profitability:** Return on Equity (ROE).
* **Size:** Market Capitalization.

**3. Statistical Moments**

Evaluates the shape of the return distribution to assess tail risk (the likelihood of rare, extreme events).
* **Skewness:** Asymmetry of the return distribution (e.g., negative skew indicates a higher probability of large losses).
* **Kurtosis:** Fatness of the distribution tails (measures the frequency of extreme outliers).

# **1. K-Means Clustering (Benchmark)**

**Objective**

We utilize K-Means as a baseline algorithm to partition the S&P 500 universe into distinct "risk-return buckets." By grouping stocks with similar characteristics (volatility, liquidity, fundamentals), we can ensure our portfolio diversifies across different behavioral clusters rather than just industrial sectors.

**Algorithm Overview**

K-Means is an iterative algorithm that partitions a dataset of $n$ stocks into $k$ non-overlapping clusters. It aims to minimize the within-cluster sum of squares (variance), ensuring that stocks inside a cluster are as similar as possible.

The objective function $J$ is defined as:

$$J = \sum_{j=1}^{k} \sum_{x_i \in C_j} ||x_i - \mu_j||^2$$

Where:
* $k$ is the number of clusters.
* $C_j$ is the set of points belonging to cluster $j$.
* $\mu_j$ is the centroid (mean) of cluster $j$.
* $||x_i - \mu_j||^2$ is the squared Euclidean distance between a stock $x_i$ and the centroid.

**Implementation Steps**
1.  **Feature Normalization:** All features (e.g., Volatility, P/E, Amihud Ratio) are scaled using Z-Score standardization to prevent large-magnitude features (like Market Cap) from dominating the distance metric.
2.  **Optimal $k$ Selection:** We use the **Elbow Method** and **Silhouette Analysis** to determine the optimal number of clusters that best separates the data without overfitting.
3.  **Cluster assignment:** Stocks are assigned to the nearest cluster centroid based on their feature vector.
4.  **Centroid Update:** The centroids are recalculated as the mean of all stocks in the cluster. Steps 3 and 4 repeat until convergence.

# **2. Hierarchical Clustering**

* **Goal:** Identify the hierarchical structure of relationships between assets to improve diversification and risk allocation.
* **Algorithm Overview:** Agglomerative clustering progressively merges assets or clusters based on their similarity. The distance between assets is derived from the correlation matrix:

$$
d_{i,j} = \sqrt{\frac{1 - \rho_{i,j}}{2}}
$$

where $\rho_{i,j}$ is the correlation between asset $i$ and asset $j$.

* **Implementation Steps:** Build a **dendrogram** representing the hierarchical relationships between stocks, then use this structure to allocate portfolio weights with the **Hierarchical Risk Parity (HRP)** method, which distributes capital across clusters according to their risk.

### 3. Genetic Algorithm (Optimization)
* **Goal:** Optimize portfolio weights and select the best combination of assets under risk and diversification constraints.
* **Algorithm Overview:** A population of candidate portfolios evolves through **selection**, **crossover**, and **mutation** operations. Each portfolio is evaluated using a fitness function such as the **Sharpe Ratio**:

$$
S = \frac{E[R_p] - r_f}{\sigma_p}
$$

where $E[R_p]$ is the expected portfolio return, $r_f$ the risk-free rate, and $\sigma_p$ the portfolio volatility.

* **Implementation Steps:** Initialize a population of portfolios $w = (w_1,\dots,w_n)$ satisfying $\sum_{i=1}^{n} w_i = 1$. At each generation, the best-performing portfolios are selected, recombined, and slightly mutated to explore new allocations until convergence toward an optimal solution.

* **Fitness Function**

$$ \text{Score} = \underbrace{w_R \cdot R_p}_{\text{Return}} - \underbrace{w_\sigma \cdot \sigma_p}_{\text{Volatility}} - \underbrace{w_\beta \cdot |\beta_p - \beta^*|}_{\beta \text{ Deviation}} - \underbrace{w_{DD} \cdot |DD_{\max}|}_{\text{Drawdown}} + \underbrace{w_F \cdot S_F}_{\textcolor{mypurple}{\text{Fundamentals}}}$$


**Application for Insurers**
For our specific client, K-Means helps identify "Defensive Clusters" (low beta, low volatility, high liquidity) vs. "Speculative Clusters." We focus our selection on the most stable clusters to match the insurer's liability constraints.

