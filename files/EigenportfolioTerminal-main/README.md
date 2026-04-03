# ⚡ EigenPortfolio — Quant Research Terminal

> **Production-grade quantitative finance platform built on Random Matrix Theory (RMT)**  
> Built by [Sourish Dey](https://sourishdeyportfolio.vercel.app/)

---

## 🎯 What is this?

A hedge-fund-grade portfolio research terminal that:
- Fetches real-time market data (S&P 500 / NIFTY 50, 120+ stocks)
- Applies **Random Matrix Theory** to denoise covariance matrices
- Constructs **Eigenportfolios** from signal eigenvectors
- Performs full **portfolio optimization** (Sharpe, Min-Var, Risk Parity, Efficient Frontier)
- Runs **rolling window backtests** with transaction costs
- Generates **80+ interactive Plotly visualizations**
- Exports CSV, PDF reports, and portfolio configs

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
eigenportfolio_app/
├── app.py          # Main Streamlit UI (dark terminal theme, tabbed layout)
├── data.py         # Data fetching (yfinance, parallel), universe definitions
├── rmt.py          # Random Matrix Theory engine (MP distribution, denoising)
├── portfolio.py    # Portfolio construction & optimization (MVO, Risk Parity, etc.)
├── backtest.py     # Rolling window backtesting engine
├── visuals.py      # 80+ Plotly chart functions (dark quant theme)
├── utils.py        # Export (CSV/PDF), auto-insights, research mode
└── requirements.txt
```

---

## 🧠 Core Concepts

### Random Matrix Theory (RMT)
The **Marchenko-Pastur distribution** gives the theoretical eigenvalue density for a random matrix:

```
λ± = σ² (1 ± 1/√q)²,   q = T/N
```

Eigenvalues **above λ+** carry genuine financial signal (market factor, sector factors).  
Eigenvalues within the bulk are statistically indistinguishable from noise — and are filtered out.

### Eigenportfolios
An eigenportfolio has weights proportional to an eigenvector of the correlation matrix:
- **PC1** → Market factor (all stocks correlated)
- **PC2, PC3** → Sector/style factors
- Eigenportfolios are **orthogonal** — zero pairwise correlation

### Covariance Denoising
Noise eigenvalues are replaced by their average, yielding a regularized covariance matrix with better out-of-sample stability.

---

## 📊 Feature Overview

| Feature | Details |
|---|---|
| **Universe** | S&P 500 (120+ stocks) or NIFTY 50 |
| **Timeframes** | 1mo to 5yr, daily or weekly |
| **RMT** | Marchenko-Pastur fit, eigenvalue decomposition, denoising |
| **Portfolios** | Eigenportfolio, Max Sharpe, Min Variance, Risk Parity, Equal Weight |
| **Backtesting** | Rolling window, configurable rebalance, transaction costs |
| **Metrics** | CAGR, Sharpe, Max Drawdown, Volatility, Calmar, Alpha, Beta |
| **Visuals** | 80+ Plotly charts: heatmaps, 3D surfaces, radar, treemaps, equity curves |
| **Export** | CSV (prices, returns, weights, backtest), PDF report, JSON config |

---

## 🎨 UI Features
- **Dark quant-terminal theme** (JetBrains Mono + Space Grotesk)
- **5 tabs**: Overview, RMT Analysis, Eigenportfolios, Optimization, Backtesting
- **Custom metric cards, section headers, insight cards**
- **Research Mode** with educational explanations
- **Auto Insights** (rule-based AI observations)
- Animated sidebar with live status indicator

---

## ⚙️ Configuration

All parameters are controlled from the sidebar:

- **Universe**: Select market + individual stocks (up to 120+)
- **Compare**: Side-by-side analysis of up to 4 stocks
- **Period**: 1mo to 5yr
- **Eigenportfolio**: Choose which PC, how many PCs to combine, long-only vs long-short
- **Backtesting**: Training window, rebalance frequency, transaction cost (bps)
- **Features**: Monte Carlo, Research Mode, Auto Insights

---

## 📤 Exports

- **Price data CSV** — historical adjusted closes
- **Log returns CSV** — daily log-return matrix
- **Portfolio weights CSV** — sorted by weight
- **Backtest results CSV** — equity curve + daily returns
- **PDF report** — RMT summary, performance metrics, top holdings
- **Config JSON** — save and reload your portfolio setup

---

## ⚠️ Disclaimer

This application is for **research and educational purposes only**.  
It does **not** constitute financial advice.  
Past backtest performance does not guarantee future results.

---

## 👤 Author

**Sourish Dey**  
🌐 [sourishdeyportfolio.vercel.app](https://sourishdeyportfolio.vercel.app/)
