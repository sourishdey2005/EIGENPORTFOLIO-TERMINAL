"""
backtest.py — Rolling Window Backtesting Engine
Supports:
  - Rolling window construction + rebalancing
  - Transaction costs
  - Performance metrics: CAGR, Sharpe, Max Drawdown, Volatility
  - Rolling alpha & beta vs benchmark
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from rmt import run_rmt
from portfolio import (
    build_eigenportfolio,
    max_sharpe_portfolio,
    min_variance_portfolio,
    risk_parity_portfolio,
    equal_weight_portfolio,
    compute_turnover,
)


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    equity_curve: pd.Series           # Cumulative portfolio value (starting at 1)
    daily_returns: pd.Series          # Daily arithmetic returns
    rebalance_dates: list             # Dates when portfolio was rebalanced
    weights_history: list[np.ndarray] # Weights at each rebalance
    turnover_series: list[float]      # Turnover at each rebalance
    cagr: float
    sharpe: float
    max_drawdown: float
    volatility: float
    calmar: float
    name: str


# ─── Performance Metrics ──────────────────────────────────────────────────────

def compute_cagr(equity: pd.Series) -> float:
    n_years = len(equity) / 252
    if n_years <= 0 or equity.iloc[0] <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1


def compute_sharpe(daily_returns: pd.Series, rf_daily: float = 0.0) -> float:
    excess = daily_returns - rf_daily
    if excess.std() == 0:
        return 0.0
    return excess.mean() / excess.std() * np.sqrt(252)


def compute_max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())


def compute_volatility(daily_returns: pd.Series) -> float:
    return float(daily_returns.std() * np.sqrt(252))


def compute_metrics(equity: pd.Series, daily_returns: pd.Series) -> dict:
    cagr = compute_cagr(equity)
    sharpe = compute_sharpe(daily_returns)
    mdd = compute_max_drawdown(equity)
    vol = compute_volatility(daily_returns)
    calmar = cagr / abs(mdd) if mdd != 0 else 0.0
    return dict(cagr=cagr, sharpe=sharpe, max_drawdown=mdd, volatility=vol, calmar=calmar)


# ─── Main Backtesting Function ────────────────────────────────────────────────

def run_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    strategy: str = "eigenportfolio",
    train_window: int = 252,       # days of history used to build portfolio
    rebalance_freq: int = 21,      # rebalance every N trading days
    transaction_cost: float = 0.001,   # 10 bps one-way
    pc_index: int = 0,
    long_only: bool = True,
    n_assets: int = None,
) -> BacktestResult:
    """
    Rolling-window backtest.
    
    For each rebalance date:
      1. Use last `train_window` days of returns to fit portfolio
      2. Hold for `rebalance_freq` days
      3. Apply transaction costs on turnover
    """
    tickers = list(returns.columns)
    if n_assets is not None and n_assets < len(tickers):
        tickers = tickers[:n_assets]
        returns = returns[tickers]

    n = len(tickers)
    total_days = len(returns)
    
    equity = []
    daily_rets = []
    rebalance_dates = []
    weights_history = []
    current_weights = equal_weight_portfolio(n)

    portfolio_value = 1.0

    for t in range(train_window, total_days):
        # ── Check if rebalance day ────────────────────────────────────────
        is_rebalance = (t - train_window) % rebalance_freq == 0

        if is_rebalance:
            window_rets = returns.iloc[t - train_window: t]
            new_weights = _compute_weights(
                window_rets, strategy, pc_index, long_only, n
            )

            # Transaction cost = cost * one-way turnover
            to = 0.5 * np.abs(new_weights - current_weights).sum()
            cost = transaction_cost * to
            portfolio_value *= (1 - cost)

            current_weights = new_weights
            rebalance_dates.append(returns.index[t])
            weights_history.append(current_weights.copy())

        # ── Daily return ──────────────────────────────────────────────────
        day_rets = returns.iloc[t].values
        port_ret = current_weights @ day_rets
        portfolio_value *= (1 + port_ret)
        equity.append(portfolio_value)
        daily_rets.append(port_ret)

    equity_s = pd.Series(equity, index=returns.index[train_window:])
    daily_s = pd.Series(daily_rets, index=returns.index[train_window:])

    metrics = compute_metrics(equity_s, daily_s)
    turnovers = compute_turnover(weights_history)

    return BacktestResult(
        equity_curve=equity_s,
        daily_returns=daily_s,
        rebalance_dates=rebalance_dates,
        weights_history=weights_history,
        turnover_series=turnovers,
        cagr=metrics["cagr"],
        sharpe=metrics["sharpe"],
        max_drawdown=metrics["max_drawdown"],
        volatility=metrics["volatility"],
        calmar=metrics["calmar"],
        name=strategy,
    )


def _compute_weights(
    window_rets: pd.DataFrame,
    strategy: str,
    pc_index: int,
    long_only: bool,
    n: int,
) -> np.ndarray:
    """Dispatch to the appropriate portfolio construction method."""
    try:
        if strategy == "eigenportfolio":
            result = run_rmt(window_rets)
            w = build_eigenportfolio(result.eigenvectors, list(window_rets.columns), pc_index, long_only)
        elif strategy == "max_sharpe":
            w = max_sharpe_portfolio(window_rets, long_only)
        elif strategy == "min_variance":
            w = min_variance_portfolio(window_rets, long_only)
        elif strategy == "risk_parity":
            w = risk_parity_portfolio(window_rets)
        elif strategy == "equal_weight":
            w = equal_weight_portfolio(n)
        else:
            w = equal_weight_portfolio(n)
    except Exception:
        w = equal_weight_portfolio(n)
    
    # Validate
    if np.isnan(w).any() or w.sum() == 0:
        w = equal_weight_portfolio(n)
    return w


# ─── Rolling Alpha / Beta ─────────────────────────────────────────────────────

def compute_rolling_alpha_beta(
    port_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
) -> pd.DataFrame:
    """
    OLS rolling regression: port = alpha + beta * bench + epsilon
    Returns DataFrame with columns ['alpha', 'beta'].
    """
    aligned = pd.concat([port_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["port", "bench"]

    alphas, betas = [], []
    idx = []

    for i in range(window, len(aligned)):
        sub = aligned.iloc[i - window: i]
        x = sub["bench"].values
        y = sub["port"].values
        x_bar, y_bar = x.mean(), y.mean()
        beta = np.sum((x - x_bar) * (y - y_bar)) / (np.sum((x - x_bar) ** 2) + 1e-12)
        alpha = (y_bar - beta * x_bar) * 252   # annualised alpha
        alphas.append(alpha)
        betas.append(beta)
        idx.append(aligned.index[i])

    return pd.DataFrame({"alpha": alphas, "beta": betas}, index=idx)


# ─── Benchmark Fetch ──────────────────────────────────────────────────────────

def get_benchmark_returns(prices: pd.DataFrame, benchmark: str = "equal_weight") -> pd.Series:
    """Build a benchmark return series from the same price universe."""
    log_rets = np.log(prices / prices.shift(1)).dropna()
    if benchmark == "equal_weight":
        return log_rets.mean(axis=1)
    return log_rets.mean(axis=1)
