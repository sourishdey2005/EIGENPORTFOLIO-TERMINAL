"""
portfolio.py — Portfolio Construction & Optimization
Implements:
  - Eigenportfolio construction from RMT eigenvectors
  - Mean-Variance Optimization (Markowitz)
  - Sharpe Ratio Maximization
  - Minimum Variance Portfolio
  - Risk Parity
  - Efficient Frontier generation
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class PortfolioWeights:
    weights: np.ndarray          # N-dim weight vector
    tickers: list[str]
    name: str
    expected_return: float       # Annualised
    volatility: float            # Annualised
    sharpe: float


# ─── Eigenportfolios ──────────────────────────────────────────────────────────

def build_eigenportfolio(
    eigenvectors: np.ndarray,
    tickers: list[str],
    pc_index: int = 0,          # 0 = first (largest) eigenvector
    long_only: bool = True,
) -> np.ndarray:
    """
    Build a portfolio from a single eigenvector.
    
    The k-th eigenvector of the correlation matrix defines a factor portfolio:
    each component is the loading of stock i on factor k.
    
    Long-only: take positive loadings only (and renormalize).
    Long-short: use raw signed loadings (net-zero exposure).
    """
    v = eigenvectors[:, pc_index].copy()

    if long_only:
        v = np.where(v > 0, v, 0.0)
        total = v.sum()
        if total == 0:
            # flip sign if all negative
            v = np.abs(eigenvectors[:, pc_index])
            total = v.sum()
        weights = v / total
    else:
        # Long-short: normalize by L1 norm so |w|₁ = 1
        l1 = np.abs(v).sum()
        weights = v / l1 if l1 > 0 else v

    return weights


def build_top_k_eigenportfolio(
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    tickers: list[str],
    k: int = 3,
    long_only: bool = True,
) -> np.ndarray:
    """
    Combine top-k eigenvectors weighted by their eigenvalue (variance explained).
    w = Σ_{i=1}^{k}  λ_i * v_i   (then normalize)
    """
    weights = np.zeros(len(tickers))
    total_lam = eigenvalues[:k].sum()
    for i in range(k):
        v = build_eigenportfolio(eigenvectors, tickers, i, long_only)
        weights += (eigenvalues[i] / total_lam) * v
    if long_only:
        weights = np.maximum(weights, 0)
    norm = np.abs(weights).sum()
    return weights / norm if norm > 0 else weights


# ─── Return / Risk Estimation ─────────────────────────────────────────────────

def annualise_stats(returns: pd.DataFrame, weights: np.ndarray) -> tuple[float, float, float]:
    """
    Compute expected return, volatility, Sharpe for a weight vector.
    Assumes 252 trading days, risk-free rate = 0.
    """
    mu = returns.mean().values * 252
    Sigma = returns.cov().values * 252
    port_ret = weights @ mu
    port_vol = np.sqrt(weights @ Sigma @ weights)
    sharpe = port_ret / port_vol if port_vol > 0 else 0.0
    return port_ret, port_vol, sharpe


# ─── Optimization Helpers ─────────────────────────────────────────────────────

def _portfolio_volatility(w, Sigma):
    return np.sqrt(w @ Sigma @ w)


def _neg_sharpe(w, mu, Sigma, rf=0.0):
    ret = w @ mu - rf
    vol = _portfolio_volatility(w, Sigma)
    return -ret / vol if vol > 0 else 0.0


def _risk_parity_obj(w, Sigma):
    """Equal risk contribution objective (sum of squared RC differences)."""
    port_var = w @ Sigma @ w
    # Marginal risk contribution
    mrc = Sigma @ w
    # Risk contribution
    rc = w * mrc / port_var
    return np.sum((rc - rc.mean()) ** 2)


def _optimize(
    obj,
    n: int,
    bounds,
    constraints,
    x0: Optional[np.ndarray] = None,
    n_restarts: int = 3,
):
    """Run scipy minimize with multiple restarts for global optimum."""
    best_result = None
    for _ in range(n_restarts):
        x0_ = x0 if x0 is not None else np.random.dirichlet(np.ones(n))
        res = minimize(
            obj,
            x0_,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if best_result is None or res.fun < best_result.fun:
            best_result = res
        x0 = None  # subsequent restarts are random
    return best_result


# ─── Portfolio Optimization Functions ────────────────────────────────────────

def max_sharpe_portfolio(
    returns: pd.DataFrame,
    long_only: bool = True,
    rf: float = 0.0,
) -> np.ndarray:
    """Maximum Sharpe Ratio portfolio weights."""
    n = returns.shape[1]
    mu = returns.mean().values * 252
    Sigma = returns.cov().values * 252

    bounds = [(0, 1)] * n if long_only else [(-0.5, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = _optimize(
        lambda w: _neg_sharpe(w, mu, Sigma, rf),
        n, bounds, constraints,
    )
    w = result.x
    w = np.maximum(w, 0) if long_only else w
    return w / w.sum()


def min_variance_portfolio(
    returns: pd.DataFrame,
    long_only: bool = True,
) -> np.ndarray:
    """Global Minimum Variance portfolio."""
    n = returns.shape[1]
    Sigma = returns.cov().values * 252

    bounds = [(0, 1)] * n if long_only else [(-0.5, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = _optimize(
        lambda w: _portfolio_volatility(w, Sigma),
        n, bounds, constraints,
    )
    w = np.maximum(result.x, 0) if long_only else result.x
    return w / w.sum()


def risk_parity_portfolio(returns: pd.DataFrame) -> np.ndarray:
    """Equal Risk Contribution (Risk Parity) portfolio."""
    n = returns.shape[1]
    Sigma = returns.cov().values * 252

    bounds = [(1e-4, 1)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    result = _optimize(
        lambda w: _risk_parity_obj(w, Sigma),
        n, bounds, constraints,
    )
    w = np.maximum(result.x, 0)
    return w / w.sum()


def equal_weight_portfolio(n: int) -> np.ndarray:
    """Simple 1/N equal weight benchmark."""
    return np.ones(n) / n


# ─── Efficient Frontier ───────────────────────────────────────────────────────

def compute_efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 60,
    long_only: bool = True,
) -> pd.DataFrame:
    """
    Trace the efficient frontier by solving min-variance for a range of
    target returns. Returns a DataFrame with columns:
    ['return', 'volatility', 'sharpe'].
    """
    n = returns.shape[1]
    mu = returns.mean().values * 252
    Sigma = returns.cov().values * 252

    # target return range: from min-var return to max individual asset return
    w_mv = min_variance_portfolio(returns, long_only)
    r_min = w_mv @ mu
    r_max = mu.max() * 0.95

    if r_min >= r_max:
        r_min = mu.min()
        r_max = mu.max()

    targets = np.linspace(r_min, r_max, n_points)
    frontier = []

    bounds = [(0, 1)] * n if long_only else [(-0.5, 1)] * n

    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        try:
            res = _optimize(
                lambda w: _portfolio_volatility(w, Sigma),
                n, bounds, constraints,
                n_restarts=2,
            )
            if res.success or res.fun < 1.0:
                vol = _portfolio_volatility(res.x, Sigma)
                ret = res.x @ mu
                sharpe = ret / vol if vol > 0 else 0
                frontier.append({"return": ret, "volatility": vol, "sharpe": sharpe})
        except Exception:
            pass

    return pd.DataFrame(frontier).dropna()


# ─── Monte Carlo Simulation ───────────────────────────────────────────────────

def monte_carlo_portfolios(
    returns: pd.DataFrame,
    n_simulations: int = 3000,
) -> pd.DataFrame:
    """
    Random portfolio simulation for plotting the feasible set.
    Returns DataFrame with ['return', 'volatility', 'sharpe'] for each sim.
    """
    n = returns.shape[1]
    mu = returns.mean().values * 252
    Sigma = returns.cov().values * 252

    records = []
    for _ in range(n_simulations):
        w = np.random.dirichlet(np.ones(n))
        ret = w @ mu
        vol = np.sqrt(w @ Sigma @ w)
        records.append({"return": ret, "volatility": vol, "sharpe": ret / vol if vol > 0 else 0})

    return pd.DataFrame(records)


# ─── Portfolio Turnover ───────────────────────────────────────────────────────

def compute_turnover(
    weights_series: list[np.ndarray],
) -> list[float]:
    """
    Turnover = 0.5 * Σ |w_t - w_{t-1}|  (one-way turnover).
    """
    turnovers = []
    for i in range(1, len(weights_series)):
        turnovers.append(0.5 * np.abs(weights_series[i] - weights_series[i - 1]).sum())
    return turnovers
