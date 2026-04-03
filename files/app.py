"""
app.py — EigenPortfolio: Quant Research Terminal
Main Streamlit application entry point.
Dark quant-terminal UI with custom CSS, tabbed layout, and 80+ visualizations.

Author: Sourish Dey
Portfolio: https://sourishdeyportfolio.vercel.app/
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import itertools
import json
import hashlib
import time

_PLOTLY_CHART_COUNTER = itertools.count()

def _plotly_chart(container, fig, **kwargs):
    key = kwargs.pop("key", None)
    if key is None:
        key = f"plotly_chart_{next(_PLOTLY_CHART_COUNTER)}"

    width = kwargs.pop("width", None)
    use_container_width = kwargs.pop("use_container_width", None)

    if width is None and use_container_width is not None:
        width = "stretch" if use_container_width else "content"
    if width is None:
        width = "stretch"

    try:
        return container.plotly_chart(fig, width=width, key=key, **kwargs)
    except TypeError:
        kwargs.setdefault("use_container_width", width == "stretch")
        try:
            return container.plotly_chart(fig, key=key, **kwargs)
        except TypeError:
            return container.plotly_chart(fig, **kwargs)


def _button(container, label, **kwargs):
    width = kwargs.pop("width", None)
    use_container_width = kwargs.pop("use_container_width", None)

    if width is None and use_container_width is not None:
        width = "stretch" if use_container_width else "content"
    if width is None:
        width = "stretch"

    try:
        return container.button(label, width=width, **kwargs)
    except TypeError:
        kwargs.setdefault("use_container_width", width == "stretch")
        return container.button(label, **kwargs)


def _download_button(container, label, **kwargs):
    width = kwargs.pop("width", None)
    use_container_width = kwargs.pop("use_container_width", None)

    if width is None and use_container_width is not None:
        width = "stretch" if use_container_width else "content"
    if width is None:
        width = "stretch"

    try:
        return container.download_button(label, width=width, **kwargs)
    except TypeError:
        kwargs.setdefault("use_container_width", width == "stretch")
        return container.download_button(label, **kwargs)

# ─── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="EigenPortfolio | Quant Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Module Imports ───────────────────────────────────────────────────────────
# Cache schema (prevents stale `st.cache_data` values after deploys)
CACHE_SCHEMA_VERSION = 2
if st.session_state.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.session_state["cache_schema_version"] = CACHE_SCHEMA_VERSION

from data import (
    fetch_price_data, compute_log_returns, normalize_returns,
    get_rolling_volatility, get_universe_dict, get_sector,
    SECTOR_MAP_SP500,
)
from rmt import run_rmt, get_eigenvector_stability
from portfolio import (
    build_eigenportfolio, build_top_k_eigenportfolio,
    max_sharpe_portfolio, min_variance_portfolio, risk_parity_portfolio,
    equal_weight_portfolio, compute_efficient_frontier, monte_carlo_portfolios,
    annualise_stats,
)
from backtest import run_backtest, compute_rolling_alpha_beta, get_benchmark_returns, compute_metrics
from visuals import (
    plot_price_overlay, plot_candlestick, plot_rolling_returns,
    plot_volatility_clustering, plot_correlation_heatmap, plot_return_distribution,
    plot_rolling_correlation, plot_volume_return_scatter,
    plot_cumulative_returns, plot_rolling_volatility_lines, plot_latest_rolling_corr_heatmap,
    plot_eigenvalue_spectrum, plot_scree, plot_signal_vs_noise,
    plot_mp_fit, plot_corr_before_after, plot_eigenvalue_density_3d,
    plot_eigenvalues_histogram, plot_cumulative_variance_explained, plot_3d_corr_surface,
    plot_drawdown_from_log_returns, plot_rolling_sharpe_series, plot_returns_heatmap, plot_3d_asset_moments,
    plot_3d_time_asset_surface, plot_3d_asset_metrics, plot_3d_pca_loadings,
    plot_3d_portfolio_cloud, plot_3d_frontier_line, plot_3d_backtest_metrics, plot_3d_return_hist_surface,
    plot_3d_matrix_surface,
    plot_eigenvector_weights, plot_sector_exposure, plot_factor_loading_heatmap,
    plot_variance_contribution, plot_eigenvector_comparison,
    plot_efficient_frontier, plot_risk_return_scatter, plot_allocation_pie,
    plot_weight_treemap, plot_weights_bar_compare,
    plot_equity_curves, plot_drawdown_curves, plot_rolling_sharpe,
    plot_alpha_beta, plot_rebalance_impact, plot_monthly_returns_heatmap,
    plot_return_distribution_strategy, plot_rolling_metrics_dashboard,
    plot_turnover_series, plot_weight_drift, plot_eigenvector_stability,
    plot_cumulative_cost,
    plot_3d_risk_return_time, plot_monte_carlo, plot_metrics_radar,
    plot_performance_table,
    THEME,
)
from utils import (
    export_prices_csv, export_returns_csv, export_weights_csv, export_backtest_csv,
    generate_pdf_report, save_portfolio_config, generate_auto_insights, generate_kpi_summary,
    RESEARCH_TEXTS,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Dark Quant Terminal
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

/* ─── Global Reset ─────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0e1a !important;
    color: #e2e8f0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1220 0%, #111827 100%) !important;
    border-right: 1px solid #1f2d45 !important;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* ─── Header Bar ────────────────────────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #0d1220 0%, #111827 50%, #0d1220 100%);
    border: 1px solid #1f2d45;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, #7c3aed, #00d4ff, transparent);
}
.main-header h1 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: #00d4ff !important;
    margin: 0 !important;
    letter-spacing: 2px;
}
.main-header p {
    color: #64748b !important;
    font-size: 0.82rem !important;
    margin: 4px 0 0 0 !important;
    font-family: 'JetBrains Mono', monospace;
}

/* ─── Metric Cards ──────────────────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2236 100%);
    border: 1px solid #1f2d45;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    border-color: #00d4ff;
    transform: translateY(-2px);
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00d4ff, transparent);
    opacity: 0;
    transition: opacity 0.2s;
}
.metric-card:hover::after { opacity: 1; }
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #00d4ff;
}
.metric-value.positive { color: #10b981; }
.metric-value.negative { color: #ef4444; }
.metric-value.neutral  { color: #f59e0b; }
.metric-delta {
    font-size: 0.72rem;
    color: #64748b;
    margin-top: 2px;
    font-family: 'JetBrains Mono', monospace;
}

/* ─── Section Headers ──────────────────────────────────────────────────────── */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 3px;
    padding: 8px 0;
    border-bottom: 1px solid #1f2d45;
    margin: 18px 0 12px 0;
}

/* ─── Info / Insight Cards ─────────────────────────────────────────────────── */
.insight-card {
    background: #111827;
    border: 1px solid #1f2d45;
    border-left: 3px solid #00d4ff;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.85rem;
    line-height: 1.6;
}
.research-box {
    background: #0d1220;
    border: 1px solid #1f2d45;
    border-radius: 10px;
    padding: 20px;
    margin: 12px 0;
}

/* ─── Tabs ──────────────────────────────────────────────────────────────────── */
[data-baseweb="tab-list"] {
    background: #111827 !important;
    border-bottom: 1px solid #1f2d45 !important;
    gap: 4px !important;
}
[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #64748b !important;
    padding: 8px 18px !important;
    border-radius: 6px 6px 0 0 !important;
    letter-spacing: 1px;
    text-transform: uppercase;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #00d4ff !important;
    background: #1a2236 !important;
    border-top: 2px solid #00d4ff !important;
}

/* ─── Buttons ────────────────────────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #1a2236, #0d1220) !important;
    border: 1px solid #1f2d45 !important;
    color: #00d4ff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #00d4ff !important;
    box-shadow: 0 0 12px rgba(0, 212, 255, 0.2) !important;
}

/* ─── Slider + Select ────────────────────────────────────────────────────────── */
.stSlider > div > div > div > div {
    background: #00d4ff !important;
}
.stSelectbox > div > div {
    background: #111827 !important;
    border-color: #1f2d45 !important;
    color: #e2e8f0 !important;
}

/* ─── Expander ────────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid #1f2d45 !important;
    border-radius: 8px !important;
}

/* ─── Dividers ────────────────────────────────────────────────────────────────── */
hr {
    border-color: #1f2d45 !important;
    margin: 16px 0 !important;
}

/* ─── Footer ────────────────────────────────────────────────────────────────── */
.footer-bar {
    background: #0d1220;
    border-top: 1px solid #1f2d45;
    padding: 14px 20px;
    text-align: center;
    margin-top: 40px;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
}
.footer-bar a {
    color: #00d4ff;
    text-decoration: none;
}
.footer-bar a:hover { color: #7c3aed; }

/* ─── Status badge ───────────────────────────────────────────────────────────── */
.status-live {
    display: inline-block;
    background: #10b981;
    color: #0a0e1a;
    font-size: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 700;
    letter-spacing: 1px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.6; }
}

/* ─── Loading bar ─────────────────────────────────────────────────────────────── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00d4ff, #7c3aed) !important;
}

/* ─── Scrollbar ─────────────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1f2d45; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #00d4ff; }

/* ─── Streamlit default overrides ────────────────────────────────────────────── */
.stMarkdown, .stText, p, li, label { color: #e2e8f0 !important; }
h1, h2, h3, h4 { color: #e2e8f0 !important; }
[data-testid="stMetricLabel"] { color: #64748b !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricValue"] { color: #00d4ff !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', monospace !important; }
.block-container { padding-top: 1rem !important; max-width: 1400px !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:12px 0 16px; border-bottom:1px solid #1f2d45; margin-bottom:16px">
        <div style="font-family:'JetBrains Mono',monospace; font-size:1.3rem; color:#00d4ff; font-weight:700; letter-spacing:2px">⚡ EIGEN</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#64748b; letter-spacing:4px">PORTFOLIO TERMINAL</div>
        <div style="margin-top:6px"><span class="status-live">● LIVE</span></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Universe ──────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Universe</div>', unsafe_allow_html=True)
    market = st.selectbox("Market", ["S&P 500", "NIFTY 50"], index=0)
    universe_dict = get_universe_dict(market)
    all_tickers = list(universe_dict.values())
    all_names = list(universe_dict.keys())

    # Multi-select for portfolio universe (default top 30)
    default_names = all_names[:30]
    selected_names = st.multiselect(
        "Select Stocks (Portfolio Universe)",
        options=all_names,
        default=default_names[:20],
        help="Choose assets for the portfolio universe (minimum 5 recommended)",
    )
    selected_tickers = [universe_dict[n] for n in selected_names] if selected_names else [universe_dict[n] for n in default_names[:15]]

    # Deployment-safe fetch limit (live providers are rate-limited in many environments)
    max_fetch_tickers = st.slider(
        "Max Stocks (Fetch)",
        5,
        10,
        10,
        help="Limits how many tickers are fetched from market data providers per run (recommended for deployments).",
    )

    # Compare up to 10 stocks
    st.markdown('<div class="section-header">Compare Stocks</div>', unsafe_allow_html=True)
    compare_names = st.multiselect(
        "Compare (up to 10)",
        options=all_names,
        default=all_names[: min(10, len(all_names))],
        max_selections=10,
    )
    compare_tickers = [universe_dict[n] for n in compare_names] if compare_names else [universe_dict[n] for n in all_names[: min(10, len(all_names))]]

    # ── Timeframe ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Timeframe</div>', unsafe_allow_html=True)
    period_map = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
                  "1 Year": "1y", "2 Years": "2y", "3 Years": "3y", "5 Years": "5y"}
    period_label = st.selectbox("Historical Period", list(period_map.keys()), index=3)
    period = period_map[period_label]

    interval_map = {"Daily": "1d", "Weekly": "1wk"}
    interval_label = st.selectbox("Interval", list(interval_map.keys()), index=0)
    interval = interval_map[interval_label]

    # ── RMT Settings ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">RMT Settings</div>', unsafe_allow_html=True)
    n_signal_override = st.slider("Min Signal PCs", 1, 20, 3, help="Override minimum signal components")

    # ── Eigenportfolio ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Eigenportfolio</div>', unsafe_allow_html=True)
    pc_index = st.slider("Primary PC Index", 0, 9, 0)
    top_k = st.slider("Top-K PCs (Combined)", 1, 10, 3)
    long_only = st.toggle("Long-Only Mode", value=True)

    # ── Backtesting ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Backtesting</div>', unsafe_allow_html=True)
    train_window = st.slider("Training Window (days)", 60, 504, 126)
    rebalance_freq = st.slider("Rebalance Frequency (days)", 5, 63, 21)
    transaction_cost = st.slider("Transaction Cost (bps)", 0, 50, 10) / 10000
    strategies_to_run = st.multiselect(
        "Strategies",
        ["eigenportfolio", "max_sharpe", "min_variance", "risk_parity", "equal_weight"],
        default=["eigenportfolio", "max_sharpe", "equal_weight"],
    )

    # ── Features ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Features</div>', unsafe_allow_html=True)
    research_mode = st.toggle("Research Mode", value=False)
    auto_insights = st.toggle("Auto Insights", value=True)
    auto_fetch_on_change = st.toggle("Auto Fetch on Change", value=True)
    fetch_delay_sec = st.slider("Fetch Delay (sec)", 0.0, 3.0, 2.0, step=0.5, help="Sleep between Yahoo requests to reduce rate-limits.")
    run_mc = st.toggle("Monte Carlo Simulation", value=True)
    n_mc = st.slider("MC Simulations", 500, 5000, 2000, step=500) if run_mc else 2000

    st.markdown("---")
    fetch_btn = _button(st, "🚀  FETCH & ANALYZE", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="main-header">
    <h1>⚡ EIGENPORTFOLIO TERMINAL</h1>
    <p>Quantitative Portfolio Construction via Random Matrix Theory  ·  {market}  ·  {period_label}  ·  {len(selected_tickers)} assets</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

for key in ["prices", "returns", "rmt_result", "backtest_results", "frontier", "mc_df"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Fetch and process all data. Updates session state."""
    # Ordered + capped list for deployment stability
    all_needed = []
    for t in list(selected_tickers) + list(compare_tickers):
        if t and t not in all_needed:
            all_needed.append(t)
    if len(all_needed) > max_fetch_tickers:
        st.warning(f"Fetching limited to first {max_fetch_tickers} tickers (deployment-safe mode).")
        all_needed = all_needed[:max_fetch_tickers]
    
    with st.spinner("Fetching market data..."):
        result = fetch_price_data(
            all_needed,
            period=period,
            interval=interval,
            sleep_seconds=float(fetch_delay_sec),
            allow_stooq_fallback=False,
            allow_alpha_vantage_fallback=True,
            max_alpha_calls=int(max_fetch_tickers),
        )
        if isinstance(result, tuple) and len(result) == 2:
            prices_all, fetch_report = result
        else:
            prices_all = result
            try:
                fetched_cols = set(getattr(prices_all, "columns", []))
            except Exception:
                fetched_cols = set()
            fetch_report = {
                "method": "legacy_cache_df",
                "n_requested": len(all_needed),
                "n_ok": len(fetched_cols) if fetched_cols else 0,
                "failed": [t for t in all_needed if t not in fetched_cols],
            }

        st.session_state["fetch_report"] = fetch_report

    if getattr(prices_all, "empty", True):
        rep = st.session_state.get("fetch_report") or {}
        failed = rep.get("failed") or []
        method = rep.get("method") or "unknown"
        st.error("❌ Failed to fetch market data.")
        st.caption(f"Fetch method: `{method}` | Requested: {rep.get('n_requested')} | Fetched: {rep.get('n_ok')}")
        if rep.get("alpha_vantage_attempted"):
            st.caption("Alpha Vantage attempted: " + ", ".join([str(x) for x in rep.get("alpha_vantage_attempted", [])[:10]]))
        if failed:
            st.caption("Failed tickers (sample): " + ", ".join([str(x) for x in failed[:10]]))
        st.info("If this is a deployment: verify outbound HTTPS access and set `ALPHAVANTAGE_API_KEY` in deployment secrets. Yahoo is tried first, then Alpha Vantage.")
        return False

    # Portfolio universe prices/returns
    port_tickers = [t for t in selected_tickers if t in prices_all.columns]
    if len(port_tickers) < 5:
        st.warning(f"Only {len(port_tickers)} live tickers were fetched. Need ≥5 for RMT. Try fewer symbols, a longer period, or confirm deployment network access.")
        return False

    prices = prices_all[port_tickers].dropna(how="all", axis=1)
    returns = compute_log_returns(prices)

    if len(returns) < 60:
        st.warning("Insufficient data points. Try a longer period.")
        return False

    st.session_state["prices"] = prices
    st.session_state["returns"] = returns
    st.session_state["prices_all"] = prices_all
    st.session_state["port_tickers"] = list(prices.columns)

    # RMT
    with st.spinner("Running Random Matrix Theory analysis..."):
        rmt_result = run_rmt(returns)
        st.session_state["rmt_result"] = rmt_result

    # Backtesting
    backtest_results = {}
    if strategies_to_run:
        prog = st.progress(0, text="Backtesting strategies...")
        for i, strat in enumerate(strategies_to_run):
            prog.progress((i + 1) / len(strategies_to_run), text=f"Backtesting: {strat}...")
            try:
                result = run_backtest(
                    prices, returns, strategy=strat,
                    train_window=train_window, rebalance_freq=rebalance_freq,
                    transaction_cost=transaction_cost, pc_index=pc_index,
                    long_only=long_only,
                )
                backtest_results[strat] = result
            except Exception as e:
                st.warning(f"Backtest failed for {strat}: {e}")
        prog.empty()
    st.session_state["backtest_results"] = backtest_results

    # Efficient frontier
    with st.spinner("Computing efficient frontier..."):
        try:
            frontier = compute_efficient_frontier(returns, n_points=50, long_only=long_only)
            st.session_state["frontier"] = frontier
        except Exception:
            st.session_state["frontier"] = pd.DataFrame()

    # Monte Carlo
    if run_mc:
        with st.spinner("Running Monte Carlo simulation..."):
            try:
                mc_df = monte_carlo_portfolios(returns, n_simulations=n_mc)
                st.session_state["mc_df"] = mc_df
            except Exception:
                st.session_state["mc_df"] = pd.DataFrame()

    return True


def _config_signature() -> str:
    cfg = {
        "market": market,
        "period": period,
        "interval": interval,
        "selected_tickers": list(selected_tickers),
        "compare_tickers": list(compare_tickers),
        "n_signal_override": int(n_signal_override),
        "pc_index": int(pc_index),
        "top_k": int(top_k),
        "train_window": int(train_window),
        "rebalance_freq": int(rebalance_freq),
        "transaction_cost": float(transaction_cost),
        "long_only": bool(long_only),
        "strategies_to_run": list(strategies_to_run),
        "run_mc": bool(run_mc),
        "n_mc": int(n_mc),
        "max_fetch_tickers": int(max_fetch_tickers),
        "fetch_delay_sec": float(fetch_delay_sec),
    }
    raw = json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


current_sig = _config_signature()
prev_sig = st.session_state.get("config_sig")
config_changed = prev_sig is not None and prev_sig != current_sig
st.session_state["config_sig"] = current_sig

if config_changed:
    for key in ["prices", "returns", "rmt_result", "backtest_results", "frontier", "mc_df", "prices_all", "port_tickers"]:
        st.session_state[key] = None


# Auto-fetch on first load
if st.session_state["prices"] is None:
    if auto_fetch_on_change or prev_sig is None:
        load_data()

# Manual fetch button
if fetch_btn:
    # Clear cached data
    for key in ["prices", "returns", "rmt_result", "backtest_results", "frontier", "mc_df"]:
        st.session_state[key] = None
    load_data()


# ═══════════════════════════════════════════════════════════════════════════════
# CHECK DATA READINESS
# ═══════════════════════════════════════════════════════════════════════════════

prices        = st.session_state.get("prices")
returns       = st.session_state.get("returns")
rmt_result    = st.session_state.get("rmt_result")
backtest_results = st.session_state.get("backtest_results") or {}
frontier      = st.session_state.get("frontier")
mc_df         = st.session_state.get("mc_df")
prices_all    = st.session_state.get("prices_all")
port_tickers  = st.session_state.get("port_tickers") or selected_tickers

if prices is None or returns is None or rmt_result is None:
    st.info("👆 Configure settings in the sidebar and click **FETCH & ANALYZE** to begin.")
    st.stop()

# Build sector map for this universe
sector_map = {t: get_sector(t, market) for t in port_tickers}
n_assets = len(port_tickers)


# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW METRICS ROW
# ═══════════════════════════════════════════════════════════════════════════════

mu_all = returns.mean() * 252
vol_all = returns.std() * np.sqrt(252)

best_r = port_tickers[mu_all.values.argmax()] if len(mu_all) else "—"
avg_corr = (rmt_result.correlation_matrix.sum() - n_assets) / max(n_assets * (n_assets - 1), 1)

cols = st.columns(6)
metrics_overview = [
    ("Assets", str(n_assets), ""),
    ("Signal PCs", str(rmt_result.n_signal), f"of {n_assets} eigenvalues"),
    ("Avg Correlation", f"{avg_corr:.3f}", "Pairwise mean"),
    ("T/N Ratio (q)", f"{rmt_result.q:.2f}", "Data quality"),
    ("λ_max", f"{rmt_result.lambda_max:.2f}", "Dominant eigenvalue"),
    ("Market Period", period_label, interval_label),
]
for col, (label, val, delta) in zip(cols, metrics_overview):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{val}</div>
            <div class="metric-delta">{delta}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

if auto_insights:
    kpi = generate_kpi_summary(rmt_result, returns, backtest_results or None)

    with st.expander("📡 RMT / Strategy KPIs", expanded=True):

        cols = st.columns(6)
        kpi_cards = [
            ("RMT Signal PCs", str(kpi["signal_pcs"]), f'{kpi["signal_pct"]:.1f}% above λ⁺={kpi["lambda_plus"]:.3f}'),
            ("Data Quality (q)", f'{kpi["q_t_over_n"]:.2f}', f'T={kpi["observations_t"]} / N={kpi["assets_n"]}'),
            ("Market Factor (PC1)", f'{kpi["pc1_var_pct"]:.1f}%', "Variance explained"),
            ("Avg Pairwise Corr", f'{kpi["avg_pairwise_corr"]:.3f}', "Diversification headroom"),
            ("Top-3 PCs", f'{kpi["top3_var_pct"]:.1f}%', "Total variance explained"),
            ("λ_max", f'{kpi["lambda_max"]:.3f}', "Dominant eigenvalue"),
        ]
        for col, (label, val, delta) in zip(cols, kpi_cards):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{val}</div>
                        <div class="metric-delta">{delta}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        if kpi.get("best_strategy"):
            cols2 = st.columns(3)
            extra = [
                ("Best Strategy", str(kpi["best_strategy"]), f'Sharpe {kpi["best_sharpe"]:.3f} | CAGR {kpi["best_cagr"]:.2%}'),
                ("Deepest Drawdown", str(kpi["worst_drawdown_strategy"]), f'{kpi["worst_drawdown"]:.2%}'),
                ("Signal vs Noise", "Denoised", f'{kpi["signal_pcs"]} signal / {kpi["assets_n"]-kpi["signal_pcs"]} noise PCs'),
            ]
            for col, (label, val, delta) in zip(cols2, extra):
                with col:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">{label}</div>
                            <div class="metric-value">{val}</div>
                            <div class="metric-delta">{delta}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Narrative notes intentionally omitted: KPI-only view.

    with st.expander("🤖 AI Analyst (Gemini)", expanded=False):
        st.caption("Set `GEMINI_API_KEY` in your environment to enable Gemini analysis.")
        user_q = st.text_area(
            "Ask Gemini",
            value="Summarize the universe health and strategy takeaways.",
            height=90,
            key="gemini_question",
        )
        model_name = st.text_input("Model", value="gemini-3-flash-preview", key="gemini_model")
        if st.button("Generate AI Summary", key="gemini_run"):
            try:
                from ai import gemini_generate, build_kpi_prompt

                prompt = build_kpi_prompt(kpi, user_q)
                with st.spinner("Calling Gemini..."):
                    st.session_state["gemini_response"] = gemini_generate(prompt, model=model_name)
            except Exception as e:
                st.error(str(e))

        if st.session_state.get("gemini_response"):
            st.markdown(st.session_state["gemini_response"])


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_3d, tab_overview, tab_rmt, tab_eigen, tab_optim, tab_backtest, tab_export = st.tabs([
    "🧊  3D Lab",
    "📈  Overview",
    "🔬  RMT Analysis",
    "🧮  Eigenportfolios",
    "⚙️  Optimization",
    "📊  Backtesting",
    "📤  Export",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tab_3d:
    st.markdown('<div class="section-header">3D Lab</div>', unsafe_allow_html=True)
    st.caption("25+ interactive 3D diagnostics. Use the toggle to avoid rendering heavy charts unless needed.")

    render_3d = st.toggle("Render 3D Lab charts", value=False, key="render_3d_lab")
    if not render_3d:
        st.info("Enable **Render 3D Lab charts** to display the 3D gallery.")
    else:
        n_assets_3d = st.slider("3D assets (max)", 5, min(25, n_assets), min(12, n_assets), key="n_assets_3d")
        tickers_3d = port_tickers[:n_assets_3d]
        rets_3d = returns[tickers_3d] if tickers_3d else returns.iloc[:, : min(12, returns.shape[1])]
        prices_3d = prices[tickers_3d] if tickers_3d and prices is not None and not prices.empty else prices

        norm_prices = None
        if prices_3d is not None and not prices_3d.empty:
            norm_prices = prices_3d / prices_3d.iloc[0] * 100.0

        labels_3d = [str(c) for c in rets_3d.columns]
        corr_roll = rets_3d.tail(63).corr().values if len(rets_3d) >= 63 else rets_3d.corr().values

        zrets = (rets_3d - rets_3d.mean()) / rets_3d.std().replace(0, np.nan) if not rets_3d.empty else pd.DataFrame()
        cum_rets = (np.exp(rets_3d.cumsum()) - 1.0) if not rets_3d.empty else pd.DataFrame()
        vol21 = rets_3d.rolling(21).std() * np.sqrt(252) if not rets_3d.empty else pd.DataFrame()
        vol63 = rets_3d.rolling(63).std() * np.sqrt(252) if not rets_3d.empty else pd.DataFrame()
        mu63 = rets_3d.rolling(63).mean() * 252 if not rets_3d.empty else pd.DataFrame()
        sharpe63 = mu63 / vol63.replace(0, np.nan) if (not mu63.empty and not vol63.empty) else pd.DataFrame()

        charts = {
            "Price Surface (Indexed)": lambda: plot_3d_time_asset_surface(norm_prices, "Price Surface (Indexed)", z_label="Index", colorscale="Viridis") if norm_prices is not None else None,
            "Return Surface (Log Returns)": lambda: plot_3d_time_asset_surface(rets_3d, "Return Surface (Daily Log Returns)", z_label="Log Return", colorscale="RdBu", zmid=0),
            "Z-Scored Return Surface": lambda: plot_3d_time_asset_surface(zrets, "Z-Scored Return Surface", z_label="Z", colorscale="RdBu", zmid=0),
            "Cumulative Return Surface": lambda: plot_3d_time_asset_surface(cum_rets, "Cumulative Return Surface", z_label="Cumulative", colorscale="Viridis"),
            "Rolling Vol Surface (21d)": lambda: plot_3d_time_asset_surface(vol21, "Rolling Volatility Surface (21d)", z_label="Vol (ann)", colorscale="Viridis"),
            "Rolling Vol Surface (63d)": lambda: plot_3d_time_asset_surface(vol63, "Rolling Volatility Surface (63d)", z_label="Vol (ann)", colorscale="Viridis"),
            "Rolling Sharpe Surface (63d)": lambda: plot_3d_time_asset_surface(sharpe63, "Rolling Sharpe Surface (63d)", z_label="Sharpe", colorscale="RdBu", zmid=0),
            "Rolling Correlation Surface": lambda: plot_3d_matrix_surface(corr_roll, labels_3d, "Rolling Correlation Surface (last window)", "Corr", zmin=-1, zmax=1),
            "Covariance Surface": lambda: plot_3d_matrix_surface(rets_3d.cov().values, labels_3d, "Covariance Surface", "Cov"),
            "Raw Correlation Surface (RMT)": lambda: plot_3d_matrix_surface(rmt_result.correlation_matrix[:n_assets_3d, :n_assets_3d], labels_3d, "Raw Correlation Surface", "Corr", zmin=-1, zmax=1),
            "Cleaned Correlation Surface (RMT)": lambda: plot_3d_matrix_surface(rmt_result.cleaned_corr[:n_assets_3d, :n_assets_3d], labels_3d, "Cleaned Correlation Surface", "Corr", zmin=-1, zmax=1),
            "Correlation Distance Surface": lambda: plot_3d_matrix_surface(1 - rets_3d.corr().values, labels_3d, "Correlation Distance Surface", "1 - Corr", zmin=0, zmax=2),
            "Assets: Vol vs Return vs Kurt (3D)": lambda: plot_3d_asset_metrics(rets_3d, "vol", "return", "kurt", "Assets: Vol vs Return vs Kurtosis (3D)"),
            "Assets: Vol vs Return vs Skew (3D)": lambda: plot_3d_asset_metrics(rets_3d, "vol", "return", "skew", "Assets: Vol vs Return vs Skew (3D)"),
            "Assets: Sharpe vs Return vs MaxDD (3D)": lambda: plot_3d_asset_metrics(rets_3d, "sharpe", "return", "max_dd", "Assets: Sharpe vs Return vs Max Drawdown (3D)"),
            "Assets: Sharpe vs Vol vs VaR5 (3D)": lambda: plot_3d_asset_metrics(rets_3d, "sharpe", "vol", "var5", "Assets: Sharpe vs Vol vs VaR(5%) (3D)"),
            "Assets: VaR5 vs CVaR5 vs Vol (3D)": lambda: plot_3d_asset_metrics(rets_3d, "var5", "cvar5", "vol", "Assets: VaR(5%) vs CVaR(5%) vs Vol (3D)"),
            "Assets: Downside vs Vol vs Return (3D)": lambda: plot_3d_asset_metrics(rets_3d, "downside", "vol", "return", "Assets: Downside Dev vs Vol vs Return (3D)"),
            "Assets: ACF1 vs ACF5 vs Vol (3D)": lambda: plot_3d_asset_metrics(rets_3d, "acf1", "acf5", "vol", "Assets: Autocorr(1) vs Autocorr(5) vs Vol (3D)"),
            "Assets: Skew vs Kurt vs VaR5 (3D)": lambda: plot_3d_asset_metrics(rets_3d, "skew", "kurt", "var5", "Assets: Skew vs Kurt vs VaR(5%) (3D)"),
            "PCA Loadings (PC1/2/3)": lambda: plot_3d_pca_loadings(rmt_result.eigenvectors, tickers_3d, pcs=(0, 1, 2)),
            "PCA Loadings (PC2/3/4)": lambda: plot_3d_pca_loadings(rmt_result.eigenvectors, tickers_3d, pcs=(1, 2, 3), title="PCA Loadings: PC2 vs PC3 vs PC4"),
            "PCA Loadings (PC1/4/6)": lambda: plot_3d_pca_loadings(rmt_result.eigenvectors, tickers_3d, pcs=(0, 3, 5), title="PCA Loadings: PC1 vs PC4 vs PC6"),
            "PCA Loadings (PC5/6/7)": lambda: plot_3d_pca_loadings(rmt_result.eigenvectors, tickers_3d, pcs=(4, 5, 6), title="PCA Loadings: PC5 vs PC6 vs PC7"),
            "Monte Carlo Feasible Set (3D)": lambda: plot_3d_portfolio_cloud(mc_df, "Monte Carlo Feasible Set (3D)") if (mc_df is not None and not mc_df.empty) else None,
            "Efficient Frontier Curve (3D)": lambda: plot_3d_frontier_line(frontier, "Efficient Frontier Curve (3D)") if (frontier is not None and isinstance(frontier, pd.DataFrame) and not frontier.empty) else None,
            "Backtests: CAGR/Vol/MaxDD (3D)": lambda: plot_3d_backtest_metrics(
                {name: {"cagr": r.cagr, "sharpe": r.sharpe, "max_drawdown": r.max_drawdown, "volatility": r.volatility, "calmar": r.calmar} for name, r in backtest_results.items()},
                "cagr",
                "volatility",
                "max_drawdown",
                "Backtests: CAGR vs Vol vs Max Drawdown (3D)",
            ) if backtest_results else None,
            "Backtests: Sharpe/CAGR/Calmar (3D)": lambda: plot_3d_backtest_metrics(
                {name: {"cagr": r.cagr, "sharpe": r.sharpe, "max_drawdown": r.max_drawdown, "volatility": r.volatility, "calmar": r.calmar} for name, r in backtest_results.items()},
                "sharpe",
                "cagr",
                "calmar",
                "Backtests: Sharpe vs CAGR vs Calmar (3D)",
            ) if backtest_results else None,
            "Return Distribution Surface (3D)": lambda: plot_3d_return_hist_surface(rets_3d, bins=45, title="Return Distribution Surface (3D)"),
        }

        view_mode = st.radio("3D Render Mode", ["Single (Recommended)", "Multi (Max 6)"], horizontal=True, key="lab_3d_mode")
        options = list(charts.keys())
        if view_mode.startswith("Single"):
            choice = st.selectbox("3D View", options, index=options.index("Rolling Correlation Surface") if "Rolling Correlation Surface" in options else 0, key="lab_3d_single")
            fig = charts[choice]()
            if fig is None or (isinstance(fig, go.Figure) and len(fig.data) == 0):
                st.warning("No data available for this view with the current settings.")
            else:
                _plotly_chart(st, fig)
        else:
            default_multi = [o for o in ["Rolling Correlation Surface", "PCA Loadings (PC1/2/3)", "Return Surface (Log Returns)"] if o in options]
            picks = st.multiselect("3D Views (limit 6)", options=options, default=default_multi, key="lab_3d_multi")
            picks = picks[:6]
            for name in picks:
                fig = charts[name]()
                if fig is None or (isinstance(fig, go.Figure) and len(fig.data) == 0):
                    st.warning(f"Skipped `{name}` (no data).")
                else:
                    _plotly_chart(st, fig)


with tab_overview:
    if research_mode:
        with st.expander("📚 What is this terminal?", expanded=False):
            st.markdown("""
            <div class="research-box">
            This terminal applies <strong>Random Matrix Theory (RMT)</strong> to construct 
            statistically robust portfolios. It distinguishes <em>genuine risk factors</em> 
            from <em>noise</em> in the correlation matrix and builds <strong>Eigenportfolios</strong> 
            from the signal subspace.
            </div>
            """, unsafe_allow_html=True)

    # Compare stocks section
    st.markdown('<div class="section-header">Price Comparison</div>', unsafe_allow_html=True)
    avail_compare = [t for t in compare_tickers if prices_all is not None and t in prices_all.columns]
    if avail_compare:
        compare_prices = prices_all[avail_compare].dropna(how="all", axis=1)
        if not compare_prices.empty:
            _plotly_chart(
                st,
                plot_price_overlay(compare_prices, "Normalized Price Comparison (Base 100)"),
            )

            # Individual candlestick
            c1, c2 = st.columns(2)
            for i, t in enumerate(avail_compare[:2]):
                if t in compare_prices.columns:
                    _plotly_chart((c1 if i == 0 else c2), plot_candlestick(compare_prices, t))

    # Rolling Returns
    st.markdown('<div class="section-header">Rolling Returns & Volatility</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    roll_win = 21
    compare_rets = compute_log_returns(compare_prices) if avail_compare and not compare_prices.empty else returns
    _plotly_chart(c1, plot_rolling_returns(compare_rets.iloc[:, :10], roll_win))
    _plotly_chart(c2, plot_volatility_clustering(compare_rets))

    # Distribution
    st.markdown('<div class="section-header">Return Distributions</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    _plotly_chart(c1, plot_return_distribution(compare_rets.iloc[:, :10]))
    _plotly_chart(c2, plot_rolling_correlation(compare_rets))

    # Pairwise scatter
    st.markdown('<div class="section-header">Pairwise Return Scatter</div>', unsafe_allow_html=True)
    _plotly_chart(st, plot_volume_return_scatter(compare_rets.iloc[:, :10]))

    # Correlation heatmap (portfolio universe)
    st.markdown('<div class="section-header">Portfolio Universe Correlation</div>', unsafe_allow_html=True)
    n_heat = min(30, len(port_tickers))
    _plotly_chart(
        st,
        plot_correlation_heatmap(
            rmt_result.correlation_matrix[:n_heat, :n_heat],
            port_tickers[:n_heat],
            title="Correlation Matrix — Portfolio Universe",
        ),
    )

    # Risk-return of individual assets
    st.markdown('<div class="section-header">Asset Risk-Return</div>', unsafe_allow_html=True)
    _plotly_chart(st, plot_risk_return_scatter(returns, sector_map))

    show_gallery_overview = st.toggle("Show visualization gallery (Overview)", value=False, key="gallery_overview")
    if show_gallery_overview:
        st.markdown('<div class="section-header">Overview Gallery</div>', unsafe_allow_html=True)
        subset_rets = compare_rets.iloc[:, : min(6, compare_rets.shape[1])]
        c1, c2 = st.columns(2)
        _plotly_chart(c1, plot_cumulative_returns(subset_rets, "Cumulative Returns (Selected)"))
        _plotly_chart(c2, plot_rolling_volatility_lines(subset_rets, window=21))

        c1, c2 = st.columns(2)
        _plotly_chart(c1, plot_rolling_volatility_lines(subset_rets, window=63))
        _plotly_chart(c2, plot_latest_rolling_corr_heatmap(subset_rets, window=63))

        _plotly_chart(st, plot_rolling_returns(subset_rets, window=63))

        corr_labels = [str(c) for c in subset_rets.columns]
        corr_mat = subset_rets.corr().values if not subset_rets.empty else np.array([])
        _plotly_chart(st, plot_3d_corr_surface(corr_mat, corr_labels, title="Correlation Surface (Selected, 3D)"))

        heavy_overview = st.toggle("Show heavy gallery (50+ charts)", value=False, key="gallery_overview_heavy")
        if heavy_overview:
            st.markdown('<div class="section-header">Candlestick Wall (up to 10)</div>', unsafe_allow_html=True)
            candle_tickers = [t for t in compare_tickers if prices_all is not None and t in prices_all.columns][:10]
            if candle_tickers:
                cols = st.columns(5)
                for i, t in enumerate(candle_tickers):
                    _plotly_chart(cols[i % 5], plot_candlestick(prices_all[candle_tickers], t))

            st.markdown('<div class="section-header">Diagnostics (up to 10)</div>', unsafe_allow_html=True)
            diag_rets = compare_rets.iloc[:, : min(10, compare_rets.shape[1])]
            if not diag_rets.empty:
                _plotly_chart(st, plot_returns_heatmap(diag_rets, "Returns Heatmap (Selected)"))
                _plotly_chart(st, plot_3d_asset_moments(diag_rets))

                # 10× Rolling Vol + 10× Drawdown + 10× Rolling Sharpe + 10× Distribution
                st.markdown('<div class="section-header">Per-Asset Panels</div>', unsafe_allow_html=True)
                for col_name in diag_rets.columns[:10]:
                    lr = diag_rets[col_name].dropna()
                    if lr.empty:
                        continue

                    st.markdown(f"**{col_name}**")
                    c1, c2 = st.columns(2)
                    _plotly_chart(c1, plot_volatility_clustering(diag_rets, ticker=col_name))
                    _plotly_chart(c2, plot_drawdown_from_log_returns(lr, title=f"{col_name} Drawdown"))

                    c1, c2 = st.columns(2)
                    _plotly_chart(c1, plot_rolling_sharpe_series(lr, window=63, title=f"{col_name} Rolling Sharpe (63d)"))
                    _plotly_chart(c2, plot_return_distribution(pd.DataFrame({str(col_name): lr})))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: RMT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_rmt:
    if research_mode:
        st.markdown(f'<div class="research-box">{RESEARCH_TEXTS["rmt_intro"]}</div>', unsafe_allow_html=True)

    # Key RMT metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">q = T/N</div>
        <div class="metric-value">{rmt_result.q:.3f}</div><div class="metric-delta">Data quality ratio</div></div>""",
        unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">λ⁺ (MP Upper Edge)</div>
        <div class="metric-value">{rmt_result.lambda_plus:.4f}</div><div class="metric-delta">Noise threshold</div></div>""",
        unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">λ_max (Empirical)</div>
        <div class="metric-value">{rmt_result.lambda_max:.4f}</div><div class="metric-delta">{rmt_result.lambda_max / rmt_result.lambda_plus:.1f}× above MP edge</div></div>""",
        unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card"><div class="metric-label">σ² (Noise Fit)</div>
        <div class="metric-value">{rmt_result.sigma2:.4f}</div><div class="metric-delta">MP variance parameter</div></div>""",
        unsafe_allow_html=True)

    st.markdown("---")

    # Main RMT plots
    st.markdown('<div class="section-header">Eigenvalue Spectrum vs Marchenko-Pastur</div>', unsafe_allow_html=True)
    _plotly_chart(
        st,
        plot_eigenvalue_spectrum(
            rmt_result.eigenvalues, rmt_result.mp_x, rmt_result.mp_pdf,
            rmt_result.lambda_plus, rmt_result.n_signal,
        ),
    )

    c1, c2 = st.columns(2)
    _plotly_chart(c1, plot_scree(rmt_result.eigenvalues, rmt_result.n_signal))
    _plotly_chart(c2, plot_signal_vs_noise(rmt_result.eigenvalues, rmt_result.lambda_plus, rmt_result.n_signal))

    c1, c2 = st.columns(2)
    _plotly_chart(c1, plot_mp_fit(rmt_result.mp_x, rmt_result.mp_pdf, rmt_result.q, rmt_result.sigma2))
    _plotly_chart(c2, plot_variance_contribution(rmt_result.eigenvalues, rmt_result.n_signal))

    # Correlation before/after denoising
    st.markdown('<div class="section-header">Covariance Denoising</div>', unsafe_allow_html=True)
    if research_mode:
        st.markdown(f'<div class="research-box">{RESEARCH_TEXTS["denoising"]}</div>', unsafe_allow_html=True)
    n_show = min(20, n_assets)
    _plotly_chart(
        st,
        plot_corr_before_after(
            rmt_result.correlation_matrix[:n_show, :n_show],
            rmt_result.cleaned_corr[:n_show, :n_show],
            port_tickers[:n_show],
        ),
    )

    # 3D MP surface
    st.markdown('<div class="section-header">3D Marchenko-Pastur Surface</div>', unsafe_allow_html=True)
    _plotly_chart(st, plot_eigenvalue_density_3d(rmt_result.eigenvalues, rmt_result.q))

    # Eigenvector stability
    st.markdown('<div class="section-header">Eigenvector Stability Over Time</div>', unsafe_allow_html=True)
    with st.spinner("Computing eigenvector stability..."):
        stab_df = get_eigenvector_stability(returns, n_windows=8, top_k=min(5, rmt_result.n_signal))
    if not stab_df.empty:
        _plotly_chart(st, plot_eigenvector_stability(stab_df))
    else:
        st.info("Insufficient data for stability analysis.")

    show_gallery_rmt = st.toggle("Show visualization gallery (RMT)", value=False, key="gallery_rmt")
    if show_gallery_rmt:
        st.markdown('<div class="section-header">RMT Gallery</div>', unsafe_allow_html=True)
        _plotly_chart(st, plot_eigenvalues_histogram(rmt_result.eigenvalues, lambda_plus=rmt_result.lambda_plus))

        c1, c2 = st.columns(2)
        _plotly_chart(c1, plot_cumulative_variance_explained(rmt_result.eigenvalues, n_mark=rmt_result.n_signal))
        _plotly_chart(c2, plot_signal_vs_noise(rmt_result.eigenvalues, rmt_result.lambda_plus, rmt_result.n_signal))

        _plotly_chart(st, plot_mp_fit(rmt_result.mp_x, rmt_result.mp_pdf, rmt_result.q, rmt_result.sigma2))

        n_show_g = min(15, n_assets)
        _plotly_chart(
            st,
            plot_correlation_heatmap(
                rmt_result.cleaned_corr[:n_show_g, :n_show_g],
                port_tickers[:n_show_g],
                title="Cleaned Correlation Matrix",
            ),
        )
        _plotly_chart(
            st,
            plot_3d_corr_surface(
                rmt_result.cleaned_corr[:n_show_g, :n_show_g],
                port_tickers[:n_show_g],
                title="Cleaned Correlation Surface (3D)",
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: EIGENPORTFOLIOS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_eigen:
    if research_mode:
        st.markdown(f'<div class="research-box">{RESEARCH_TEXTS["eigenportfolio"]}</div>', unsafe_allow_html=True)

    # Build eigenportfolios
    w_eigen_single = build_eigenportfolio(rmt_result.eigenvectors, port_tickers, pc_index, long_only)
    w_eigen_topk   = build_top_k_eigenportfolio(rmt_result.eigenvectors, rmt_result.eigenvalues,
                                                  port_tickers, top_k, long_only)

    ret_single, vol_single, sr_single = annualise_stats(returns, w_eigen_single)
    ret_topk,   vol_topk,   sr_topk   = annualise_stats(returns, w_eigen_topk)

    # Metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics_ep = [
        (f"PC{pc_index+1} Return", f"{ret_single:.2%}", "Ann."),
        (f"PC{pc_index+1} Vol",    f"{vol_single:.2%}", "Ann."),
        (f"PC{pc_index+1} Sharpe", f"{sr_single:.3f}",  ""),
        (f"Top-{top_k} Return",    f"{ret_topk:.2%}",   "Ann."),
        (f"Top-{top_k} Vol",       f"{vol_topk:.2%}",   "Ann."),
        (f"Top-{top_k} Sharpe",    f"{sr_topk:.3f}",    ""),
    ]
    for col, (lbl, val, delta) in zip([c1,c2,c3,c4,c5,c6], metrics_ep):
        with col:
            cls = "positive" if "Return" in lbl or "Sharpe" in lbl else "neutral"
            st.markdown(f"""<div class="metric-card"><div class="metric-label">{lbl}</div>
            <div class="metric-value {cls}">{val}</div><div class="metric-delta">{delta}</div></div>""",
            unsafe_allow_html=True)

    st.markdown("---")

    # Single eigenvector weights
    st.markdown(f'<div class="section-header">PC{pc_index+1} Eigenvector Weights</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.5, 1])
    _plotly_chart(c1, plot_eigenvector_weights(w_eigen_single, port_tickers, pc_index, sector_map))
    _plotly_chart(c2, plot_allocation_pie(w_eigen_single, port_tickers, f"PC{pc_index+1} Eigenportfolio"))

    # Sector exposure
    st.markdown('<div class="section-header">Sector Exposure</div>', unsafe_allow_html=True)
    _plotly_chart(st, plot_sector_exposure(w_eigen_single, port_tickers, sector_map))

    # Factor loading heatmap
    st.markdown('<div class="section-header">Factor Loading Heatmap</div>', unsafe_allow_html=True)
    n_factors = min(8, rmt_result.n_signal + 2)
    n_show2 = min(25, n_assets)
    _plotly_chart(
        st,
        plot_factor_loading_heatmap(rmt_result.eigenvectors[:n_show2, :], port_tickers[:n_show2], n_factors),
    )

    # Eigenvector comparison radar
    st.markdown('<div class="section-header">Multi-PC Eigenvector Profile</div>', unsafe_allow_html=True)
    n_compare_pcs = min(4, rmt_result.n_signal)
    _plotly_chart(
        st,
        plot_eigenvector_comparison(
            rmt_result.eigenvectors[:min(12, n_assets), :],
            port_tickers[:min(12, n_assets)],
            list(range(n_compare_pcs)),
        ),
    )

    # Top-K combined portfolio
    st.markdown(f'<div class="section-header">Top-{top_k} Combined Eigenportfolio</div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.5, 1])
    _plotly_chart(c1, plot_eigenvector_weights(w_eigen_topk, port_tickers, pc_index, sector_map))
    _plotly_chart(c2, plot_weight_treemap(w_eigen_topk, port_tickers, sector_map))

    # Weight comparison across PCs
    st.markdown('<div class="section-header">Weight Comparison Across PCs</div>', unsafe_allow_html=True)
    pc_weights = {}
    for pci in range(min(4, n_assets)):
        w = build_eigenportfolio(rmt_result.eigenvectors, port_tickers, pci, long_only)
        pc_weights[f"PC{pci+1}"] = w
    pc_weights[f"Top-{top_k}"] = w_eigen_topk
    n_show3 = min(20, n_assets)
    reduced_pcs = {k: v[:n_show3] for k, v in pc_weights.items()}
    _plotly_chart(st, plot_weights_bar_compare(reduced_pcs, port_tickers[:n_show3]))

    show_gallery_eigen = st.toggle("Show visualization gallery (Eigenportfolios)", value=False, key="gallery_eigen")
    if show_gallery_eigen:
        st.markdown('<div class="section-header">Eigenportfolios Gallery</div>', unsafe_allow_html=True)

        top_n = min(25, n_assets)
        w_df = {
            f"PC{pc_index+1}": w_eigen_single[:top_n],
            f"Top-{top_k}": w_eigen_topk[:top_n],
        }
        _plotly_chart(st, plot_weights_bar_compare(w_df, port_tickers[:top_n]))

        c1, c2 = st.columns(2)
        _plotly_chart(c1, plot_weight_treemap(w_eigen_single, port_tickers, sector_map))
        _plotly_chart(c2, plot_allocation_pie(w_eigen_topk, port_tickers, f"Top-{top_k} Allocation"))

        c1, c2 = st.columns(2)
        _plotly_chart(c1, plot_sector_exposure(w_eigen_topk, port_tickers, sector_map))
        _plotly_chart(c2, plot_factor_loading_heatmap(rmt_result.eigenvectors[:top_n, :], port_tickers[:top_n], n_factors=min(10, n_assets)))

        # Eigenportfolio vs EW cumulative returns
        try:
            ew = equal_weight_portfolio(n_assets)
            ep_series = (returns @ w_eigen_single).rename(f"PC{pc_index+1}")
            ew_series = (returns @ ew).rename("Equal Weight")
            _plotly_chart(st, plot_cumulative_returns(pd.concat([ep_series, ew_series], axis=1), "Cumulative: Eigenportfolio vs Equal Weight"))
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

with tab_optim:
    st.markdown('<div class="section-header">Portfolio Weights</div>', unsafe_allow_html=True)

    with st.spinner("Computing optimized portfolios..."):
        try:
            w_ms  = max_sharpe_portfolio(returns, long_only)
            w_mv  = min_variance_portfolio(returns, long_only)
            w_rp  = risk_parity_portfolio(returns)
            w_ew  = equal_weight_portfolio(n_assets)
        except Exception as e:
            st.error(f"Optimization error: {e}")
            w_ms = w_mv = w_rp = w_ew = equal_weight_portfolio(n_assets)

    # Compute stats
    def safe_stats(w):
        try: return annualise_stats(returns, w)
        except: return (0.0, 0.0, 0.0)

    stats_ms = safe_stats(w_ms)
    stats_mv = safe_stats(w_mv)
    stats_rp = safe_stats(w_rp)
    stats_ew = safe_stats(w_ew)
    stats_ep = safe_stats(w_eigen_single)

    # Metrics table row
    opt_metrics = {
        "⭐ Max Sharpe":   {"cagr": stats_ms[0], "volatility": stats_ms[1], "sharpe": stats_ms[2], "max_drawdown": -0.1, "calmar": stats_ms[0]/0.1},
        "🔷 Min Variance": {"cagr": stats_mv[0], "volatility": stats_mv[1], "sharpe": stats_mv[2], "max_drawdown": -0.1, "calmar": stats_mv[0]/0.1},
        "⚖️ Risk Parity":  {"cagr": stats_rp[0], "volatility": stats_rp[1], "sharpe": stats_rp[2], "max_drawdown": -0.1, "calmar": stats_rp[0]/0.1},
        "📊 Equal Weight": {"cagr": stats_ew[0], "volatility": stats_ew[1], "sharpe": stats_ew[2], "max_drawdown": -0.1, "calmar": stats_ew[0]/0.1},
        f"⚡ EigenPC{pc_index+1}": {"cagr": stats_ep[0], "volatility": stats_ep[1], "sharpe": stats_ep[2], "max_drawdown": -0.1, "calmar": stats_ep[0]/0.1},
    }

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, (name, m) in zip([c1,c2,c3,c4,c5], opt_metrics.items()):
        with col:
            st.markdown(f"""<div class="metric-card">
            <div class="metric-label">{name}</div>
            <div class="metric-value positive">{m['cagr']:.2%}</div>
            <div class="metric-delta">Sharpe: {m['sharpe']:.3f} | Vol: {m['volatility']:.2%}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Efficient Frontier
    st.markdown('<div class="section-header">Efficient Frontier</div>', unsafe_allow_html=True)
    special_pts = {
        "⭐ Max Sharpe":   (stats_ms[0], stats_ms[1]),
        "🔷 Min Var":      (stats_mv[0], stats_mv[1]),
        "⚖️ Risk Parity":  (stats_rp[0], stats_rp[1]),
        "📊 Equal Weight": (stats_ew[0], stats_ew[1]),
    }
    if frontier is not None and not frontier.empty:
        _plotly_chart(st, plot_efficient_frontier(frontier, special_pts))

    # Monte Carlo
    if run_mc and mc_df is not None and not mc_df.empty:
        st.markdown('<div class="section-header">Monte Carlo Simulation</div>', unsafe_allow_html=True)
        _plotly_chart(
            st,
            plot_monte_carlo(mc_df, frontier if frontier is not None else pd.DataFrame(), special_pts),
        )

    # Weight comparison
    st.markdown('<div class="section-header">Weight Comparison</div>', unsafe_allow_html=True)
    n_s = min(20, n_assets)
    all_weights = {
        "Max Sharpe":   w_ms[:n_s],
        "Min Variance": w_mv[:n_s],
        "Risk Parity":  w_rp[:n_s],
        "Equal Weight": w_ew[:n_s],
        f"EigenPC{pc_index+1}": w_eigen_single[:n_s],
    }
    _plotly_chart(st, plot_weights_bar_compare(all_weights, port_tickers[:n_s]))

    # Individual allocation pies
    st.markdown('<div class="section-header">Allocation Breakdown</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    _plotly_chart(c1, plot_allocation_pie(w_ms, port_tickers, "Max Sharpe Allocation"))
    _plotly_chart(c2, plot_allocation_pie(w_mv, port_tickers, "Min Variance Allocation"))
    _plotly_chart(c3, plot_allocation_pie(w_rp, port_tickers, "Risk Parity Allocation"))

    # Performance radar
    st.markdown('<div class="section-header">Strategy Radar</div>', unsafe_allow_html=True)
    if backtest_results:
        radar_metrics = {
            name: {"cagr": r.cagr, "sharpe": r.sharpe, "max_drawdown": r.max_drawdown,
                   "volatility": r.volatility, "calmar": r.calmar}
            for name, r in backtest_results.items()
        }
        _plotly_chart(st, plot_metrics_radar(radar_metrics))

    show_gallery_optim = st.toggle("Show visualization gallery (Optimization)", value=False, key="gallery_optim")
    if show_gallery_optim:
        st.markdown('<div class="section-header">Optimization Gallery</div>', unsafe_allow_html=True)

        strat_rets = None
        try:
            strat_rets = pd.DataFrame({
                "Max Sharpe": returns @ w_ms,
                "Min Variance": returns @ w_mv,
                "Risk Parity": returns @ w_rp,
                "Equal Weight": returns @ w_ew,
            })
        except Exception:
            strat_rets = None

        if strat_rets is not None and not strat_rets.empty:
            _plotly_chart(st, plot_cumulative_returns(strat_rets, "Cumulative Returns (Strategies)"))
            c1, c2 = st.columns(2)
            _plotly_chart(c1, plot_rolling_volatility_lines(strat_rets, window=21, title="21d Rolling Volatility (Strategies)"))
            _plotly_chart(c2, plot_latest_rolling_corr_heatmap(strat_rets, window=63, title="Strategy Correlation (Rolling)"))

            labels = [str(c) for c in strat_rets.columns]
            _plotly_chart(st, plot_3d_corr_surface(strat_rets.corr().values, labels, title="Strategy Correlation Surface (3D)"))

        # RMT context overlays
        _plotly_chart(st, plot_cumulative_variance_explained(rmt_result.eigenvalues, n_mark=rmt_result.n_signal))
        _plotly_chart(st, plot_eigenvalues_histogram(rmt_result.eigenvalues, lambda_plus=rmt_result.lambda_plus))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: BACKTESTING
# ═══════════════════════════════════════════════════════════════════════════════

with tab_backtest:
    if not backtest_results:
        st.info("No backtest results. Select strategies in the sidebar and click FETCH & ANALYZE.")
    else:
        # Performance summary table
        st.markdown('<div class="section-header">Performance Summary</div>', unsafe_allow_html=True)
        bt_metrics = {
            name: {"cagr": r.cagr, "sharpe": r.sharpe, "max_drawdown": r.max_drawdown,
                   "volatility": r.volatility, "calmar": r.calmar}
            for name, r in backtest_results.items()
        }
        _plotly_chart(st, plot_performance_table(bt_metrics))

        # Metrics cards
        c_list = st.columns(min(len(backtest_results), 5))
        for col, (name, r) in zip(c_list, backtest_results.items()):
            with col:
                st.markdown(f"""<div class="metric-card">
                <div class="metric-label">{name}</div>
                <div class="metric-value {'positive' if r.cagr > 0 else 'negative'}">{r.cagr:.2%}</div>
                <div class="metric-delta">SR: {r.sharpe:.2f} | MDD: {r.max_drawdown:.2%}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Equity curves
        st.markdown('<div class="section-header">Equity Curves</div>', unsafe_allow_html=True)
        _plotly_chart(st, plot_equity_curves(backtest_results))

        # Drawdowns
        st.markdown('<div class="section-header">Drawdown Analysis</div>', unsafe_allow_html=True)
        _plotly_chart(st, plot_drawdown_curves(backtest_results))

        # Rolling metrics
        st.markdown('<div class="section-header">Rolling Sharpe</div>', unsafe_allow_html=True)
        _plotly_chart(st, plot_rolling_sharpe(backtest_results))

        # Pick best strategy for detailed dashboard
        best_name = max(backtest_results, key=lambda x: backtest_results[x].sharpe)
        best_result = backtest_results[best_name]

        st.markdown(f'<div class="section-header">Rolling Dashboard — {best_name}</div>', unsafe_allow_html=True)
        _plotly_chart(st, plot_rolling_metrics_dashboard(best_result))

        # Monthly returns heatmap
        st.markdown('<div class="section-header">Monthly Returns Heatmap</div>', unsafe_allow_html=True)
        for name, r in backtest_results.items():
            _plotly_chart(st, plot_monthly_returns_heatmap(r.daily_returns, name))

        # Return distribution comparison
        st.markdown('<div class="section-header">Return Distribution Comparison</div>', unsafe_allow_html=True)
        _plotly_chart(st, plot_return_distribution_strategy(backtest_results))

        # Alpha / Beta (vs equal-weight benchmark)
        st.markdown('<div class="section-header">Rolling Alpha & Beta</div>', unsafe_allow_html=True)
        bench_rets = get_benchmark_returns(prices)
        for name, r in backtest_results.items():
            if name == "equal_weight":
                continue
            ab = compute_rolling_alpha_beta(r.daily_returns, bench_rets)
            if not ab.empty:
                st.caption(f"Alpha/Beta vs Equal-Weight Benchmark — {name}")
                _plotly_chart(st, plot_alpha_beta(ab))

        # Rebalance impact
        st.markdown('<div class="section-header">Rebalance Events</div>', unsafe_allow_html=True)
        _plotly_chart(st, plot_rebalance_impact(best_result, prices))

        # Turnover analysis
        st.markdown('<div class="section-header">Turnover Analysis</div>', unsafe_allow_html=True)
        for name, r in backtest_results.items():
            if r.turnover_series:
                c1, c2 = st.columns(2)
                _plotly_chart(c1, plot_turnover_series(r, name))
                _plotly_chart(c2, plot_cumulative_cost(r, transaction_cost))
                break

        # Weight drift
        st.markdown('<div class="section-header">Weight Drift Over Time</div>', unsafe_allow_html=True)
        _plotly_chart(st, plot_weight_drift(best_result, port_tickers))

        # 3D surface
        st.markdown('<div class="section-header">3D Performance Surface</div>', unsafe_allow_html=True)
        _plotly_chart(st, plot_3d_risk_return_time(backtest_results))

        # Performance radar
        st.markdown('<div class="section-header">Strategy Radar</div>', unsafe_allow_html=True)
        _plotly_chart(st, plot_metrics_radar(bt_metrics))

        show_gallery_bt = st.toggle("Show visualization gallery (Backtesting)", value=False, key="gallery_backtest")
        if show_gallery_bt:
            st.markdown('<div class="section-header">Backtesting Gallery</div>', unsafe_allow_html=True)
            try:
                strat_lr = pd.DataFrame({name: np.log1p(r.daily_returns) for name, r in backtest_results.items()})
            except Exception:
                strat_lr = pd.DataFrame()

            if not strat_lr.empty:
                _plotly_chart(st, plot_cumulative_returns(strat_lr, "Cumulative Returns (Backtests)"))
                c1, c2 = st.columns(2)
                _plotly_chart(c1, plot_rolling_volatility_lines(strat_lr, window=21, title="21d Rolling Volatility (Backtests)"))
                _plotly_chart(c2, plot_latest_rolling_corr_heatmap(strat_lr, window=63, title="Backtest Correlation (Rolling)"))
                labels = [str(c) for c in strat_lr.columns]
                _plotly_chart(st, plot_3d_corr_surface(strat_lr.corr().values, labels, title="Backtest Correlation Surface (3D)"))

            # RMT context snapshot for this backtest universe
            _plotly_chart(st, plot_cumulative_variance_explained(rmt_result.eigenvalues, n_mark=rmt_result.n_signal))
            _plotly_chart(st, plot_eigenvalues_histogram(rmt_result.eigenvalues, lambda_plus=rmt_result.lambda_plus))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

with tab_export:
    st.markdown('<div class="section-header">Data Export</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        _download_button(st,
            "⬇️  Download Price Data (CSV)",
            data=export_prices_csv(prices),
            file_name="eigenportfolio_prices.csv",
            mime="text/csv",
            use_container_width=True,
        )
        _download_button(st,
            "⬇️  Download Log Returns (CSV)",
            data=export_returns_csv(returns),
            file_name="eigenportfolio_returns.csv",
            mime="text/csv",
            use_container_width=True,
        )
        _download_button(st,
            "⬇️  Download Eigenportfolio Weights (CSV)",
            data=export_weights_csv(w_eigen_single, port_tickers, f"pc{pc_index+1}"),
            file_name=f"eigenportfolio_pc{pc_index+1}_weights.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        if backtest_results:
            best_name2 = max(backtest_results, key=lambda x: backtest_results[x].sharpe)
            best2 = backtest_results[best_name2]
            _download_button(st,
                f"⬇️  Download Backtest Results — {best_name2} (CSV)",
                data=export_backtest_csv(best2.equity_curve, best2.daily_returns),
                file_name=f"backtest_{best_name2}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        # PDF Report
        st.markdown("---")
        if _button(st, "📄  Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                rmt_info = {
                    "Assets (N)": str(n_assets),
                    "Observations (T)": str(len(returns)),
                    "q = T/N": f"{rmt_result.q:.4f}",
                    "MP Upper Edge λ⁺": f"{rmt_result.lambda_plus:.4f}",
                    "Signal PCs": str(rmt_result.n_signal),
                    "Noise PCs": str(rmt_result.n_noise),
                    "λ_max": f"{rmt_result.lambda_max:.4f}",
                }
                metrics_for_pdf = bt_metrics if backtest_results else {"Current": {"cagr": stats_ms[0], "sharpe": stats_ms[2], "max_drawdown": -0.1, "volatility": stats_ms[1], "calmar": 0.0}}
                pdf_bytes = generate_pdf_report(
                    f"EigenPortfolio — {market} | {period_label}",
                    metrics_for_pdf, w_eigen_single, port_tickers, rmt_info,
                )
                _download_button(st,
                    "⬇️  Download PDF Report",
                    data=pdf_bytes,
                    file_name="eigenportfolio_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    st.markdown("---")
    st.markdown('<div class="section-header">Save/Load Portfolio Config</div>', unsafe_allow_html=True)
    config = {
        "market": market,
        "period": period,
        "interval": interval,
        "selected_tickers": port_tickers,
        "pc_index": pc_index,
        "top_k": top_k,
        "long_only": long_only,
        "train_window": train_window,
        "rebalance_freq": rebalance_freq,
        "transaction_cost_bps": int(transaction_cost * 10000),
    }
    config_json = save_portfolio_config(config)
    _download_button(st,
        "⬇️  Save Portfolio Config (JSON)",
        data=config_json,
        file_name="eigenportfolio_config.json",
        mime="application/json",
        use_container_width=True,
    )

    st.text_area("Config Preview (JSON)", value=config_json, height=200)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="footer-bar">
    <span style="color:#64748b">Built with ⚡ by </span>
    <a href="https://sourishdeyportfolio.vercel.app/" target="_blank">
        <strong style="color:#00d4ff">Sourish Dey</strong>
    </a>
    <span style="color:#64748b"> · EigenPortfolio Quant Terminal · Powered by RMT + Streamlit</span>
    <br>
    <span style="font-size:0.65rem; color:#1f2d45">
        Disclaimer: For research purposes only. Not financial advice.
    </span>
</div>
""", unsafe_allow_html=True)
