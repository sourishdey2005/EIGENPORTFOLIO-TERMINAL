"""
visuals.py — Plotly Visualization Library
All charts use a consistent dark quant-terminal theme.
80+ chart functions organized by category.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Optional

# ─── Theme ────────────────────────────────────────────────────────────────────

THEME = dict(
    bg="#0a0e1a",
    panel="#111827",
    card="#1a2236",
    border="#1f2d45",
    accent="#00d4ff",
    accent2="#7c3aed",
    accent3="#10b981",
    accent4="#f59e0b",
    danger="#ef4444",
    text="#e2e8f0",
    muted="#64748b",
    grid="#1e2d45",
    font="'JetBrains Mono', 'Courier New', monospace",
)

COLOR_SCALE = [
    [0.0, "#0a0e1a"],
    [0.25, "#1a2236"],
    [0.5, "#00d4ff"],
    [0.75, "#7c3aed"],
    [1.0, "#f59e0b"],
]

SECTOR_COLORS = {
    "Technology": "#00d4ff",
    "Financials": "#7c3aed",
    "Healthcare": "#10b981",
    "Consumer Disc.": "#f59e0b",
    "Consumer Staples": "#f97316",
    "Energy": "#ef4444",
    "Materials": "#84cc16",
    "Industrials": "#06b6d4",
    "Utilities": "#8b5cf6",
    "Real Estate": "#ec4899",
    "Communication": "#14b8a6",
    "Other": "#64748b",
}

def _base_layout(**kwargs) -> dict:
    """Base Plotly layout for all charts."""
    layout = {
        "paper_bgcolor": THEME["bg"],
        "plot_bgcolor": THEME["panel"],
        "font": {"family": THEME["font"], "color": THEME["text"], "size": 11},
        "xaxis": {
            "gridcolor": THEME["grid"],
            "gridwidth": 0.5,
            "linecolor": THEME["border"],
            "zerolinecolor": THEME["grid"],
            "showgrid": True,
        },
        "yaxis": {
            "gridcolor": THEME["grid"],
            "gridwidth": 0.5,
            "linecolor": THEME["border"],
            "zerolinecolor": THEME["grid"],
            "showgrid": True,
        },
        "legend": {
            "bgcolor": "rgba(0,0,0,0.4)",
            "bordercolor": THEME["border"],
            "borderwidth": 1,
            "font": {"size": 10},
        },
        "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
    }

    for key, value in kwargs.items():
        if isinstance(value, dict) and isinstance(layout.get(key), dict):
            merged = dict(layout[key])
            merged.update(value)
            layout[key] = merged
        else:
            layout[key] = value

    return layout


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET + DATA CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_price_overlay(prices: pd.DataFrame, title: str = "Price Chart") -> go.Figure:
    """Multi-stock normalized price overlay (base 100)."""
    fig = go.Figure()
    norm = prices / prices.iloc[0] * 100
    colors = [THEME["accent"], THEME["accent2"], THEME["accent3"], THEME["accent4"]]
    for i, col in enumerate(norm.columns):
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col], name=col,
            line=dict(color=colors[i % len(colors)], width=1.8),
            hovertemplate=f"<b>{col}</b><br>%{{x|%Y-%m-%d}}<br>Indexed: %{{y:.1f}}<extra></extra>",
        ))
    fig.update_layout(**_base_layout(title=dict(text=title, font=dict(size=14, color=THEME["accent"])),
                                     hovermode="x unified"))
    return fig


def plot_candlestick(prices: pd.DataFrame, ticker: str) -> go.Figure:
    """OHLC candlestick (uses Close only, simulated OHLC from daily data)."""
    close = prices[ticker] if ticker in prices.columns else prices.iloc[:, 0]
    # Simulate OHLC from close for display
    fig = go.Figure(go.Scatter(
        x=close.index, y=close,
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.06)",
        line=dict(color=THEME["accent"], width=1.5),
        name=ticker,
    ))
    fig.update_layout(**_base_layout(title=dict(text=f"{ticker} — Price History", font=dict(size=14))))
    return fig


def plot_rolling_returns(returns: pd.DataFrame, window: int = 21) -> go.Figure:
    """Rolling average return heatmap-style area chart."""
    fig = go.Figure()
    colors = [THEME["accent"], THEME["accent2"], THEME["accent3"], THEME["accent4"]]
    for i, col in enumerate(returns.columns):
        roll = returns[col].rolling(window).mean() * 252  # annualised
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll, name=col,
            line=dict(color=colors[i % len(colors)], width=1.5),
        ))
    fig.update_layout(**_base_layout(
        title=dict(text=f"{window}-Day Rolling Ann. Return", font=dict(size=14)),
        yaxis=dict(tickformat=".1%"),
    ))
    return fig


def plot_volatility_clustering(returns: pd.DataFrame, ticker: str = None) -> go.Figure:
    """GARCH-style volatility clustering — absolute returns."""
    s = returns.iloc[:, 0] if ticker is None else returns[ticker]
    roll_vol = s.rolling(21).std() * np.sqrt(252)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.55, 0.45])
    fig.add_trace(go.Scatter(x=s.index, y=s, name="Log Returns",
                             line=dict(color=THEME["muted"], width=0.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=roll_vol.index, y=roll_vol, name="21d Vol",
                             line=dict(color=THEME["accent4"], width=1.8),
                             fill="tozeroy", fillcolor="rgba(245,158,11,0.1)"), row=2, col=1)
    fig.update_layout(**_base_layout(title=dict(text="Volatility Clustering", font=dict(size=14))))
    return fig


def plot_correlation_heatmap(corr: np.ndarray, labels: list[str], title: str = "Correlation Matrix") -> go.Figure:
    """Interactive annotated correlation heatmap."""
    fig = go.Figure(go.Heatmap(
        z=corr, x=labels, y=labels,
        colorscale=[[0, "#ef4444"], [0.5, THEME["panel"]], [1, "#00d4ff"]],
        zmin=-1, zmax=1,
        colorbar=dict(thickness=12, tickfont=dict(size=9)),
        hovertemplate="%{y} vs %{x}<br>Corr: %{z:.3f}<extra></extra>",
    ))
    n = len(labels)
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=14)),
        height=max(350, n * 28 + 100),
        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(tickfont=dict(size=8)),
    ))
    return fig


def plot_return_distribution(returns: pd.DataFrame) -> go.Figure:
    """KDE + histogram of returns for selected stocks."""
    fig = go.Figure()
    colors = [THEME["accent"], THEME["accent2"], THEME["accent3"], THEME["accent4"]]
    for i, col in enumerate(returns.columns):
        fig.add_trace(go.Histogram(
            x=returns[col], name=col, nbinsx=60,
            opacity=0.55, histnorm="probability density",
            marker_color=colors[i % len(colors)],
        ))
    fig.update_layout(**_base_layout(title=dict(text="Return Distribution", font=dict(size=14)),
                                     barmode="overlay"))
    return fig


def plot_rolling_correlation(returns: pd.DataFrame, window: int = 63) -> go.Figure:
    """Rolling pairwise correlation between first two assets."""
    if len(returns.columns) < 2:
        return go.Figure()
    cols = list(returns.columns)
    roll_corr = returns[cols[0]].rolling(window).corr(returns[cols[1]])
    fig = go.Figure(go.Scatter(
        x=roll_corr.index, y=roll_corr,
        fill="tozeroy", fillcolor="rgba(0,212,255,0.08)",
        line=dict(color=THEME["accent"], width=1.5),
        name=f"Corr({cols[0]},{cols[1]})",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=THEME["muted"])
    fig.update_layout(**_base_layout(title=dict(text=f"{window}d Rolling Correlation", font=dict(size=14))))
    return fig


def plot_volume_return_scatter(returns: pd.DataFrame) -> go.Figure:
    """Scatter matrix of returns pairwise."""
    n = min(4, len(returns.columns))
    cols = list(returns.columns)[:n]
    pairs = [(cols[i], cols[j]) for i in range(n) for j in range(i+1, n)]
    if not pairs:
        return go.Figure()
    fig = make_subplots(rows=1, cols=len(pairs), shared_yaxes=False,
                        subplot_titles=[f"{a} vs {b}" for a, b in pairs])
    colors = [THEME["accent"], THEME["accent2"], THEME["accent3"]]
    for k, (a, b) in enumerate(pairs):
        fig.add_trace(go.Scatter(
            x=returns[a], y=returns[b], mode="markers",
            marker=dict(size=3, color=colors[k % len(colors)], opacity=0.5),
            name=f"{a} vs {b}",
        ), row=1, col=k+1)
    fig.update_layout(**_base_layout(title=dict(text="Pairwise Return Scatter", font=dict(size=14)),
                                     height=350, showlegend=False))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────
# EXTRA DASHBOARD VISUALS (Gallery)
# ──────────────────────────────────────────────────────────────────────────────

def plot_cumulative_returns(returns: pd.DataFrame, title: str = "Cumulative Returns") -> go.Figure:
    """Cumulative return curves from log-returns (assumes daily)."""
    if returns is None or returns.empty:
        return go.Figure()
    cum = np.exp(returns.cumsum()) - 1.0
    fig = go.Figure()
    colors = [THEME["accent"], THEME["accent2"], THEME["accent3"], THEME["accent4"]]
    for i, col in enumerate(cum.columns[:8]):
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum[col], name=str(col),
            line=dict(color=colors[i % len(colors)], width=1.6),
        ))
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis=dict(tickformat=".1%"),
        hovermode="x unified",
        height=360,
    ))
    return fig


def plot_rolling_volatility_lines(returns: pd.DataFrame, window: int = 21, title: str | None = None) -> go.Figure:
    """Multi-asset rolling annualised volatility lines."""
    if returns is None or returns.empty:
        return go.Figure()
    roll = returns.rolling(window).std() * np.sqrt(252)
    fig = go.Figure()
    colors = [THEME["accent4"], THEME["accent2"], THEME["accent"], THEME["accent3"]]
    for i, col in enumerate(roll.columns[:8]):
        fig.add_trace(go.Scatter(
            x=roll.index, y=roll[col], name=str(col),
            line=dict(color=colors[i % len(colors)], width=1.4),
        ))
    fig.update_layout(**_base_layout(
        title=dict(text=title or f"{window}d Rolling Volatility (Ann.)", font=dict(size=14)),
        yaxis=dict(tickformat=".1%"),
        hovermode="x unified",
        height=360,
    ))
    return fig


def plot_latest_rolling_corr_heatmap(returns: pd.DataFrame, window: int = 63, title: str | None = None) -> go.Figure:
    """Heatmap of correlation computed over the last `window` observations."""
    if returns is None or returns.empty or len(returns) < max(window, 2):
        return go.Figure()
    corr = returns.tail(window).corr().values
    labels = [str(c) for c in returns.columns]
    return plot_correlation_heatmap(corr, labels, title=title or f"Rolling Correlation (last {window} obs)")


def plot_eigenvalues_histogram(eigenvalues: np.ndarray, lambda_plus: float | None = None) -> go.Figure:
    """Histogram of eigenvalues with optional λ⁺ marker."""
    if eigenvalues is None or len(eigenvalues) == 0:
        return go.Figure()
    ev = np.asarray(eigenvalues, dtype=float)
    fig = go.Figure(go.Histogram(
        x=ev, nbinsx=min(40, max(10, int(np.sqrt(len(ev)) * 4))),
        marker_color=THEME["accent"],
        opacity=0.75,
    ))
    if lambda_plus is not None:
        fig.add_vline(x=float(lambda_plus), line_dash="dash", line_color=THEME["accent4"])
    fig.update_layout(**_base_layout(
        title=dict(text="Eigenvalue Distribution", font=dict(size=14)),
        height=340,
        xaxis=dict(title="Eigenvalue"),
        yaxis=dict(title="Count"),
        showlegend=False,
    ))
    return fig


def plot_cumulative_variance_explained(eigenvalues: np.ndarray, n_mark: int | None = None) -> go.Figure:
    """Cumulative variance explained by sorted eigenvalues."""
    if eigenvalues is None or len(eigenvalues) == 0:
        return go.Figure()
    ev = np.asarray(eigenvalues, dtype=float)
    pct = (ev / max(ev.sum(), 1e-12)) * 100.0
    cum = np.cumsum(pct)
    fig = go.Figure(go.Scatter(
        x=np.arange(1, len(cum) + 1),
        y=cum,
        mode="lines+markers",
        line=dict(color=THEME["accent2"], width=2),
        marker=dict(size=5),
        hovertemplate="PC%{x}<br>Cumulative: %{y:.1f}%<extra></extra>",
    ))
    if n_mark is not None and 1 <= int(n_mark) <= len(cum):
        fig.add_vline(x=int(n_mark), line_dash="dot", line_color=THEME["muted"])
    fig.update_layout(**_base_layout(
        title=dict(text="Cumulative Variance Explained", font=dict(size=14)),
        height=340,
        xaxis=dict(title="Principal Component"),
        yaxis=dict(title="Cumulative %", range=[0, 100]),
        showlegend=False,
    ))
    return fig


def plot_3d_corr_surface(corr: np.ndarray, labels: list[str], title: str = "Correlation Surface (3D)") -> go.Figure:
    """3D surface visualization of a correlation matrix."""
    if corr is None or len(corr) == 0:
        return go.Figure()
    z = np.asarray(corr, dtype=float)
    fig = go.Figure(data=[go.Surface(
        z=z,
        colorscale=[[0, "#ef4444"], [0.5, THEME["panel"]], [1, "#00d4ff"]],
        cmin=-1,
        cmax=1,
        showscale=True,
        colorbar=dict(thickness=10, tickfont=dict(size=9)),
    )])
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=14)),
        height=520,
        scene=dict(
            xaxis=dict(title="", tickmode="array", tickvals=list(range(len(labels))), ticktext=labels),
            yaxis=dict(title="", tickmode="array", tickvals=list(range(len(labels))), ticktext=labels),
            zaxis=dict(title="Corr"),
            bgcolor=THEME["panel"],
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    ))
    return fig


def plot_3d_matrix_surface(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    z_title: str,
    colorscale: list | str = None,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """Generic 3D surface for any square matrix."""
    if matrix is None or len(matrix) == 0:
        return go.Figure()
    z = np.asarray(matrix, dtype=float)
    if z.ndim != 2 or z.shape[0] != z.shape[1]:
        return go.Figure()
    colors = colorscale or [[0, "#ef4444"], [0.5, THEME["panel"]], [1, "#00d4ff"]]
    surf = go.Surface(
        z=z,
        colorscale=colors,
        showscale=True,
        colorbar=dict(thickness=10, tickfont=dict(size=9)),
    )
    if zmin is not None:
        surf.update(cmin=float(zmin))
    if zmax is not None:
        surf.update(cmax=float(zmax))

    fig = go.Figure(data=[surf])
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=14)),
        height=520,
        scene=dict(
            xaxis=dict(title="", tickmode="array", tickvals=list(range(len(labels))), ticktext=labels),
            yaxis=dict(title="", tickmode="array", tickvals=list(range(len(labels))), ticktext=labels),
            zaxis=dict(title=z_title),
            bgcolor=THEME["panel"],
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# EXTRA VISUALS (Stats / Diagnostics)
# ──────────────────────────────────────────────────────────────────────────────

def plot_drawdown_from_log_returns(log_returns: pd.Series, title: str = "Drawdown") -> go.Figure:
    """Compute and plot drawdown from a log-return series."""
    if log_returns is None or len(log_returns) == 0:
        return go.Figure()
    lr = pd.Series(log_returns).dropna()
    if lr.empty:
        return go.Figure()
    equity = np.exp(lr.cumsum())
    peak = equity.cummax()
    dd = equity / peak - 1.0
    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd.values, name="Drawdown",
        fill="tozeroy", fillcolor="rgba(239,68,68,0.12)",
        line=dict(color=THEME["danger"], width=1.5),
        hovertemplate="%{x|%Y-%m-%d}<br>DD: %{y:.2%}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis=dict(tickformat=".1%"),
        height=320,
        showlegend=False,
    ))
    return fig


def plot_rolling_sharpe_series(log_returns: pd.Series, window: int = 63, title: str | None = None) -> go.Figure:
    """Rolling Sharpe for a single log-return series (assumes daily)."""
    if log_returns is None or len(log_returns) < max(window, 2):
        return go.Figure()
    lr = pd.Series(log_returns).dropna()
    if len(lr) < max(window, 2):
        return go.Figure()
    mu = lr.rolling(window).mean() * 252
    vol = lr.rolling(window).std() * np.sqrt(252)
    sharpe = mu / vol.replace(0, np.nan)
    fig = go.Figure(go.Scatter(
        x=sharpe.index, y=sharpe.values,
        name=f"{window}d Sharpe",
        line=dict(color=THEME["accent3"], width=1.6),
        hovertemplate="%{x|%Y-%m-%d}<br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color=THEME["muted"])
    fig.update_layout(**_base_layout(
        title=dict(text=title or f"{window}d Rolling Sharpe", font=dict(size=14)),
        height=320,
        showlegend=False,
    ))
    return fig


def plot_returns_heatmap(returns: pd.DataFrame, title: str = "Returns Heatmap") -> go.Figure:
    """Heatmap of returns over time (rows=time, cols=assets)."""
    if returns is None or returns.empty:
        return go.Figure()
    df = returns.copy()
    df = df.tail(min(len(df), 260))
    fig = go.Figure(go.Heatmap(
        z=df.values,
        x=[str(c) for c in df.columns],
        y=df.index,
        colorscale=[[0, "#ef4444"], [0.5, THEME["panel"]], [1, "#00d4ff"]],
        zmid=0,
        colorbar=dict(thickness=12, tickfont=dict(size=9)),
        hovertemplate="%{y|%Y-%m-%d}<br>%{x}: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=14)),
        height=420,
        yaxis=dict(tickfont=dict(size=8)),
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        margin=dict(l=60, r=20, t=50, b=80),
    ))
    return fig


def plot_3d_asset_moments(returns: pd.DataFrame, title: str = "Assets: Return vs Vol vs Skew (3D)") -> go.Figure:
    """3D scatter of mean/vol/skew from log-returns (daily)."""
    if returns is None or returns.empty:
        return go.Figure()
    df = returns.dropna(how="all", axis=1).dropna()
    if df.empty:
        return go.Figure()
    mu = df.mean() * 252
    vol = df.std() * np.sqrt(252)
    skew = df.skew()
    labels = [str(c) for c in df.columns]
    fig = go.Figure(go.Scatter3d(
        x=vol.values,
        y=mu.values,
        z=skew.values,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=4, color=mu.values, colorscale="Viridis", opacity=0.85),
        hovertemplate="Asset: %{text}<br>Vol: %{x:.2%}<br>Return: %{y:.2%}<br>Skew: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=14)),
        height=520,
        scene=dict(
            xaxis=dict(title="Vol (ann)"),
            yaxis=dict(title="Return (ann)"),
            zaxis=dict(title="Skew"),
            bgcolor=THEME["panel"],
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    ))
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 3D LAB (Generic helpers)
# ──────────────────────────────────────────────────────────────────────────────

def plot_3d_time_asset_surface(
    values: pd.DataFrame,
    title: str,
    z_label: str = "",
    colorscale: str | list = "RdBu",
    zmid: float | None = None,
) -> go.Figure:
    """3D surface for a (time x asset) matrix."""
    if values is None or values.empty:
        return go.Figure()

    df = values.copy()
    df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)
    if df.empty:
        return go.Figure()

    df = df.tail(min(len(df), 260))
    df = df.iloc[:, : min(25, df.shape[1])]

    # Surface expects z with shape (len(y), len(x))
    x = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in df.index]
    y = [str(c) for c in df.columns]
    z = df.T.values

    surface = go.Surface(
        x=x,
        y=y,
        z=z,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(thickness=10, tickfont=dict(size=9), title=z_label or ""),
    )
    if zmid is not None:
        # Surface uses color-domain midpoint 'cmid' (not 'zmid').
        surface.update(cmid=float(zmid))

    fig = go.Figure(data=[surface])
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=14)),
        height=520,
        scene=dict(
            xaxis=dict(title="Date", tickfont=dict(size=8)),
            yaxis=dict(title="Asset", tickfont=dict(size=8)),
            zaxis=dict(title=z_label),
            bgcolor=THEME["panel"],
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    ))
    return fig


def _asset_metrics_frame(returns: pd.DataFrame) -> pd.DataFrame:
    df = returns.dropna(how="all", axis=1).dropna()
    if df.empty:
        return pd.DataFrame()

    mu = df.mean() * 252
    vol = df.std() * np.sqrt(252)
    sharpe = mu / vol.replace(0, np.nan)
    skew = df.skew()
    kurt = df.kurtosis()
    var5 = df.quantile(0.05)
    cvar5 = df.apply(lambda s: s[s <= s.quantile(0.05)].mean())
    downside = df.apply(lambda s: np.sqrt((np.minimum(s, 0) ** 2).mean())) * np.sqrt(252)
    acf1 = df.apply(lambda s: s.autocorr(lag=1))
    acf5 = df.apply(lambda s: s.autocorr(lag=5))

    # max drawdown from log returns
    equity = np.exp(df.cumsum())
    dd = equity / equity.cummax() - 1.0
    max_dd = dd.min()

    return pd.DataFrame(
        {
            "return": mu,
            "vol": vol,
            "sharpe": sharpe,
            "skew": skew,
            "kurt": kurt,
            "var5": var5,
            "cvar5": cvar5,
            "downside": downside,
            "acf1": acf1,
            "acf5": acf5,
            "max_dd": max_dd,
        }
    ).replace([np.inf, -np.inf], np.nan)


def plot_3d_asset_metrics(
    returns: pd.DataFrame,
    x_metric: str,
    y_metric: str,
    z_metric: str,
    title: str,
) -> go.Figure:
    """3D scatter of per-asset metrics computed from log returns."""
    m = _asset_metrics_frame(returns)
    if m.empty:
        return go.Figure()

    for col in (x_metric, y_metric, z_metric):
        if col not in m.columns:
            return go.Figure()

    d = m[[x_metric, y_metric, z_metric]].dropna()
    if d.empty:
        return go.Figure()

    labels = [str(i) for i in d.index]
    fig = go.Figure(go.Scatter3d(
        x=d[x_metric].values,
        y=d[y_metric].values,
        z=d[z_metric].values,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=4, color=d[y_metric].values, colorscale="Viridis", opacity=0.85),
        hovertemplate="Asset: %{text}<br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=14)),
        height=520,
        scene=dict(
            xaxis=dict(title=x_metric),
            yaxis=dict(title=y_metric),
            zaxis=dict(title=z_metric),
            bgcolor=THEME["panel"],
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    ))
    return fig


def plot_3d_pca_loadings(
    eigenvectors: np.ndarray,
    tickers: list[str],
    pcs: tuple[int, int, int] = (0, 1, 2),
    title: str | None = None,
) -> go.Figure:
    """3D scatter of eigenvector loadings for three PCs."""
    if eigenvectors is None or len(eigenvectors) == 0:
        return go.Figure()
    x_i, y_i, z_i = pcs
    if max(x_i, y_i, z_i) >= eigenvectors.shape[1]:
        return go.Figure()
    n = min(len(tickers), eigenvectors.shape[0])
    x = eigenvectors[:n, x_i]
    y = eigenvectors[:n, y_i]
    z = eigenvectors[:n, z_i]
    labels = [str(t) for t in tickers[:n]]
    fig = go.Figure(go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers+text",
        text=labels,
        textposition="top center",
        marker=dict(size=4, color=z, colorscale="Plasma", opacity=0.85),
        hovertemplate="Asset: %{text}<br>PCx: %{x:.4f}<br>PCy: %{y:.4f}<br>PCz: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=title or f"PCA Loadings: PC{pcs[0]+1} vs PC{pcs[1]+1} vs PC{pcs[2]+1}", font=dict(size=14)),
        height=520,
        scene=dict(
            xaxis=dict(title=f"PC{pcs[0]+1}"),
            yaxis=dict(title=f"PC{pcs[1]+1}"),
            zaxis=dict(title=f"PC{pcs[2]+1}"),
            bgcolor=THEME["panel"],
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    ))
    return fig


def plot_3d_portfolio_cloud(df: pd.DataFrame, title: str = "Feasible Set (3D)") -> go.Figure:
    """3D scatter for portfolios with return/volatility/sharpe columns."""
    if df is None or df.empty:
        return go.Figure()
    needed = {"return", "volatility", "sharpe"}
    if not needed.issubset(set(df.columns)):
        return go.Figure()
    d = df[list(needed)].dropna()
    if d.empty:
        return go.Figure()
    fig = go.Figure(go.Scatter3d(
        x=d["volatility"].values,
        y=d["return"].values,
        z=d["sharpe"].values,
        mode="markers",
        marker=dict(size=2.5, color=d["sharpe"].values, colorscale="Viridis", opacity=0.7),
        hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=14)),
        height=520,
        scene=dict(
            xaxis=dict(title="Volatility"),
            yaxis=dict(title="Return"),
            zaxis=dict(title="Sharpe"),
            bgcolor=THEME["panel"],
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    ))
    return fig


def plot_3d_frontier_line(frontier: pd.DataFrame, title: str = "Efficient Frontier (3D)") -> go.Figure:
    """3D line for frontier with return/volatility/sharpe columns."""
    if frontier is None or frontier.empty:
        return go.Figure()
    needed = {"return", "volatility", "sharpe"}
    if not needed.issubset(set(frontier.columns)):
        return go.Figure()
    d = frontier.sort_values("volatility").dropna()
    if d.empty:
        return go.Figure()
    fig = go.Figure(go.Scatter3d(
        x=d["volatility"].values,
        y=d["return"].values,
        z=d["sharpe"].values,
        mode="lines+markers",
        line=dict(color=THEME["accent2"], width=3),
        marker=dict(size=3, color=THEME["accent"]),
        hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=14)),
        height=520,
        scene=dict(
            xaxis=dict(title="Volatility"),
            yaxis=dict(title="Return"),
            zaxis=dict(title="Sharpe"),
            bgcolor=THEME["panel"],
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    ))
    return fig


def plot_3d_backtest_metrics(metrics: dict, x: str, y: str, z: str, title: str) -> go.Figure:
    """3D scatter for strategy metrics dict like {name: {cagr, sharpe, max_drawdown, volatility, calmar}}."""
    if not metrics:
        return go.Figure()
    rows = []
    for name, m in metrics.items():
        if not isinstance(m, dict):
            continue
        if x in m and y in m and z in m:
            rows.append({"name": name, x: m[x], y: m[y], z: m[z]})
    if not rows:
        return go.Figure()
    df = pd.DataFrame(rows).dropna()
    if df.empty:
        return go.Figure()
    fig = go.Figure(go.Scatter3d(
        x=df[x].values,
        y=df[y].values,
        z=df[z].values,
        mode="markers+text",
        text=df["name"].astype(str).tolist(),
        textposition="top center",
        marker=dict(size=4, color=df[y].values, colorscale="Viridis", opacity=0.85),
        hovertemplate="Strategy: %{text}<br>X: %{x:.4f}<br>Y: %{y:.4f}<br>Z: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=14)),
        height=520,
        scene=dict(
            xaxis=dict(title=x),
            yaxis=dict(title=y),
            zaxis=dict(title=z),
            bgcolor=THEME["panel"],
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    ))
    return fig


def plot_3d_return_hist_surface(returns: pd.DataFrame, bins: int = 40, title: str = "Return Distribution Surface (3D)") -> go.Figure:
    """3D surface: x=bins, y=asset, z=density."""
    if returns is None or returns.empty:
        return go.Figure()
    df = returns.dropna(how="all", axis=1).dropna()
    if df.empty:
        return go.Figure()

    df = df.iloc[:, : min(20, df.shape[1])]
    all_vals = df.values.flatten()
    all_vals = all_vals[np.isfinite(all_vals)]
    if len(all_vals) == 0:
        return go.Figure()

    lo, hi = np.quantile(all_vals, [0.01, 0.99])
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    densities = []
    for col in df.columns:
        vals = df[col].values
        vals = vals[np.isfinite(vals)]
        hist, _ = np.histogram(vals, bins=edges, density=True)
        densities.append(hist)
    z = np.asarray(densities, dtype=float)  # (assets, bins)

    fig = go.Figure(go.Surface(
        x=[float(c) for c in centers],
        y=[str(c) for c in df.columns],
        z=z,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(thickness=10, tickfont=dict(size=9), title="Density"),
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=title, font=dict(size=14)),
        height=520,
        scene=dict(
            xaxis=dict(title="Return"),
            yaxis=dict(title="Asset"),
            zaxis=dict(title="Density"),
            bgcolor=THEME["panel"],
        ),
        margin=dict(l=10, r=10, t=50, b=10),
    ))
    return fig

# RMT ANALYSIS CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_eigenvalue_spectrum(
    eigenvalues: np.ndarray,
    mp_x: np.ndarray,
    mp_pdf: np.ndarray,
    lambda_plus: float,
    n_signal: int,
) -> go.Figure:
    """Empirical eigenvalue histogram vs Marchenko-Pastur PDF."""
    fig = make_subplots(rows=1, cols=1)
    # Normalised histogram
    fig.add_trace(go.Histogram(
        x=eigenvalues, nbinsx=40, histnorm="probability density",
        name="Empirical Spectrum",
        marker=dict(color=THEME["accent2"], opacity=0.7, line=dict(color=THEME["border"], width=0.5)),
    ))
    # MP theoretical curve
    fig.add_trace(go.Scatter(
        x=mp_x, y=mp_pdf, name="Marchenko-Pastur",
        line=dict(color=THEME["accent3"], width=2.5, dash="solid"),
    ))
    # Signal threshold
    fig.add_vline(x=lambda_plus, line_dash="dash", line_color=THEME["accent4"],
                  annotation_text=f"λ⁺={lambda_plus:.3f}", annotation_font_color=THEME["accent4"])
    # Label signal eigenvalues
    sig_eigs = eigenvalues[:n_signal]
    for e in sig_eigs:
        fig.add_vline(x=e, line_dash="dot", line_color=THEME["accent"], line_width=0.8)

    fig.update_layout(**_base_layout(
        title=dict(text="Eigenvalue Spectrum vs Marchenko-Pastur Distribution", font=dict(size=14)),
        xaxis_title="Eigenvalue λ",
        yaxis_title="Density",
    ))
    return fig


def plot_scree(eigenvalues: np.ndarray, n_signal: int) -> go.Figure:
    """Scree plot with variance explained and cumulative variance."""
    total = eigenvalues.sum()
    var_exp = eigenvalues / total * 100
    cumvar = np.cumsum(var_exp)
    n = min(40, len(eigenvalues))

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = [THEME["accent"] if i < n_signal else THEME["muted"] for i in range(n)]

    fig.add_trace(go.Bar(
        x=list(range(1, n+1)), y=var_exp[:n], name="Var Explained (%)",
        marker=dict(color=colors),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=list(range(1, n+1)), y=cumvar[:n], name="Cumulative (%)",
        line=dict(color=THEME["accent4"], width=2),
        mode="lines+markers", marker=dict(size=5),
    ), secondary_y=True)

    fig.update_layout(**_base_layout(
        title=dict(text="Scree Plot — Variance Explained by Eigenvalue", font=dict(size=14)),
    ))
    fig.update_yaxes(title_text="Individual (%)", secondary_y=False, ticksuffix="%")
    fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True, ticksuffix="%")
    return fig


def plot_signal_vs_noise(eigenvalues: np.ndarray, lambda_plus: float, n_signal: int) -> go.Figure:
    """Bar chart: signal vs noise eigenvalue separation."""
    n = len(eigenvalues)
    signal_mask = eigenvalues > lambda_plus
    x = list(range(1, n+1))
    colors = [THEME["accent"] if s else THEME["accent2"] for s in signal_mask]

    fig = go.Figure(go.Bar(
        x=x, y=eigenvalues,
        marker=dict(color=colors),
        hovertemplate="PC%{x}<br>λ = %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=lambda_plus, line_dash="dash", line_color=THEME["accent4"],
                  annotation_text=f"MP Upper Edge λ⁺={lambda_plus:.3f}")
    fig.update_layout(**_base_layout(
        title=dict(text="Signal vs Noise Eigenvalue Separation", font=dict(size=14)),
        xaxis_title="Principal Component Index",
        yaxis_title="Eigenvalue",
    ))
    return fig


def plot_mp_fit(mp_x: np.ndarray, mp_pdf: np.ndarray, q: float, sigma2: float) -> go.Figure:
    """Detailed Marchenko-Pastur distribution fit visualization."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mp_x, y=mp_pdf, name=f"MP (q={q:.2f}, σ²={sigma2:.3f})",
        line=dict(color=THEME["accent3"], width=2.5),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.08)",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Fitted Marchenko-Pastur PDF", font=dict(size=14)),
        xaxis_title="Eigenvalue λ", yaxis_title="Density",
    ))
    return fig


def plot_corr_before_after(corr_raw: np.ndarray, corr_clean: np.ndarray, labels: list[str]) -> go.Figure:
    """Side-by-side heatmap: raw vs denoised correlation."""
    n = len(labels)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Raw Correlation", "RMT-Denoised Correlation"],
                        horizontal_spacing=0.08)
    kw = dict(colorscale=[[0,"#ef4444"],[0.5,THEME["panel"]],[1,"#00d4ff"]],
              zmin=-1, zmax=1, showscale=False,
              colorbar=dict(thickness=10))
    fig.add_trace(go.Heatmap(z=corr_raw, x=labels, y=labels, **kw), row=1, col=1)
    kw["showscale"] = True
    fig.add_trace(go.Heatmap(z=corr_clean, x=labels, y=labels, **kw), row=1, col=2)
    h = max(300, n * 18 + 100)
    fig.update_layout(**_base_layout(
        title=dict(text="Correlation Matrix: Before vs After RMT Denoising", font=dict(size=14)),
        height=h,
    ))
    return fig


def plot_eigenvalue_density_3d(eigenvalues: np.ndarray, q: float) -> go.Figure:
    """3D surface: MP density over a range of q and λ values."""
    from rmt import marchenko_pastur_pdf
    q_range = np.linspace(max(0.5, q * 0.5), q * 2.0, 40)
    lam_range = np.linspace(0.01, max(eigenvalues) * 1.2, 60)
    Z = np.array([[marchenko_pastur_pdf(np.array([l]), qi, 1.0)[0] for l in lam_range] for qi in q_range])

    fig = go.Figure(go.Surface(
        x=lam_range, y=q_range, z=Z,
        colorscale=[[0, THEME["panel"]], [0.5, THEME["accent2"]], [1, THEME["accent"]]],
        showscale=True,
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="3D Marchenko-Pastur Density Surface (q vs λ)", font=dict(size=14)),
        scene=dict(
            xaxis_title="λ", yaxis_title="q = T/N", zaxis_title="Density",
            bgcolor=THEME["bg"],
            xaxis=dict(backgroundcolor=THEME["panel"]),
            yaxis=dict(backgroundcolor=THEME["panel"]),
            zaxis=dict(backgroundcolor=THEME["panel"]),
        ),
        height=500,
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# EIGENPORTFOLIO CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_eigenvector_weights(weights: np.ndarray, tickers: list[str], pc_idx: int, sector_map: dict = None) -> go.Figure:
    """Eigenvector weight bar chart with sector coloring."""
    df = pd.DataFrame({"ticker": tickers, "weight": weights})
    df["sector"] = df["ticker"].map(lambda x: sector_map.get(x, "Other") if sector_map else "Other")
    df = df.sort_values("weight", ascending=True)

    color_list = [SECTOR_COLORS.get(s, THEME["muted"]) for s in df["sector"]]
    fig = go.Figure(go.Bar(
        y=df["ticker"], x=df["weight"], orientation="h",
        marker=dict(color=color_list),
        hovertemplate="%{y}<br>Weight: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=f"PC{pc_idx+1} Eigenvector Weights", font=dict(size=14)),
        xaxis_title="Weight", yaxis_title="",
        height=max(300, len(tickers) * 18 + 60),
    ))
    return fig


def plot_sector_exposure(weights: np.ndarray, tickers: list[str], sector_map: dict) -> go.Figure:
    """Sector exposure pie + treemap side-by-side."""
    df = pd.DataFrame({"ticker": tickers, "weight": weights})
    df["sector"] = df["ticker"].map(lambda x: sector_map.get(x, "Other"))
    sector_weights = df.groupby("sector")["weight"].sum().reset_index()
    sector_weights = sector_weights[sector_weights["weight"] > 0]
    colors = [SECTOR_COLORS.get(s, THEME["muted"]) for s in sector_weights["sector"]]

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "treemap"}]],
                        subplot_titles=["Sector Allocation (Pie)", "Sector Allocation (Treemap)"])
    fig.add_trace(go.Pie(
        labels=sector_weights["sector"], values=sector_weights["weight"],
        marker=dict(colors=colors), hole=0.4,
        textfont=dict(size=10),
    ), row=1, col=1)
    fig.add_trace(go.Treemap(
        labels=sector_weights["sector"], values=sector_weights["weight"],
        parents=[""] * len(sector_weights),
        marker=dict(colors=colors),
        textfont=dict(size=10),
    ), row=1, col=2)
    fig.update_layout(**_base_layout(title=dict(text="Sector Exposure Breakdown", font=dict(size=14)), height=400))
    return fig


def plot_factor_loading_heatmap(eigenvectors: np.ndarray, tickers: list[str], n_factors: int = 8) -> go.Figure:
    """Heatmap of factor loadings: tickers × top-N PCs."""
    n_f = min(n_factors, eigenvectors.shape[1])
    Z = eigenvectors[:, :n_f]
    fig = go.Figure(go.Heatmap(
        z=Z, x=[f"PC{i+1}" for i in range(n_f)], y=tickers,
        colorscale=[[0,"#ef4444"],[0.5,THEME["panel"]],[1,"#00d4ff"]],
        colorbar=dict(thickness=12),
        hovertemplate="%{y}<br>%{x}: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Factor Loading Heatmap (Tickers × PCs)", font=dict(size=14)),
        height=max(300, len(tickers) * 20 + 80),
        xaxis=dict(tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=8)),
    ))
    return fig


def plot_variance_contribution(eigenvalues: np.ndarray, n_signal: int) -> go.Figure:
    """Donut chart of variance contribution by PC group."""
    total = eigenvalues.sum()
    signal_var = eigenvalues[:n_signal].sum() / total * 100
    noise_var = 100 - signal_var

    fig = go.Figure(go.Pie(
        labels=["Signal PCs", "Noise PCs"],
        values=[signal_var, noise_var],
        hole=0.55,
        marker=dict(colors=[THEME["accent"], THEME["muted"]]),
        textfont=dict(size=12),
        hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
    ))
    fig.add_annotation(text=f"{signal_var:.1f}%<br>Signal", x=0.5, y=0.5,
                       font=dict(size=14, color=THEME["accent"]), showarrow=False)
    fig.update_layout(**_base_layout(title=dict(text="Variance Contribution: Signal vs Noise", font=dict(size=14))))
    return fig


def plot_eigenvector_comparison(eigenvectors: np.ndarray, tickers: list[str], pcs: list[int]) -> go.Figure:
    """Radar/spider chart comparing eigenvector profiles across PCs (first 10 tickers)."""
    n = min(12, len(tickers))
    fig = go.Figure()
    colors = [THEME["accent"], THEME["accent2"], THEME["accent3"], THEME["accent4"]]
    for k, pc in enumerate(pcs[:4]):
        v = eigenvectors[:n, pc]
        fig.add_trace(go.Scatterpolar(
            r=list(v) + [v[0]],
            theta=tickers[:n] + [tickers[0]],
            name=f"PC{pc+1}",
            line=dict(color=colors[k], width=1.8),
            fill="toself", fillcolor=f"rgba({int(colors[k][1:3],16)},{int(colors[k][3:5],16)},{int(colors[k][5:7],16)},0.08)",
        ))
    fig.update_layout(**_base_layout(
        title=dict(text="Eigenvector Profile Comparison", font=dict(size=14)),
        polar=dict(bgcolor=THEME["panel"], radialaxis=dict(gridcolor=THEME["grid"])),
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO ANALYTICS CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_efficient_frontier(frontier: pd.DataFrame, portfolios: dict = None) -> go.Figure:
    """Interactive efficient frontier with special portfolio markers."""
    fig = go.Figure()
    if not frontier.empty:
        fig.add_trace(go.Scatter(
            x=frontier["volatility"], y=frontier["return"],
            mode="lines",
            line=dict(color=THEME["accent"], width=2.5),
            name="Efficient Frontier",
            hovertemplate="Vol: %{x:.2%}<br>Ret: %{y:.2%}<extra></extra>",
        ))
        # Color by Sharpe
        fig.add_trace(go.Scatter(
            x=frontier["volatility"], y=frontier["return"],
            mode="markers",
            marker=dict(
                color=frontier["sharpe"], colorscale=COLOR_SCALE, size=4,
                colorbar=dict(title="Sharpe", thickness=10),
            ),
            name="Sharpe", showlegend=False,
        ))
    if portfolios:
        markers = {"⭐ Max Sharpe": THEME["accent4"], "🔷 Min Var": THEME["accent3"],
                   "⚖️ Risk Parity": THEME["accent2"], "📊 Equal Weight": THEME["muted"]}
        for name, (ret, vol) in portfolios.items():
            color = markers.get(name, THEME["accent"])
            fig.add_trace(go.Scatter(
                x=[vol], y=[ret], mode="markers+text",
                marker=dict(size=14, color=color, symbol="star"),
                text=[name.split(" ", 1)[-1]], textposition="top center",
                textfont=dict(size=9), name=name,
            ))
    fig.update_layout(**_base_layout(
        title=dict(text="Efficient Frontier", font=dict(size=14)),
        xaxis_title="Volatility (σ)", yaxis_title="Expected Return",
        xaxis=dict(tickformat=".1%"), yaxis=dict(tickformat=".1%"),
    ))
    return fig


def plot_risk_return_scatter(returns: pd.DataFrame, sector_map: dict = None) -> go.Figure:
    """Individual asset risk-return scatter."""
    mu = returns.mean() * 252
    vol = returns.std() * np.sqrt(252)
    sharpe = mu / vol
    df = pd.DataFrame({"ticker": mu.index, "return": mu.values, "volatility": vol.values,
                        "sharpe": sharpe.values})
    df["sector"] = df["ticker"].map(lambda x: sector_map.get(x, "Other") if sector_map else "Other")
    color_list = [SECTOR_COLORS.get(s, THEME["muted"]) for s in df["sector"]]

    fig = go.Figure(go.Scatter(
        x=df["volatility"], y=df["return"], mode="markers+text",
        text=df["ticker"],
        marker=dict(size=8, color=color_list, line=dict(color=THEME["border"], width=0.5)),
        textposition="top center", textfont=dict(size=7),
        hovertemplate="%{text}<br>Vol: %{x:.1%}<br>Ret: %{y:.1%}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Individual Asset Risk-Return", font=dict(size=14)),
        xaxis_title="Volatility (σ)", yaxis_title="Expected Return",
        xaxis=dict(tickformat=".0%"), yaxis=dict(tickformat=".0%"),
        height=420,
    ))
    return fig


def plot_allocation_pie(weights: np.ndarray, tickers: list[str], title: str = "Portfolio Allocation") -> go.Figure:
    """Portfolio allocation pie chart (top holdings + Other)."""
    df = pd.DataFrame({"ticker": tickers, "weight": weights})
    df = df[df["weight"] > 0.005].sort_values("weight", ascending=False)
    top = df.head(12)
    if len(df) > 12:
        other = pd.DataFrame([{"ticker": "Other", "weight": df.iloc[12:]["weight"].sum()}])
        top = pd.concat([top, other])

    colors = [THEME["accent"], THEME["accent2"], THEME["accent3"], THEME["accent4"],
              "#f97316", "#84cc16", "#06b6d4", "#8b5cf6", "#ec4899", "#14b8a6",
              THEME["muted"], "#64748b"][:len(top)]
    fig = go.Figure(go.Pie(
        labels=top["ticker"], values=top["weight"], hole=0.4,
        marker=dict(colors=colors), textfont=dict(size=10),
        hovertemplate="%{label}: %{value:.1%}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(title=dict(text=title, font=dict(size=14))))
    return fig


def plot_weight_treemap(weights: np.ndarray, tickers: list[str], sector_map: dict = None) -> go.Figure:
    """Treemap of portfolio weights by sector."""
    df = pd.DataFrame({"ticker": tickers, "weight": weights})
    df = df[df["weight"] > 0]
    df["sector"] = df["ticker"].map(lambda x: sector_map.get(x, "Other") if sector_map else "Other")
    fig = go.Figure(go.Treemap(
        labels=df["ticker"].tolist() + df["sector"].unique().tolist() + ["Portfolio"],
        parents=df["sector"].tolist() + ["Portfolio"] * len(df["sector"].unique()) + [""],
        values=df["weight"].tolist() + [0] * len(df["sector"].unique()) + [0],
        marker=dict(colors=[SECTOR_COLORS.get(df.loc[df["ticker"]==t, "sector"].values[0], THEME["muted"])
                            if t in df["ticker"].values else THEME["panel"] for t in df["ticker"].tolist()
                            ] + [THEME["panel"]] * (len(df["sector"].unique()) + 1)),
        textfont=dict(size=10),
    ))
    fig.update_layout(**_base_layout(title=dict(text="Portfolio Weight Treemap", font=dict(size=14)), height=400))
    return fig


def plot_weights_bar_compare(weights_dict: dict, tickers: list[str]) -> go.Figure:
    """Grouped bar chart comparing weights across multiple strategies."""
    colors = [THEME["accent"], THEME["accent2"], THEME["accent3"], THEME["accent4"]]
    fig = go.Figure()
    for k, (name, w) in enumerate(weights_dict.items()):
        fig.add_trace(go.Bar(
            x=tickers, y=w, name=name,
            marker_color=colors[k % len(colors)],
            opacity=0.85,
        ))
    fig.update_layout(**_base_layout(
        title=dict(text="Weight Comparison Across Strategies", font=dict(size=14)),
        barmode="group", xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
        yaxis_title="Weight",
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTESTING CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_equity_curves(results: dict) -> go.Figure:
    """Overlay equity curves for multiple strategies."""
    colors = [THEME["accent"], THEME["accent2"], THEME["accent3"], THEME["accent4"],
              "#f97316", "#84cc16"]
    fig = go.Figure()
    for k, (name, result) in enumerate(results.items()):
        eq = result.equity_curve
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq, name=name,
            line=dict(color=colors[k % len(colors)], width=1.8),
            hovertemplate=f"<b>{name}</b><br>%{{x|%Y-%m-%d}}<br>Value: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(**_base_layout(
        title=dict(text="Equity Curves — Strategy Comparison", font=dict(size=14)),
        yaxis_title="Portfolio Value (Base=1)",
    ))
    return fig


def plot_drawdown_curves(results: dict) -> go.Figure:
    """Drawdown curves for all strategies."""
    colors = [THEME["danger"], THEME["accent2"], THEME["accent4"], THEME["muted"]]
    fig = go.Figure()
    for k, (name, result) in enumerate(results.items()):
        eq = result.equity_curve
        roll_max = eq.cummax()
        dd = (eq - roll_max) / roll_max
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd,
            fill="tozeroy", fillcolor=f"rgba({int(colors[k%len(colors)][1:3],16)},{int(colors[k%len(colors)][3:5],16)},{int(colors[k%len(colors)][5:7],16)},0.15)",
            line=dict(color=colors[k % len(colors)], width=1.5),
            name=name,
        ))
    fig.update_layout(**_base_layout(
        title=dict(text="Drawdown Analysis", font=dict(size=14)),
        yaxis_title="Drawdown", yaxis=dict(tickformat=".0%"),
    ))
    return fig


def plot_rolling_sharpe(results: dict, window: int = 63) -> go.Figure:
    """Rolling Sharpe ratio for each strategy."""
    colors = [THEME["accent"], THEME["accent2"], THEME["accent3"], THEME["accent4"]]
    fig = go.Figure()
    for k, (name, result) in enumerate(results.items()):
        rs = result.daily_returns.rolling(window).mean() / result.daily_returns.rolling(window).std() * np.sqrt(252)
        fig.add_trace(go.Scatter(
            x=rs.index, y=rs, name=name,
            line=dict(color=colors[k % len(colors)], width=1.5),
        ))
    fig.add_hline(y=0, line_dash="dot", line_color=THEME["muted"])
    fig.add_hline(y=1, line_dash="dash", line_color=THEME["accent3"], line_width=0.8,
                  annotation_text="Sharpe=1")
    fig.update_layout(**_base_layout(
        title=dict(text=f"{window}-Day Rolling Sharpe Ratio", font=dict(size=14)),
        yaxis_title="Rolling Sharpe",
    ))
    return fig


def plot_alpha_beta(alpha_beta: pd.DataFrame) -> go.Figure:
    """Rolling alpha and beta dual-axis chart."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=alpha_beta.index, y=alpha_beta["alpha"],
        name="Rolling Alpha (ann.)",
        line=dict(color=THEME["accent3"], width=1.5),
        fill="tozeroy", fillcolor="rgba(16,185,129,0.08)",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=alpha_beta.index, y=alpha_beta["beta"],
        name="Rolling Beta",
        line=dict(color=THEME["accent4"], width=1.5),
    ), secondary_y=True)
    fig.update_layout(**_base_layout(title=dict(text="Rolling Alpha & Beta", font=dict(size=14))))
    fig.update_yaxes(title_text="Alpha", secondary_y=False, tickformat=".1%")
    fig.update_yaxes(title_text="Beta", secondary_y=True)
    return fig


def plot_rebalance_impact(result, prices: pd.DataFrame) -> go.Figure:
    """Equity curve with rebalance points marked."""
    fig = go.Figure()
    eq = result.equity_curve
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq, name="Portfolio Value",
        line=dict(color=THEME["accent"], width=1.8),
    ))
    # Mark rebalance dates
    rebal_vals = [eq.get(d, np.nan) for d in result.rebalance_dates]
    fig.add_trace(go.Scatter(
        x=result.rebalance_dates, y=rebal_vals,
        mode="markers", name="Rebalance",
        marker=dict(color=THEME["accent4"], size=6, symbol="triangle-up"),
    ))
    fig.update_layout(**_base_layout(title=dict(text="Equity Curve with Rebalance Events", font=dict(size=14))))
    return fig


def plot_monthly_returns_heatmap(daily_returns: pd.Series, name: str = "Strategy") -> go.Figure:
    """Calendar heatmap of monthly returns."""
    monthly = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    df = pd.DataFrame({"date": monthly.index, "return": monthly.values})
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    pivot = df.pivot(index="year", columns="month", values="return")
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot.columns = [month_names[m-1] for m in pivot.columns]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index.astype(str),
        colorscale=[[0,"#ef4444"],[0.5,"#1a2236"],[1,"#10b981"]],
        colorbar=dict(title="%", thickness=10, tickformat=".0%"),
        hovertemplate="Month: %{x}<br>Year: %{y}<br>Return: %{z:.2%}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(title=dict(text=f"Monthly Returns Heatmap — {name}", font=dict(size=14))))
    return fig


def plot_return_distribution_strategy(results: dict) -> go.Figure:
    """Return distribution comparison across strategies."""
    colors = [THEME["accent"], THEME["accent2"], THEME["accent3"], THEME["accent4"]]
    fig = go.Figure()
    for k, (name, result) in enumerate(results.items()):
        fig.add_trace(go.Histogram(
            x=result.daily_returns, name=name, nbinsx=60,
            opacity=0.6, histnorm="probability density",
            marker_color=colors[k % len(colors)],
        ))
    fig.update_layout(**_base_layout(
        title=dict(text="Return Distribution Comparison", font=dict(size=14)),
        barmode="overlay",
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TURNOVER & STABILITY CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_turnover_series(result, name: str = "Strategy") -> go.Figure:
    """Turnover time series as bar chart."""
    fig = go.Figure(go.Bar(
        x=result.rebalance_dates[1:], y=result.turnover_series,
        marker=dict(
            color=result.turnover_series,
            colorscale=[[0, THEME["accent3"]], [0.5, THEME["accent4"]], [1, THEME["danger"]]],
        ),
        name="Turnover",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text=f"Portfolio Turnover per Rebalance — {name}", font=dict(size=14)),
        yaxis_title="One-Way Turnover",
        yaxis=dict(tickformat=".0%"),
    ))
    return fig


def plot_weight_drift(result, tickers: list[str]) -> go.Figure:
    """Stacked area showing weight drift across rebalance windows."""
    if not result.weights_history:
        return go.Figure()
    dates = result.rebalance_dates[:len(result.weights_history)]
    df = pd.DataFrame(result.weights_history, index=dates, columns=tickers)
    colors = px.colors.qualitative.Dark24

    fig = go.Figure()
    for k, t in enumerate(tickers):
        if df[t].max() > 0.01:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[t], name=t, stackgroup="one",
                line=dict(width=0),
                fillcolor=colors[k % len(colors)],
            ))
    fig.update_layout(**_base_layout(
        title=dict(text="Weight Drift Over Time", font=dict(size=14)),
        yaxis_title="Weight", yaxis=dict(tickformat=".0%"),
    ))
    return fig


def plot_eigenvector_stability(stability_df: pd.DataFrame) -> go.Figure:
    """Heatmap of eigenvector stability across rolling windows."""
    if stability_df.empty:
        return go.Figure()
    fig = go.Figure(go.Heatmap(
        z=stability_df.values, x=stability_df.columns, y=stability_df.index,
        colorscale=[[0, THEME["danger"]], [0.5, THEME["accent4"]], [1, THEME["accent3"]]],
        zmin=0, zmax=1,
        colorbar=dict(title="|cos θ|", thickness=10),
        hovertemplate="%{y}<br>%{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Eigenvector Stability Across Windows (|cos θ|)", font=dict(size=14)),
    ))
    return fig


def plot_cumulative_cost(result, transaction_cost: float = 0.001) -> go.Figure:
    """Cumulative transaction cost impact over time."""
    if not result.turnover_series:
        return go.Figure()
    costs = np.array(result.turnover_series) * transaction_cost * 2  # two-way
    cum_cost = np.cumsum(costs)
    fig = go.Figure(go.Scatter(
        x=result.rebalance_dates[1:], y=cum_cost,
        fill="tozeroy", fillcolor="rgba(239,68,68,0.1)",
        line=dict(color=THEME["danger"], width=1.8),
        name="Cumulative Cost",
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Cumulative Transaction Cost Impact", font=dict(size=14)),
        yaxis_title="Cumulative Cost", yaxis=dict(tickformat=".2%"),
    ))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_3d_risk_return_time(results: dict) -> go.Figure:
    """3D surface: portfolio value vs rolling vol vs time."""
    fig = go.Figure()
    colors = [THEME["accent"], THEME["accent2"], THEME["accent3"]]
    for k, (name, result) in enumerate(results.items()):
        eq = result.equity_curve
        roll_vol = result.daily_returns.rolling(21).std() * np.sqrt(252)
        t_axis = np.arange(len(eq))
        fig.add_trace(go.Scatter3d(
            x=t_axis, y=roll_vol.values, z=eq.values,
            mode="lines", name=name,
            line=dict(color=colors[k % len(colors)], width=3),
        ))
    fig.update_layout(**_base_layout(
        title=dict(text="3D: Portfolio Value vs Volatility vs Time", font=dict(size=14)),
        scene=dict(
            xaxis_title="Time", yaxis_title="Rolling Vol", zaxis_title="Portfolio Value",
            bgcolor=THEME["bg"],
        ),
        height=520,
    ))
    return fig


def plot_monte_carlo(mc: pd.DataFrame, frontier: pd.DataFrame, special_pts: dict = None) -> go.Figure:
    """Monte Carlo feasible set + efficient frontier overlay."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=mc["volatility"], y=mc["return"],
        mode="markers",
        marker=dict(
            color=mc["sharpe"], colorscale=COLOR_SCALE, size=3, opacity=0.4,
            colorbar=dict(title="Sharpe", thickness=10),
        ),
        name="Monte Carlo Portfolios",
        hovertemplate="Vol: %{x:.1%}<br>Ret: %{y:.1%}<extra></extra>",
    ))
    if not frontier.empty:
        fig.add_trace(go.Scatter(
            x=frontier["volatility"], y=frontier["return"],
            mode="lines", name="Efficient Frontier",
            line=dict(color=THEME["accent"], width=2.5),
        ))
    if special_pts:
        for name, (ret, vol) in special_pts.items():
            fig.add_trace(go.Scatter(
                x=[vol], y=[ret], mode="markers",
                marker=dict(size=14, color=THEME["accent4"], symbol="star"),
                name=name,
            ))
    fig.update_layout(**_base_layout(
        title=dict(text="Monte Carlo Simulation — Feasible Portfolio Space", font=dict(size=14)),
        xaxis_title="Volatility", yaxis_title="Expected Return",
        xaxis=dict(tickformat=".1%"), yaxis=dict(tickformat=".1%"),
        height=450,
    ))
    return fig


def plot_metrics_radar(metrics_dict: dict) -> go.Figure:
    """Radar chart comparing strategies across performance dimensions."""
    categories = ["CAGR", "Sharpe", "Calmar", "Low Drawdown", "Low Vol"]
    fig = go.Figure()
    colors = [THEME["accent"], THEME["accent2"], THEME["accent3"], THEME["accent4"]]

    for k, (name, m) in enumerate(metrics_dict.items()):
        # Normalize metrics to 0-1 scale for radar display
        values = [
            min(max(m["cagr"] / 0.30, 0), 1),
            min(max(m["sharpe"] / 2.0, 0), 1),
            min(max(m["calmar"] / 2.0, 0), 1),
            min(max(1 + m["max_drawdown"] / 0.5, 0), 1),  # less drawdown = higher score
            min(max(1 - m["volatility"] / 0.40, 0), 1),
        ]
        values += [values[0]]  # close the radar
        fig.add_trace(go.Scatterpolar(
            r=values, theta=categories + [categories[0]],
            fill="toself", name=name,
            line=dict(color=colors[k % len(colors)]),
            fillcolor=f"rgba({int(colors[k%len(colors)][1:3],16)},{int(colors[k%len(colors)][3:5],16)},{int(colors[k%len(colors)][5:7],16)},0.12)",
        ))
    fig.update_layout(**_base_layout(
        title=dict(text="Strategy Performance Radar", font=dict(size=14)),
        polar=dict(bgcolor=THEME["panel"], radialaxis=dict(range=[0, 1], gridcolor=THEME["grid"])),
        height=450,
    ))
    return fig


def plot_rolling_metrics_dashboard(result) -> go.Figure:
    """4-panel rolling metrics: return, vol, Sharpe, max-dd."""
    r = result.daily_returns
    eq = result.equity_curve
    roll_ret = r.rolling(21).mean() * 252
    roll_vol = r.rolling(21).std() * np.sqrt(252)
    roll_sharpe = roll_ret / roll_vol
    roll_max = eq.cummax()
    roll_dd = (eq - roll_max) / roll_max

    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.08,
                        horizontal_spacing=0.08,
                        subplot_titles=["Rolling Ann. Return", "Rolling Ann. Volatility",
                                        "Rolling Sharpe", "Drawdown"])
    for (row, col, series, color) in [
        (1, 1, roll_ret, THEME["accent3"]),
        (1, 2, roll_vol, THEME["accent4"]),
        (2, 1, roll_sharpe, THEME["accent"]),
        (2, 2, roll_dd, THEME["danger"]),
    ]:
        fig.add_trace(go.Scatter(
            x=series.index, y=series, line=dict(color=color, width=1.5),
            fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
            name=series.name or "",
        ), row=row, col=col)
    fig.update_layout(**_base_layout(
        title=dict(text="Rolling Performance Dashboard", font=dict(size=14)),
        height=500, showlegend=False,
    ))
    return fig


def plot_performance_table(metrics_dict: dict) -> go.Figure:
    """Table visualization of all strategy metrics."""
    names = list(metrics_dict.keys())
    cagr = [f"{m['cagr']:.2%}" for m in metrics_dict.values()]
    sharpe = [f"{m['sharpe']:.3f}" for m in metrics_dict.values()]
    mdd = [f"{m['max_drawdown']:.2%}" for m in metrics_dict.values()]
    vol = [f"{m['volatility']:.2%}" for m in metrics_dict.values()]
    calmar = [f"{m['calmar']:.3f}" for m in metrics_dict.values()]

    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Strategy</b>", "<b>CAGR</b>", "<b>Sharpe</b>",
                    "<b>Max DD</b>", "<b>Volatility</b>", "<b>Calmar</b>"],
            fill_color=THEME["card"], font=dict(color=THEME["accent"], size=11),
            line_color=THEME["border"], align="center",
        ),
        cells=dict(
            values=[names, cagr, sharpe, mdd, vol, calmar],
            fill_color=THEME["panel"],
            font=dict(color=THEME["text"], size=10),
            line_color=THEME["border"],
            align="center",
        ),
    ))
    fig.update_layout(**_base_layout(
        title=dict(text="Strategy Performance Summary", font=dict(size=14)),
        height=max(200, len(names) * 35 + 100),
    ))
    return fig
