"""
data.py — Data Layer
Fetches, cleans, and prepares stock data for RMT and portfolio analysis.
Supports S&P 500 and NIFTY 50 universes with configurable timeframes.
Uses yahooquery for data fetching.
"""

import os
from pathlib import Path
from yahooquery import Ticker
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
import time
import requests
from datetime import datetime
import io

# ─── Universe Definitions ───────────────────────────────────────────────────

SP500_STOCKS = {
    "Apple Inc.": "AAPL",
    "Microsoft Corp.": "MSFT",
    "Amazon.com Inc.": "AMZN",
    "NVIDIA Corp.": "NVDA",
    "Alphabet Inc. (GOOGL)": "GOOGL",
    "Meta Platforms": "META",
    "Tesla Inc.": "TSLA",
    "Berkshire Hathaway": "BRK-B",
    "UnitedHealth Group": "UNH",
    "Johnson & Johnson": "JNJ",
    "Exxon Mobil": "XOM",
    "JPMorgan Chase": "JPM",
    "Visa Inc.": "V",
    "Procter & Gamble": "PG",
    "Mastercard": "MA",
    "Home Depot": "HD",
    "Chevron Corp.": "CVX",
    "Merck & Co.": "MRK",
    "AbbVie Inc.": "ABBV",
    "Eli Lilly": "LLY",
    "PepsiCo": "PEP",
    "Coca-Cola": "KO",
    "Costco Wholesale": "COST",
    "McDonald's Corp.": "MCD",
    "Salesforce": "CRM",
    "Adobe Inc.": "ADBE",
    "Netflix": "NFLX",
    "Intel Corp.": "INTC",
    "Cisco Systems": "CSCO",
    "Walt Disney": "DIS",
    "Verizon": "VZ",
    "AT&T": "T",
    "Boeing": "BA",
    "Caterpillar": "CAT",
    "3M Company": "MMM",
    "Goldman Sachs": "GS",
    "Morgan Stanley": "MS",
    "American Express": "AXP",
    "BlackRock": "BLK",
    "Citigroup": "C",
    "Bank of America": "BAC",
    "Wells Fargo": "WFC",
    "Pfizer": "PFE",
    "Bristol-Myers Squibb": "BMY",
    "Amgen": "AMGN",
    "Gilead Sciences": "GILD",
    "Anthem": "ANTM",
    "CVS Health": "CVS",
    "Medtronic": "MDT",
    "Stryker": "SYK",
    "Intuitive Surgical": "ISRG",
    "Lowe's Companies": "LOW",
    "Nike": "NKE",
    "Starbucks": "SBUX",
    "Target": "TGT",
    "Walmart": "WMT",
    "Dollar General": "DG",
    "General Electric": "GE",
    "Honeywell": "HON",
    "Raytheon Tech.": "RTX",
    "Lockheed Martin": "LMT",
    "Northrop Grumman": "NOC",
    "Deere & Co.": "DE",
    "Emerson Electric": "EMR",
    "Illinois Tool Works": "ITW",
    "Parker Hannifin": "PH",
    "Eaton Corp.": "ETN",
    "Cummins": "CMI",
    "Prologis": "PLD",
    "American Tower": "AMT",
    "Crown Castle": "CCI",
    "Equinix": "EQIX",
    "Simon Property Group": "SPG",
    "Duke Energy": "DUK",
    "NextEra Energy": "NEE",
    "Southern Company": "SO",
    "Dominion Energy": "D",
    "Exelon": "EXC",
    "Linde plc": "LIN",
    "Air Products": "APD",
    "Sherwin-Williams": "SHW",
    "PPG Industries": "PPG",
    "Freeport-McMoRan": "FCX",
    "Nucor": "NUE",
    "Newmont": "NEM",
    "Mosaic": "MOS",
    "Archer-Daniels-Midland": "ADM",
    "Celanese": "CE",
    "Lyondell Basell": "LYB",
    "Eastman Chemical": "EMN",
    "Albemarle": "ALB",
    "Applied Materials": "AMAT",
    "KLA Corp.": "KLAC",
    "Lam Research": "LRCX",
    "Micron Technology": "MU",
    "Qualcomm": "QCOM",
    "Texas Instruments": "TXN",
    "Analog Devices": "ADI",
    "Broadcom": "AVGO",
    "Marvell Technology": "MRVL",
    "ON Semiconductor": "ON",
    "Monolithic Power": "MPWR",
    "Skyworks Solutions": "SWKS",
    "Keysight Tech.": "KEYS",
    "Cognizant": "CTSH",
    "Accenture": "ACN",
    "IBM": "IBM",
    "Oracle": "ORCL",
    "ServiceNow": "NOW",
    "Workday": "WDAY",
    "Snowflake": "SNOW",
    "Palantir": "PLTR",
    "Datadog": "DDOG",
    "Cloudflare": "NET",
    "CrowdStrike": "CRWD",
    "Fortinet": "FTNT",
    "Palo Alto Networks": "PANW",
    "Zscaler": "ZS",
    "HubSpot": "HUBS",
    "Twilio": "TWLO",
    "Zendesk": "ZEN",
    "Splunk": "SPLK",
    "Pure Storage": "PSTG",
    "Nutanix": "NTNX",
    "F5 Networks": "FFIV",
    "Akamai Tech.": "AKAM",
    "VeriSign": "VRSN",
    "Gartner": "IT",
}

NIFTY50_STOCKS = {
    "Reliance Industries": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Infosys": "INFY.NS",
    "HUL": "HINDUNILVR.NS",
    "ITC Ltd": "ITC.NS",
    "SBI": "SBIN.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "L&T": "LT.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Titan Company": "TITAN.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Wipro": "WIPRO.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "NTPC": "NTPC.NS",
    "Power Grid": "POWERGRID.NS",
    "ONGC": "ONGC.NS",
    "Bajaj Auto": "BAJAJ-AUTO.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Tech Mahindra": "TECHM.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Tata Steel": "TATASTEEL.NS",
    "JSW Steel": "JSWSTEEL.NS",
    "Hindalco": "HINDALCO.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Bajaj Finserv": "BAJAJFINSV.NS",
    "Grasim": "GRASIM.NS",
    "Cipla": "CIPLA.NS",
    "Dr. Reddy's": "DRREDDY.NS",
    "Divis Lab": "DIVISLAB.NS",
    "Eicher Motors": "EICHERMOT.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS",
    "Tata Consumer": "TATACONSUM.NS",
    "Nestlé India": "NESTLEIND.NS",
    "Britannia": "BRITANNIA.NS",
    "Dmart": "DMART.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Coal India": "COALINDIA.NS",
    "IOC": "IOC.NS",
    "BPCL": "BPCL.NS",
    "Apollo Hospitals": "APOLLOHOSP.NS",
    "Shree Cement": "SHREECEM.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "SBI Life": "SBILIFE.NS",
}

SECTOR_MAP_SP500 = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology", "ADBE": "Technology",
    "CRM": "Technology", "ORCL": "Technology", "IBM": "Technology",
    "INTC": "Technology", "CSCO": "Technology", "QCOM": "Technology",
    "TXN": "Technology", "AVGO": "Technology", "AMD": "Technology",
    "AMAT": "Technology", "KLAC": "Technology", "LRCX": "Technology",
    "MU": "Technology", "ADI": "Technology", "MRVL": "Technology",
    "SNOW": "Technology", "NOW": "Technology", "WDAY": "Technology",
    "PLTR": "Technology", "DDOG": "Technology", "NET": "Technology",
    "CRWD": "Technology", "FTNT": "Technology", "PANW": "Technology",
    "ZS": "Technology", "HUBS": "Technology", "SPLK": "Technology",
    "AKAM": "Technology", "ACN": "Technology", "CTSH": "Technology",
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "AXP": "Financials",
    "V": "Financials", "MA": "Financials", "BLK": "Financials",
    "C": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "MRK": "Healthcare",
    "ABBV": "Healthcare", "LLY": "Healthcare", "PFE": "Healthcare",
    "BMY": "Healthcare", "AMGN": "Healthcare", "GILD": "Healthcare",
    "ANTM": "Healthcare", "CVS": "Healthcare", "MDT": "Healthcare",
    "SYK": "Healthcare", "ISRG": "Healthcare",
    "AMZN": "Consumer Disc.", "TSLA": "Consumer Disc.", "HD": "Consumer Disc.",
    "MCD": "Consumer Disc.", "NKE": "Consumer Disc.", "SBUX": "Consumer Disc.",
    "LOW": "Consumer Disc.", "TGT": "Consumer Disc.", "DIS": "Consumer Disc.",
    "NFLX": "Consumer Disc.",
    "PG": "Consumer Staples", "PEP": "Consumer Staples", "KO": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples", "DG": "Consumer Staples",
    "ADM": "Consumer Staples",
    "XOM": "Energy", "CVX": "Energy", "FCX": "Materials",
    "NUE": "Materials", "NEM": "Materials", "MOS": "Materials",
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "PPG": "Materials", "LYB": "Materials", "ALB": "Materials",
    "BA": "Industrials", "CAT": "Industrials", "MMM": "Industrials",
    "GE": "Industrials", "HON": "Industrials", "RTX": "Industrials",
    "LMT": "Industrials", "NOC": "Industrials", "DE": "Industrials",
    "EMR": "Industrials", "ITW": "Industrials", "PH": "Industrials",
    "ETN": "Industrials", "CMI": "Industrials",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "EXC": "Utilities",
    "PLD": "Real Estate", "AMT": "Real Estate", "CCI": "Real Estate",
    "EQIX": "Real Estate", "SPG": "Real Estate",
    "VZ": "Communication", "T": "Communication",
    "BRK-B": "Financials",
}

# ─── Data Fetching ───────────────────────────────────────────────────────────

def _fetch_single(ticker: str, period: str, interval: str) -> tuple[str, pd.Series | None]:
    """Fetch closing prices for a single ticker using yahooquery."""
    try:
        t = Ticker(ticker)
        interval_map = {"1d": "1d", "1wk": "1wk"}
        yq_interval = interval_map.get(interval, "1d")
        data = t.history(period=period, interval=yq_interval)
        if data.empty or len(data) < 10:
            return ticker, None
        close_series = data["close"].copy()
        close_series.name = ticker
        return ticker, close_series
    except Exception:
        return ticker, None
        close_series = data["close"].copy()
        close_series.name = ticker
        return ticker, close_series
    except Exception:
        return ticker, None


def _period_to_start(period: str, end: pd.Timestamp) -> pd.Timestamp:
    try:
        if period.endswith("mo"):
            return end - pd.DateOffset(months=int(period[:-2]))
        if period.endswith("y"):
            return end - pd.DateOffset(years=int(period[:-1]))
    except Exception:
        pass
    return end - pd.DateOffset(years=1)


def _load_dotenv_if_present() -> None:
    """
    Lightweight .env loader (no external dependency).
    Loads environment variables from `Eigenportfolio/files/.env` (this file's folder),
    without overwriting already-set variables (deployment-safe).
    """
    env_path = Path(__file__).resolve().with_name(".env")
    if not env_path.exists():
        return

    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and not os.getenv(key):
                os.environ[key] = value
    except Exception:
        return


def _stooq_symbols(ticker: str) -> list[str]:
    """
    Return candidate Stooq symbols for a given ticker.

    Stooq uses lowercase and exchange suffixes like `.us`.
    This is best-effort; if it fails we fall back to demo data.
    """
    t = (ticker or "").strip()
    if not t:
        return []

    # NIFTY (best-effort): RELIANCE.NS -> reliance.in
    if t.upper().endswith(".NS"):
        base = t[:-3]
        return [f"{base.lower()}.in"]

    # US: AAPL -> aapl.us
    base = t.lower()
    candidates = [
        f"{base}.us",
        f"{base.replace('-', '.')}.us",
        f"{base.replace('.', '-')}.us",
    ]
    # de-dupe while preserving order
    out = []
    for c in candidates:
        if c not in out:
            out.append(c)
    return out


def _fetch_single_stooq(
    session: requests.Session,
    ticker: str,
    period: str,
    interval: str,
    timeout: int = 20,
) -> pd.Series | None:
    """
    Fetch close prices for a single ticker from Stooq (CSV endpoint).
    """
    i_map = {"1d": "d", "1wk": "w"}
    i = i_map.get(interval, "d")
    for sym in _stooq_symbols(ticker):
        try:
            url = f"https://stooq.com/q/d/l/?s={sym}&i={i}"
            r = session.get(url, timeout=timeout)
            if r.status_code != 200 or not r.text or "No data" in r.text:
                continue
            df = pd.read_csv(io.StringIO(r.text))
            if df.empty or "Date" not in df.columns:
                continue
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date"]).sort_values("Date")
            col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else None)
            if col is None:
                continue

            s = pd.Series(df[col].values, index=df["Date"], name=ticker).dropna()
            if s.empty:
                continue
            end = pd.to_datetime(s.index.max())
            start = _period_to_start(period, end)
            s = s[s.index >= start]
            if s.dropna().shape[0] < 10:
                continue
            return s
        except Exception:
            continue
    return None


def _fetch_single_alpha_vantage(
    session: requests.Session,
    ticker: str,
    period: str,
    interval: str,
    api_key: str,
    timeout: int = 25,
) -> pd.Series | None:
    """
    Fetch adjusted close prices for a ticker from Alpha Vantage.

    Free tier is rate-limited (typically 5 calls/minute). This function is best-effort and will
    return None on rate-limits or unsupported symbols.
    """
    api_key = (api_key or "").strip()
    if not api_key:
        return None

    interval = (interval or "1d").lower()
    if interval == "1wk":
        fn = "TIME_SERIES_WEEKLY_ADJUSTED"
        ts_key = "Weekly Adjusted Time Series"
        price_key_candidates = ["5. adjusted close", "4. close"]
    else:
        fn = "TIME_SERIES_DAILY_ADJUSTED"
        ts_key = "Time Series (Daily)"
        price_key_candidates = ["5. adjusted close", "4. close"]

    try:
        r = session.get(
            "https://www.alphavantage.co/query",
            params={
                "function": fn,
                "symbol": ticker,
                "apikey": api_key,
                "outputsize": "compact",
            },
            timeout=timeout,
        )
        if r.status_code != 200:
            return None
        payload = r.json()
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("Note") or payload.get("Error Message") or payload.get("Information"):
        return None

    ts = payload.get(ts_key)
    if not isinstance(ts, dict) or not ts:
        return None

    rows: list[tuple[pd.Timestamp, float]] = []
    for d, fields in ts.items():
        if not isinstance(fields, dict):
            continue
        val = None
        for k in price_key_candidates:
            if k in fields:
                val = fields.get(k)
                break
        if val is None:
            continue
        try:
            dt = pd.to_datetime(d, errors="coerce")
            if pd.isna(dt):
                continue
            px = float(val)
        except Exception:
            continue
        rows.append((dt, px))

    if not rows:
        return None

    rows.sort(key=lambda x: x[0])
    idx = [x[0] for x in rows]
    vals = [x[1] for x in rows]
    s = pd.Series(vals, index=idx, name=ticker).dropna()

    if s.empty:
        return None

    end = pd.to_datetime(s.index.max())
    start = _period_to_start(period, end)
    s = s[s.index >= start]
    if s.dropna().shape[0] < 10:
        return None

    return s


@st.cache_data(ttl=3600, show_spinner=False)
def generate_demo_prices(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
    seed: int = 7,
) -> pd.DataFrame:
    """
    Deterministic demo price series when market data providers are unavailable.
    """
    tickers = [t for t in (tickers or []) if t]
    if not tickers:
        return pd.DataFrame()

    if interval == "1wk":
        dates = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=260, freq="W-FRI")
    else:
        dates = pd.bdate_range(end=pd.Timestamp.utcnow().normalize(), periods=260)

    # Trim to period
    end = dates.max()
    start = _period_to_start(period, end)
    dates = dates[dates >= start]

    rng = np.random.default_rng(seed)
    df = pd.DataFrame(index=dates)
    # keep bounded for UI responsiveness, but allow universes + comparisons
    for i, t in enumerate(tickers[:60]):
        # stagger per ticker for uniqueness
        local = np.random.default_rng(seed + i * 11)
        mu = 0.10 + 0.02 * (i % 5)      # drift
        sigma = 0.18 + 0.03 * (i % 7)   # vol
        dt = 1 / 252
        shocks = local.normal(0, 1, size=len(dates))
        lr = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
        price = 100 * np.exp(np.cumsum(lr))
        df[t] = price
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_data(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
    sleep_seconds: float = 2.0,
    allow_stooq_fallback: bool = True,
    allow_alpha_vantage_fallback: bool = True,
    max_alpha_calls: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """
    Fetch adjusted closing prices for a list of tickers.

    Returns (df, report):
      - df: DataFrame with tickers as columns, dates as index
      - report: metadata about the fetch attempt (method, failures, etc.)
    """
    _load_dotenv_if_present()
    alpha_key = (os.getenv("ALPHAVANTAGE_API_KEY") or "").strip()

    tickers = [t for t in tickers if t]
    ticker_map = {t.replace("&", "%26"): t for t in tickers}
    tickers_yf = list(ticker_map.keys())
    report: dict = {"method": None, "n_requested": len(tickers), "n_ok": 0, "failed": []}

    if not tickers_yf:
        report["method"] = "empty"
        return pd.DataFrame(), report

    # Prefer single-ticker fetches in deployments (reduces partial/missing data issues).
    results: dict[str, pd.Series] = {}
    failed: list[str] = []
    used_yf: set[str] = set()
    used_av: set[str] = set()

    for i, t_yf in enumerate(tickers_yf):
        t_original = ticker_map.get(t_yf, t_yf)
        series = None

        for attempt in range(2):
            try:
                t = Ticker(t_yf)
                interval_map = {"1d": "1d", "1wk": "1wk"}
                yq_interval = interval_map.get(interval, "1d")
                data = t.history(period=period, interval=yq_interval)
                if data is None or data.empty:
                    raise RuntimeError("empty")

                if "close" in data.columns:
                    series = data["close"].copy()
                    series.name = t_original
                elif "adjclose" in data.columns:
                    series = data["adjclose"].copy()
                    series.name = t_original
                else:
                    raise RuntimeError("missing_close")

                if series is not None:
                    if isinstance(series.index, pd.MultiIndex):
                        series = series.droplevel(0)
                    if not isinstance(series.index, pd.DatetimeIndex):
                        series.index = pd.to_datetime(series.index, errors='coerce')
                    if series.dropna().shape[0] >= 10:
                        break
            except Exception as e:
                if attempt == 0:
                    time.sleep(0.5)
                    continue

        if series is None or series.dropna().shape[0] < 10:
            failed.append(t_original)
        else:
            results[t_original] = series
            used_yf.add(t_original)

        # critical: slow down to reduce rate-limits in deployments
        if sleep_seconds and i < len(tickers_yf) - 1:
            time.sleep(float(sleep_seconds))

    # Session for HTTP requests (Alpha Vantage, Stooq)
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        }
    )

    # Alpha Vantage fallback for the tickers that Yahoo couldn't fetch.
    # Uses env var `ALPHAVANTAGE_API_KEY` (optionally loaded from `Eigenportfolio/files/.env`).
    if allow_alpha_vantage_fallback and alpha_key and failed:
        # Free-tier safety: cap requests to avoid long waits / hard rate limits.
        alpha_targets = list(dict.fromkeys(failed))[: max(0, int(max_alpha_calls))]
        alpha_delay = max(float(sleep_seconds or 0.0), 12.5)
        alpha_failed: list[str] = []

        for i, t in enumerate(alpha_targets):
            s = _fetch_single_alpha_vantage(session, t, period=period, interval=interval, api_key=alpha_key)
            if s is None:
                alpha_failed.append(t)
            else:
                results[t] = s
                used_av.add(t)
            if alpha_delay and i < len(alpha_targets) - 1:
                time.sleep(float(alpha_delay))

        # Refresh failed list: any ticker still missing in results remains failed (includes not-attempted due to cap).
        failed = [t for t in tickers if t not in results]
        report["alpha_vantage_attempted"] = alpha_targets
        report["alpha_vantage_used"] = sorted(list(used_av))
        report["alpha_vantage_rate_limit_safe_delay_sec"] = float(alpha_delay)
        report["alpha_vantage_failed"] = alpha_failed

    if not results:
        # Optional tertiary fallback retained for non-deployment local runs.
        if allow_stooq_fallback:
            stooq_results: dict[str, pd.Series] = {}
            stooq_failed: list[str] = []
            for i, t in enumerate(tickers):
                s = _fetch_single_stooq(session, t, period=period, interval=interval)
                if s is None:
                    stooq_failed.append(t)
                else:
                    stooq_results[t] = s
                if sleep_seconds and i < len(tickers) - 1:
                    time.sleep(float(sleep_seconds))

            if stooq_results:
                df = pd.DataFrame(stooq_results)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df = df.loc[:, df.isna().mean() < 0.30]
                df = df.ffill().bfill()
                report["method"] = "stooq"
                report["n_ok"] = int(df.shape[1])
                report["failed"] = stooq_failed
                return df, report

            report["method"] = "download_single+stooq_failed"
            report["n_ok"] = 0
            report["failed"] = stooq_failed or failed or list(tickers)

        # Final fallback: generate deterministic demo data when all providers fail.
        # This ensures the app works in restricted/deployment environments.
        if tickers:
            demo_df = generate_demo_prices(tickers, period=period, interval=interval)
            if not demo_df.empty:
                report["method"] = "demo_fallback"
                report["n_ok"] = int(demo_df.shape[1])
                report["failed"] = []
                return demo_df, report

        return pd.DataFrame(), report

    df = pd.DataFrame(results)
    if df.empty:
        return df, report
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values(0)
    if df.index.dtype == object or df.index.dtype.kind == 'O':
        df.index = pd.to_datetime(df.index, errors='coerce')
    df = df.sort_index()
    df = df.loc[:, df.isna().mean() < 0.30]
    df = df.ffill().bfill()

    if used_yf and used_av:
        report["method"] = "download_single+alpha_vantage"
    elif used_av and not used_yf:
        report["method"] = "alpha_vantage"
    else:
        report["method"] = "download_single"
    report["n_ok"] = int(df.shape[1])
    report["failed"] = failed
    return df, report


@st.cache_data(ttl=300, show_spinner=False)
def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log-returns and drop any remaining NaNs."""
    lr = np.log(prices / prices.shift(1)).dropna()
    return lr


def normalize_returns(returns: pd.DataFrame) -> pd.DataFrame:
    """Z-score normalize each return series (zero mean, unit variance)."""
    return (returns - returns.mean()) / returns.std()


def get_rolling_volatility(returns: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Annualised rolling volatility (21-day window by default)."""
    return returns.rolling(window).std() * np.sqrt(252)


def get_universe_dict(market: str) -> dict[str, str]:
    return SP500_STOCKS if market == "S&P 500" else NIFTY50_STOCKS


def get_sector(ticker: str, market: str) -> str:
    if market == "S&P 500":
        return SECTOR_MAP_SP500.get(ticker, "Other")
    return "Other"
