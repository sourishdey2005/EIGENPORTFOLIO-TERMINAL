"""
data.py — Data Layer
Fetches, cleans, and prepares stock data for RMT and portfolio analysis.
Supports S&P 500 and NIFTY 50 universes with configurable timeframes.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st

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
    """Fetch closing prices for a single ticker."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval, auto_adjust=True)
        if hist.empty or len(hist) < 10:
            return ticker, None
        return ticker, hist["Close"].rename(ticker)
    except Exception:
        return ticker, None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_data(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Parallel fetch of adjusted closing prices for a list of tickers.
    Returns a DataFrame with tickers as columns, dates as index.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=min(20, len(tickers))) as ex:
        futures = {ex.submit(_fetch_single, t, period, interval): t for t in tickers}
        for f in as_completed(futures):
            ticker, series = f.result()
            if series is not None:
                results[ticker] = series

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.loc[:, df.isna().mean() < 0.30]          # drop columns with >30% NaN
    df = df.ffill().bfill()
    return df


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
