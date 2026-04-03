# EIGENPORTFOLIO TERMINAL

An interactive **quant research terminal** for portfolio construction using **Random Matrix Theory (RMT)**, **eigenportfolios**, and **optimization + backtesting**, built with **Streamlit + Plotly**.

## Highlights

- **Universe**: S&P 500 / NIFTY 50 (user-selectable)
- **RMT Engine**: Marchenko–Pastur bounds, signal/noise separation, covariance denoising
- **Eigenportfolios**: single-PC and Top‑K combined eigenportfolio construction
- **Optimization**: Max Sharpe, Min Variance, Risk Parity, Efficient Frontier + Monte Carlo feasible set
- **Backtesting**: rolling-window strategies with configurable rebalance + transaction costs
- **Visualizations**: extensive Plotly library + a dedicated **3D Lab** (surfaces, 3D scatters, portfolio clouds)
- **AI Analyst (Gemini)**: optional KPI-based summary using `google-genai`

## Project Layout

The Streamlit app lives in `Eigenportfolio/files/`:

```
Eigenportfolio/
  README.md
  .gitignore
  files/
    app.py
    data.py
    rmt.py
    portfolio.py
    backtest.py
    visuals.py
    utils.py
    ai.py
    requirements.txt
```

## Quickstart

From the repository root:

```bash
cd Eigenportfolio/files
pip install -r requirements.txt
streamlit run app.py
```

## Environment Variables (Gemini)

Gemini is optional. If you want the in-app **AI Analyst** to work:

1) Create a local env file:

```bash
cd Eigenportfolio/files
copy .env.example .env
```

2) Put your key in `Eigenportfolio/files/.env`:

```
GEMINI_API_KEY=your_key_here
```

Notes:
- `.env` is **ignored by git** via `.gitignore`.
- The app loads `GEMINI_API_KEY` from the environment or from `Eigenportfolio/files/.env`.

## Usage Tips

- Use **Auto Fetch on Change** to recompute when sidebar configuration changes.
- The **3D Lab** can be GPU/VRAM heavy. Start in *Single* render mode, then try *Multi* (up to 6).
- Use **Compare** to analyze up to **10** tickers side-by-side.

## Troubleshooting

- **Blank/white Plotly charts** (especially in 3D): reduce simultaneous 3D renders (use Single render mode), or reduce **3D assets (max)**.
- **Gemini says missing key**: ensure `GEMINI_API_KEY` is set, or create `Eigenportfolio/files/.env` from `.env.example` and restart Streamlit.
- **After updating the app, you see strange runtime errors**: clear Streamlit cache (`streamlit cache clear`) and restart the app.
- **Deployments fail to fetch data**: set **Data Source → Demo (Synthetic)** (recommended for locked-down hosts), or reduce **Max Stocks (Fetch)** and keep **Fetch Delay (sec)** at ~2s to avoid Yahoo Finance rate-limits.
- **Dependency issues**: re-run `pip install -r requirements.txt` in a clean virtual environment.

## Disclaimer

For research/education only. Not financial advice.

