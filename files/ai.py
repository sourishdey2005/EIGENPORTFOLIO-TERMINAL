import os
from pathlib import Path
from typing import Optional


def _load_dotenv_if_present() -> None:
    """
    Lightweight .env loader (no external dependency).
    Loads `GEMINI_API_KEY` from `Eigenportfolio/files/.env` (same folder as this file),
    without overwriting an already-set environment variable.
    """
    if os.getenv("GEMINI_API_KEY"):
        return

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


def gemini_generate(prompt: str, model: str = "gemini-3.1-flash-lite-preview") -> str:
    """
    Generate text using Gemini via the `google-genai` SDK.

    Requires `GEMINI_API_KEY` to be set in the environment.
    """
    if not prompt or not prompt.strip():
        return ""

    _load_dotenv_if_present()

    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")

    try:
        from google import genai  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("google-genai is not installed. Add it to requirements and install deps.") from e

    client = genai.Client()
    response = client.models.generate_content(model=model, contents=prompt)
    return (getattr(response, "text", None) or "").strip()


def build_kpi_prompt(kpi: dict, user_question: Optional[str] = None) -> str:
    """
    Build a compact prompt using dashboard KPIs.
    """
    question = (user_question or "").strip() or "Write a concise professional dashboard summary with actionable takeaways."

    parts = [
        "You are a quant research assistant. Use the KPIs below to write a professional summary in KPI/bullet style.",
        "",
        f"Assets (N): {kpi.get('assets_n')}",
        f"Observations (T): {kpi.get('observations_t')}",
        f"q = T/N: {kpi.get('q_t_over_n')}",
        f"λ⁺ (MP upper bound): {kpi.get('lambda_plus')}",
        f"λ_max: {kpi.get('lambda_max')}",
        f"Signal PCs: {kpi.get('signal_pcs')} ({kpi.get('signal_pct')}%)",
        f"PC1 variance explained: {kpi.get('pc1_var_pct')}%",
        f"Avg pairwise correlation: {kpi.get('avg_pairwise_corr')}",
        f"Top-3 PCs variance explained: {kpi.get('top3_var_pct')}%",
    ]

    if kpi.get("best_strategy"):
        parts.extend(
            [
                f"Best strategy: {kpi.get('best_strategy')} (Sharpe {kpi.get('best_sharpe')}, CAGR {kpi.get('best_cagr')})",
                f"Deepest drawdown: {kpi.get('worst_drawdown_strategy')} ({kpi.get('worst_drawdown')})",
            ]
        )

    parts.extend(["", f"User request: {question}", "", "Output format:", "- KPI headline", "- 6–10 bullets max", "- Keep numbers and units", "- Avoid emojis"])
    return "\n".join(parts)
