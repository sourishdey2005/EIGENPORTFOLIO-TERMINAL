"""
rmt.py — Random Matrix Theory (RMT) Engine
Implements:
  - Empirical correlation matrix computation
  - Eigenvalue decomposition
  - Marchenko-Pastur (MP) theoretical distribution
  - Signal vs noise eigenvalue separation
  - Covariance matrix denoising via RMT filtering
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class RMTResult:
    """Container for all RMT decomposition outputs."""
    correlation_matrix: np.ndarray      # Empirical correlation matrix (N×N)
    eigenvalues: np.ndarray             # Sorted descending eigenvalues
    eigenvectors: np.ndarray            # Corresponding eigenvectors (columns)
    lambda_plus: float                  # MP upper edge (noise threshold)
    lambda_minus: float                 # MP lower edge
    lambda_max: float                   # Largest empirical eigenvalue
    n_signal: int                       # # eigenvalues above MP upper edge
    n_noise: int                        # # eigenvalues within MP band
    signal_mask: np.ndarray            # Boolean mask: True = signal
    cleaned_corr: np.ndarray            # Denoised correlation matrix
    q: float                            # Ratio T/N
    sigma2: float                       # Variance of the noise (fit)
    mp_x: np.ndarray                    # x-axis for MP PDF curve
    mp_pdf: np.ndarray                  # Theoretical MP PDF values


# ─── Core Functions ───────────────────────────────────────────────────────────

def compute_correlation_matrix(returns: pd.DataFrame) -> np.ndarray:
    """
    Compute the sample Pearson correlation matrix from log-returns.
    Uses numpy for numerical precision.
    """
    R = returns.values  # T × N
    T, N = R.shape
    # demean each column
    R = R - R.mean(axis=0)
    # normalize by std
    std = R.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    R = R / std
    C = (R.T @ R) / (T - 1)
    # ensure exact symmetry and unit diagonal
    C = (C + C.T) / 2
    np.fill_diagonal(C, 1.0)
    return C


def marchenko_pastur_pdf(x: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """
    Marchenko-Pastur probability density function.
    
    For a Wishart matrix W = (1/T) X^T X where X is T×N with i.i.d. N(0,σ²)
    entries, the limiting eigenvalue density is:
    
        ρ(λ) = (T/N) * sqrt((λ+ - λ)(λ - λ-)) / (2π σ² λ)
    
    Parameters
    ----------
    x      : array of eigenvalues at which to evaluate
    q      : ratio T/N  (must be > 1 for bulk to exist)
    sigma2 : variance of the random matrix elements
    """
    lam_plus = sigma2 * (1 + 1 / np.sqrt(q)) ** 2
    lam_minus = sigma2 * (1 - 1 / np.sqrt(q)) ** 2
    pdf = np.zeros_like(x, dtype=float)
    mask = (x >= lam_minus) & (x <= lam_plus)
    if mask.any():
        xm = x[mask]
        pdf[mask] = (q / (2 * np.pi * sigma2 * xm)) * np.sqrt(
            np.maximum((lam_plus - xm) * (xm - lam_minus), 0)
        )
    return pdf


def fit_marchenko_pastur(eigenvalues: np.ndarray, q: float) -> float:
    """
    Fit σ² of the MP distribution by matching the empirical bulk variance.
    We use the median of eigenvalues inside the bulk region (heuristic).
    """
    # Start with σ²=1 and iterate once
    sigma2 = 1.0
    lam_plus = sigma2 * (1 + 1 / np.sqrt(q)) ** 2
    # Estimate σ² from eigenvalues that are plausibly in the noise band
    noise_eigs = eigenvalues[eigenvalues <= lam_plus * 1.5]
    if len(noise_eigs) > 0:
        # The bulk mean of MP is σ²(1 + 1/q) … solve for σ²
        sigma2 = np.median(noise_eigs) / (1 + 1 / q + 1e-8)
        sigma2 = max(sigma2, 0.01)
    return sigma2


def run_rmt(returns: pd.DataFrame) -> RMTResult:
    """
    Full RMT pipeline:
      1. Compute correlation matrix
      2. Eigen-decompose
      3. Fit Marchenko-Pastur
      4. Identify signal vs noise eigenvalues
      5. Build denoised covariance matrix
    """
    T, N = returns.shape
    q = T / N  # key ratio — must be > 1 for RMT to be sensible

    # ── Step 1: Correlation matrix ──────────────────────────────────────────
    C = compute_correlation_matrix(returns)

    # ── Step 2: Eigendecomposition ──────────────────────────────────────────
    # Use eigh (symmetric) for numerical stability; returns ascending order
    raw_vals, raw_vecs = np.linalg.eigh(C)
    # Sort descending
    idx = np.argsort(raw_vals)[::-1]
    eigenvalues = raw_vals[idx]
    eigenvectors = raw_vecs[:, idx]

    # ── Step 3: Marchenko-Pastur parameters ─────────────────────────────────
    sigma2 = fit_marchenko_pastur(eigenvalues, q)
    lambda_plus = sigma2 * (1 + 1 / np.sqrt(q)) ** 2
    lambda_minus = sigma2 * (1 - 1 / np.sqrt(q)) ** 2
    lambda_max = eigenvalues[0]

    # ── Step 4: Signal vs noise ─────────────────────────────────────────────
    signal_mask = eigenvalues > lambda_plus
    n_signal = int(signal_mask.sum())
    n_noise = N - n_signal

    # ── Step 5: Denoise (RMT filtering) ────────────────────────────────────
    # Replace noise eigenvalues with their average (preserves trace = N)
    cleaned_vals = eigenvalues.copy()
    noise_mean = cleaned_vals[~signal_mask].mean() if n_noise > 0 else 0.0
    cleaned_vals[~signal_mask] = noise_mean
    # Reconstruct: C_clean = V * diag(λ_clean) * V^T
    cleaned_corr = eigenvectors @ np.diag(cleaned_vals) @ eigenvectors.T
    # Re-normalise diagonal to 1
    d = np.sqrt(np.diag(cleaned_corr))
    d[d == 0] = 1.0
    cleaned_corr = cleaned_corr / np.outer(d, d)
    np.fill_diagonal(cleaned_corr, 1.0)

    # ── Step 6: MP PDF curve for visualization ──────────────────────────────
    mp_x = np.linspace(max(lambda_minus * 0.5, 1e-4), lambda_plus * 1.1, 500)
    mp_pdf = marchenko_pastur_pdf(mp_x, q, sigma2)

    return RMTResult(
        correlation_matrix=C,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        lambda_plus=lambda_plus,
        lambda_minus=lambda_minus,
        lambda_max=lambda_max,
        n_signal=n_signal,
        n_noise=n_noise,
        signal_mask=signal_mask,
        cleaned_corr=cleaned_corr,
        q=q,
        sigma2=sigma2,
        mp_x=mp_x,
        mp_pdf=mp_pdf,
    )


def get_eigenvector_stability(
    returns: pd.DataFrame,
    n_windows: int = 10,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Measure stability of top-k eigenvectors across rolling time windows.
    Uses the absolute dot-product between consecutive windows as a stability metric.
    Returns a DataFrame of shape (n_windows-1, top_k).
    """
    T = len(returns)
    window_size = T // n_windows
    stabilities = []

    prev_vecs = None
    for i in range(n_windows):
        window = returns.iloc[i * window_size: (i + 1) * window_size]
        if len(window) < 20:
            continue
        C = compute_correlation_matrix(window)
        vals, vecs = np.linalg.eigh(C)
        idx = np.argsort(vals)[::-1]
        vecs = vecs[:, idx[:top_k]]

        if prev_vecs is not None:
            # Stability = |cos(angle)| between corresponding eigenvectors
            dots = np.abs(np.diag(prev_vecs.T @ vecs))
            stabilities.append(dots)
        prev_vecs = vecs

    if not stabilities:
        return pd.DataFrame()

    df = pd.DataFrame(
        stabilities,
        columns=[f"PC{i+1}" for i in range(top_k)],
    )
    df.index = [f"W{i+1}→W{i+2}" for i in range(len(df))]
    return df
