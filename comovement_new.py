# comovement.py
# -----------------------------------------------------------------------------
# Enrich a period CSV (e.g., first column like '2019Q3') with KO–PEP co-movement
# numbers, using LOCAL price files (no web calls). It:
#   • reads KO.csv and PEP.csv you saved earlier,
#   • computes daily log returns from Adj Close (falls back to Close),
#   • builds a rolling correlation series,
#   • buckets by quarter (YYYYQ#) using the ROLLING WINDOW END DATE,
#   • aggregates (mean ρ, mean z, tanh(mean z)),
#   • merges into your input CSV, and writes an output CSV.
# -----------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# ======================== EDIT THESE ONLY =========================
CSV_IN   = "ko_pep_sim_by_period.csv"                 # input CSV with 2019Q3-style period labels
CSV_OUT  = "ko_pep_sim_by_period_with_corr.csv"       # output CSV path

PRICE_DIR = "price_store"                              # folder containing KO.csv, PEP.csv
FILE_A    = "KO.csv"                                   # first ticker file
FILE_B    = "PEP.csv"                                  # second ticker file

ROLLING_WINDOW = 120                              # trading days (e.g., 60 ≈ ~3 months)

# Optional date filter for the local price data (leave as "" to use full file)
FILTER_START_DATE = ""                                  # e.g., "2001-01-01"
FILTER_END_DATE   = ""                                  # e.g., "2025-11-01" (inclusive)
# =================================================================


# ------------------------- Math helpers ---------------------------
def fisher_z(series: pd.Series) -> pd.Series:
    r = series.clip(lower=-0.999999, upper=0.999999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))

def fisher_inv(z: float | np.ndarray) -> float | np.ndarray:
    return np.tanh(z)


# -------------------- Local price file loaders --------------------
def load_price_series_from_csv(path: Path) -> pd.Series:
    """
    Read a local OHLCV CSV saved by your downloader.
    Uses 'Adj Close' if present, otherwise 'Close'.
    Returns a price Series indexed by Date (DatetimeIndex).
    """
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path.resolve()}")
    df = pd.read_csv(path, parse_dates=["Date"])
    if "Adj Close" in df.columns:
        px = df["Adj Close"].astype("float64")
    elif "AdjClose" in df.columns:
        px = df["AdjClose"].astype("float64")
    elif "Close" in df.columns:
        px = df["Close"].astype("float64")
    else:
        raise ValueError(f"{path.name}: expected 'Adj Close' or 'Close' column.")
    px.index = pd.to_datetime(df["Date"])
    px = px.sort_index()
    # Optional date filter
    if FILTER_START_DATE:
        px = px[px.index >= pd.to_datetime(FILTER_START_DATE)]
    if FILTER_END_DATE:
        px = px[px.index <= pd.to_datetime(FILTER_END_DATE)]
    if px.empty:
        raise ValueError(f"{path.name}: no prices after date filtering.")
    return px

def to_log_returns(px: pd.Series) -> pd.Series:
    r = np.log(px).diff()
    return r


# ------------------- Rolling corr & aggregation -------------------
def compute_rolling_corr(ret_a: pd.Series, ret_b: pd.Series, window: int) -> pd.Series:
    df = pd.concat([ret_a.rename("a"), ret_b.rename("b")], axis=1).dropna()
    if df.shape[0] < window:
        raise ValueError(
            f"Not enough overlapping data for a {window}-day window (overlap rows={df.shape[0]})."
        )
    rho = df["a"].rolling(window).corr(df["b"])
    return rho.dropna()

def to_quarter_label(dts: pd.DatetimeIndex) -> pd.Series:
    # Map rolling-window END dates to 'YYYYQ#' (e.g., '2019Q3')
    return pd.PeriodIndex(dts, freq="Q").astype(str)

def detect_period_column(df: pd.DataFrame) -> str:
    candidates = ["period", "Period", "PERIOD", "quarter", "Quarter", "QUARTER"]
    for c in candidates:
        if c in df.columns:
            return c
    return df.columns[0]  # fallback: first column


# ------------------------------- Main -----------------------------
def main():
    # 1) Read input periods
    in_path = Path(CSV_IN)
    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path.resolve()}")
    base = pd.read_csv(in_path)
    period_col = detect_period_column(base)
    base[period_col] = base[period_col].astype(str)

    # 2) Load local price files and compute daily log returns
    pdir = Path(PRICE_DIR)
    px_a = load_price_series_from_csv(pdir / FILE_A)
    px_b = load_price_series_from_csv(pdir / FILE_B)
    ret_a = to_log_returns(px_a)
    ret_b = to_log_returns(px_b)

    # 3) Rolling correlation over full span
    rho_series = compute_rolling_corr(ret_a, ret_b, ROLLING_WINDOW)
    z_series = fisher_z(rho_series)

    # 4) Assign rolling END dates to quarters and aggregate
    qlab = to_quarter_label(rho_series.index)
    agg = pd.DataFrame({"quarter": qlab, "rho": rho_series.values, "z": z_series.values})
    grouped = (
        agg.groupby("quarter")
           .agg(rho_mean=("rho", "mean"),
                z_mean=("z", "mean"),
                n_windows=("rho", "size"))
           .reset_index()
    )
    grouped["rho_from_mean_z"] = grouped["z_mean"].apply(lambda z: float(fisher_inv(z)))

    # 5) Merge back into your CSV
    out = base.merge(grouped, left_on=period_col, right_on="quarter", how="left")
    out["co_mov_tickers"] = f"{Path(FILE_A).stem}-{Path(FILE_B).stem}"
    out["rolling_window_days"] = ROLLING_WINDOW

    # 6) Save
    out_path = Path(CSV_OUT)
    out.to_csv(out_path, index=False)

    # 7) Short printout
    eff_start = rho_series.index.min().date() if not rho_series.empty else None
    eff_end   = rho_series.index.max().date() if not rho_series.empty else None

    print("=" * 78)
    print("Co-movement (rolling Pearson corr of daily log returns) — LOCAL price files")
    print(f"Prices A file        : {pdir / FILE_A}")
    print(f"Prices B file        : {pdir / FILE_B}")
    if FILTER_START_DATE or FILTER_END_DATE:
        print(f"Date filter applied  : {FILTER_START_DATE or 'min'} → {FILTER_END_DATE or 'max'}")
    if eff_start and eff_end:
        print(f"Window end dates     : {eff_start} → {eff_end}")
    print(f"Rolling window       : {ROLLING_WINDOW} trading days")
    print(f"Input CSV            : {in_path.resolve()}")
    print(f"Output CSV           : {out_path.resolve()}")
    print("-" * 78)
    if not grouped.empty:
        print("Sample aggregated quarters:")
        print(grouped.head(5).to_string(index=False))
    else:
        print("No aggregated quarters produced. Check price files and rolling window.")
    print("=" * 78)

if __name__ == "__main__":
    main()
