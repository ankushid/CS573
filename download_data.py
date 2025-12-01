# download_prices.py
# -----------------------------------------------------------------------------
# Download daily prices for one or more tickers (default: KO, PEP) from
# 2001-01-01 to 2025-11-01 and store to disk for future reuse. Saves both
# CSV and Parquet per ticker: ./price_store/{TICKER}.csv / .parquet
#
# One-time installs:
#   pip install yfinance pandas-datareader pandas pyarrow
# -----------------------------------------------------------------------------

from __future__ import annotations
import time
from pathlib import Path
import pandas as pd

# ======================= EDIT THESE ONLY =======================
TICKERS      = ["KO", "PEP"]          # add more tickers here
START_DATE   = "2001-01-01"
END_DATE     = "2025-11-01"           # yfinance 'end' is exclusive
OUT_DIR      = "price_store"          # folder to save files
OVERWRITE    = False                  # False = skip if already saved
# Yahoo retry policy
MAX_RETRIES      = 5
BACKOFF_BASE_SEC = 1.0
# ===============================================================


def fetch_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Try Yahoo (yfinance) with exponential backoff. Returns a DataFrame with
    columns: Open, High, Low, Close, Adj Close, Volume (yfinance standard).
    """
    import yfinance as yf
    last_err = None
    for k in range(MAX_RETRIES):
        try:
            # auto_adjust=False so we get both 'Close' and 'Adj Close'
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                threads=False,
                interval="1d",
            )
            if not df.empty:
                # Standardize index name and ensure tz-naive index
                df.index.name = "Date"
                df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
                return df
            last_err = RuntimeError("Empty dataframe from Yahoo.")
        except Exception as e:
            last_err = e
        time.sleep(BACKOFF_BASE_SEC * (2 ** k))
    raise RuntimeError(f"Yahoo failed after retries for {ticker}. Last error: {last_err}")


def fetch_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fallback to Stooq via pandas-datareader. Columns: Open, High, Low, Close, Volume.
    No 'Adj Close' from Stooq; we’ll duplicate Close into Adj Close for compatibility.
    """
    from pandas_datareader import data as pdr
    sym = f"{ticker}.US"  # Stooq symbol format for US equities
    df = pdr.DataReader(sym, "stooq", start=start, end=end)
    df = df.sort_index()
    if df.empty:
        raise RuntimeError("Empty dataframe from Stooq.")
    df.index.name = "Date"
    # Align column names to Yahoo’s schema
    df = df.rename(columns=str.title)  # open->Open etc. (stooq already capitalized usually)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            raise RuntimeError(f"Stooq returned missing column {col} for {ticker}.")
    df["Adj Close"] = df["Close"]  # compatibility placeholder
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    return df


def load_prices(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Robust loader: try Yahoo first; if it fails (rate limit, outage), fall back to Stooq.
    Returns a daily time series with columns:
      Open, High, Low, Close, Adj Close, Volume
    """
    try:
        return fetch_yahoo(ticker, start, end)
    except Exception as yerr:
        try:
            df = fetch_stooq(ticker, start, end)
            # Add a flag so you know which source was used
            df.attrs["source"] = "stooq"
            return df
        except Exception as serr:
            raise RuntimeError(
                f"Failed to load {ticker} from Yahoo and Stooq.\n"
                f"Yahoo error: {yerr}\nStooq error: {serr}"
            )


def save_ticker(df: pd.DataFrame, ticker: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Standardize dtypes
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # Save CSV
    csv_path = out_dir / f"{ticker}.csv"
    df.to_csv(csv_path, index_label="Date", float_format="%.6f")
    # Save Parquet (fast to read later)
    pq_path = out_dir / f"{ticker}.parquet"
    df.to_parquet(pq_path, engine="pyarrow")
    print(f"Saved {ticker}: {csv_path} and {pq_path}")


def already_saved(ticker: str, out_dir: Path) -> bool:
    return (out_dir / f"{ticker}.csv").exists() and (out_dir / f"{ticker}.parquet").exists()


def main():
    out_dir = Path(OUT_DIR)
    for t in TICKERS:
        if not OVERWRITE and already_saved(t, out_dir):
            print(f"Skipping {t} (already saved). Set OVERWRITE=True to refresh.")
            continue
        print(f"Downloading {t} {START_DATE} → {END_DATE} ...")
        df = load_prices(t, START_DATE, END_DATE)
        # Optional sanity checks
        if df.empty:
            print(f"WARNING: {t} returned empty data. Skipping save.")
            continue
        # Ensure expected columns are present
        missing = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c not in df.columns]
        if missing:
            raise RuntimeError(f"{t} missing columns: {missing}")
        save_ticker(df, t, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
