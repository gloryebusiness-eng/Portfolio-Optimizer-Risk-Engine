"""Market data loading with local caching.

This module is the only place in the package that touches the network.
Everything downstream receives clean, aligned DataFrames.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parents[3] / "data_cache"


def _cache_key(tickers: list[str], start: str, end: str) -> Path:
    raw = f"{'|'.join(sorted(tickers))}_{start}_{end}"
    digest = hashlib.md5(raw.encode()).hexdigest()[:12]
    return CACHE_DIR / f"prices_{digest}.parquet"


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    use_cache: bool = True,
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    """Fetch split- and dividend-adjusted close prices.

    Returns a DataFrame indexed by date, one column per ticker.
    Tickers with less than `min_coverage` of the sample are dropped.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = _cache_key(tickers, start, end)

    if use_cache and path.exists():
        return pd.read_parquet(path)

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if len(tickers) == 1:
        prices.columns = tickers

    # Drop tickers that barely traded over the window.
    coverage = prices.notna().mean()
    kept = coverage[coverage >= min_coverage].index.tolist()
    dropped = sorted(set(prices.columns) - set(kept))
    if dropped:
        print(f"[loaders] dropped for insufficient history: {dropped}")

    prices = prices[kept].dropna(how="any")

    if prices.empty:
        raise ValueError("No overlapping price history for the requested tickers.")

    prices.to_parquet(path)
    return prices


def to_returns(prices: pd.DataFrame, kind: str = "simple") -> pd.DataFrame:
    """Convert a price panel to periodic returns.

    'simple' returns aggregate correctly across assets (portfolio math).
    'log' returns aggregate correctly across time (compounding math).
    """
    if kind == "simple":
        return prices.pct_change().dropna(how="any")
    if kind == "log":
        import numpy as np
        return np.log(prices / prices.shift(1)).dropna(how="any")
    raise ValueError(f"kind must be 'simple' or 'log', got {kind!r}")