"""Load 1-minute NQ/MNQ bars and split them into regular-session days.

MNQ and NQ trade the same index at the same price; only the dollar value per
point differs (MNQ $2, NQ $20). So NQ 1-minute history is a valid MNQ tape.

Expected CSV columns: a timestamp column in US/Eastern time plus
open, high, low, close, volume. Timestamps may label the bar's START or END
minute; pass ``label="end"`` when a 09:31 bar holds the 09:30-09:31 trades
(the usual export from NinjaTrader / Sierra / many vendors).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time

import numpy as np
import pandas as pd

RTH_OPEN = time(9, 30)
RTH_CLOSE = time(16, 0)


@dataclass(frozen=True)
class Session:
    """One regular-trading-hours day as plain numpy arrays (bar-start minutes)."""

    day: date
    minute: np.ndarray  # minutes since 09:30 for each bar start (0 = 09:30)
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    vwap: np.ndarray  # running RTH VWAP at each bar's close
    prior_close: float  # previous session's RTH close (nan for the first day)
    prior_range_avg: float  # mean RTH high-low of the prior 14 sessions (nan early)
    ema20_prior: float  # 20-day EMA of RTH closes up to the previous session


def load_bars(path: str, label: str = "end", ts_col: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    ts_col = ts_col or next(c for c in df.columns if "time" in c.lower() or "date" in c.lower())
    ts = pd.to_datetime(df[ts_col], format="mixed")
    if label == "end":
        ts = ts - pd.Timedelta(minutes=1)
    out = pd.DataFrame(
        {
            "ts": ts,
            "open": df["open"].astype(float),
            "high": df["high"].astype(float),
            "low": df["low"].astype(float),
            "close": df["close"].astype(float),
            "volume": df["volume"].astype(float),
        }
    )
    return out.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def build_sessions(bars: pd.DataFrame, min_bars: int = 300) -> list[Session]:
    """Cut the tape into RTH sessions. Half days and broken days are dropped."""
    t = bars["ts"].dt.time
    rth = bars[(t >= RTH_OPEN) & (t < RTH_CLOSE)].copy()
    rth["day"] = rth["ts"].dt.date
    rth["minute"] = (rth["ts"].dt.hour * 60 + rth["ts"].dt.minute) - (9 * 60 + 30)

    daily = rth.groupby("day").agg(
        close=("close", "last"), high=("high", "max"), low=("low", "min"), n=("close", "size")
    )
    daily["rng"] = daily["high"] - daily["low"]
    daily["prior_close"] = daily["close"].shift(1)
    daily["prior_range_avg"] = daily["rng"].rolling(14).mean().shift(1)
    daily["ema20_prior"] = daily["close"].ewm(span=20, adjust=False).mean().shift(1)

    sessions: list[Session] = []
    for day, g in rth.groupby("day", sort=True):
        if len(g) < min_bars or g["minute"].iloc[0] != 0:
            continue
        typical = (g["high"].values + g["low"].values + g["close"].values) / 3.0
        vol = np.maximum(g["volume"].values, 1.0)
        vwap = np.cumsum(typical * vol) / np.cumsum(vol)
        d = daily.loc[day]
        sessions.append(
            Session(
                day=day,
                minute=g["minute"].values.astype(int),
                open=g["open"].values,
                high=g["high"].values,
                low=g["low"].values,
                close=g["close"].values,
                volume=g["volume"].values,
                vwap=vwap,
                prior_close=float(d["prior_close"]),
                prior_range_avg=float(d["prior_range_avg"]),
                ema20_prior=float(d["ema20_prior"]),
            )
        )
    return sessions
