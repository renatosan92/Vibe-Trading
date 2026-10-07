"""BB Rejection v17 backtester for MNQ 15-minute bars.

Usage:
    python bb_rejection_v17.py bars.csv [--tz UTC] [--commission 1.24] [--slippage-ticks 1]

The CSV needs a timestamp column (bar OPEN time) plus open, high, low, close.
Column names are matched case-insensitively; ``--tz`` is the zone of naive
timestamps (aware timestamps are converted as-is).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

NY = "America/New_York"
TICK = 0.25          # MNQ tick size (index points)
TICK_VALUE = 0.50    # USD per tick per micro contract


@dataclass(frozen=True)
class Params:
    bb_len: int = 20
    bb_mult: float = 2.0
    st_len: int = 14
    st_smooth: int = 1
    st_d: int = 3
    st_high: float = 80.0
    st_low: float = 10.0
    tp_ticks: int = 135
    sl_ticks: int = 265
    qty: int = 15
    window_start: int = 10 * 60   # minutes after midnight NY, Candle C open
    window_end: int = 16 * 60     # entries must open before this; also forced exit
    commission_rt: float = 0.0    # USD per contract, round trip
    slippage_ticks: float = 0.0   # per side, applied to entry and every exit


def load_bars(path: str, tz: str = "UTC") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts_col = next(c for c in df.columns if c in ("timestamp", "datetime", "date", "time"))
    ts = pd.to_datetime(df[ts_col])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(tz)
    df.index = ts.dt.tz_convert(NY)
    return df[["open", "high", "low", "close"]].astype(float).sort_index()


def add_indicators(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = df.copy()
    mid = out["close"].rolling(p.bb_len).mean()
    sd = out["close"].rolling(p.bb_len).std(ddof=0)   # population std, as TradingView
    out["bb_up"] = mid + p.bb_mult * sd
    out["bb_lo"] = mid - p.bb_mult * sd
    ll = out["low"].rolling(p.st_len).min()
    hh = out["high"].rolling(p.st_len).max()
    raw = 100 * (out["close"] - ll) / (hh - ll).replace(0, np.nan)
    out["k"] = raw.rolling(p.st_smooth).mean()
    out["d"] = out["k"].rolling(p.st_d).mean()
    return out


def backtest(df: pd.DataFrame, p: Params = Params()) -> pd.DataFrame:
    """Return one row per trade. Bars are indexed by NY-local open time."""
    d = add_indicators(df, p)
    o, h, l, c = (d[x].to_numpy() for x in ("open", "high", "low", "close"))
    up, lo, k = d["bb_up"].to_numpy(), d["bb_lo"].to_numpy(), d["k"].to_numpy()
    idx = d.index
    minutes = idx.hour * 60 + idx.minute
    days = idx.date

    tp_pts, sl_pts, slip = p.tp_ticks * TICK, p.sl_ticks * TICK, p.slippage_ticks * TICK
    trades, traded_days = [], set()
    i = 2
    n = len(d)
    while i < n:
        a, b, cc = i - 2, i - 1, i   # Candle A, B, C
        day = days[cc]
        if day in traded_days or not (p.window_start <= minutes[cc] < p.window_end):
            i += 1
            continue

        side = 0
        if h[a] >= up[a] and k[a] > p.st_high and c[b] < o[b]:
            side = -1
        elif l[a] <= lo[a] and k[a] < p.st_low and c[b] > o[b]:
            side = 1
        if side == 0:
            i += 1
            continue

        entry = o[cc] + side * slip
        tp = o[cc] + side * tp_pts       # targets measured from the Candle C open
        sl = o[cc] - side * sl_pts
        exit_px = reason = None
        j = cc
        while j < n:
            if days[j] != day or minutes[j] >= p.window_end:
                exit_px, reason = o[j], "16:00"      # first bar at/after 16:00 NY
                break
            if j > cc:                                # gap through a level on a later open
                if (side == 1 and o[j] <= sl) or (side == -1 and o[j] >= sl):
                    exit_px, reason = o[j], "SL gap"
                    break
                if (side == 1 and o[j] >= tp) or (side == -1 and o[j] <= tp):
                    exit_px, reason = tp, "TP"        # limit fills at its price
                    break
            hit_sl = l[j] <= sl if side == 1 else h[j] >= sl
            hit_tp = h[j] >= tp if side == 1 else l[j] <= tp
            if hit_sl:                                # both in one bar → assume SL first
                exit_px, reason = sl, "SL"
                break
            if hit_tp:
                exit_px, reason = tp, "TP"
                break
            j += 1
        if exit_px is None:                           # data ended with position open
            j, exit_px, reason = n - 1, c[n - 1], "EOD data"
        if reason != "TP":
            exit_px -= side * slip                    # stops/market exits slip, limits don't

        pts = side * (exit_px - entry)
        pnl = pts / TICK * TICK_VALUE * p.qty - p.commission_rt * p.qty
        trades.append({
            "date": day, "side": "LONG" if side == 1 else "SHORT",
            "candle_a": idx[a], "entry_time": idx[cc], "entry": entry,
            "exit_time": idx[j], "exit": exit_px, "reason": reason,
            "points": round(pts, 2), "pnl_usd": round(pnl, 2),
        })
        traded_days.add(day)
        i = j + 1
    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}
    pnl = trades["pnl_usd"]
    equity = pnl.cumsum()
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    return {
        "trades": len(trades),
        "longs": int((trades["side"] == "LONG").sum()),
        "shorts": int((trades["side"] == "SHORT").sum()),
        "win_rate": round(len(wins) / len(trades), 4),
        "net_pnl_usd": round(pnl.sum(), 2),
        "avg_trade_usd": round(pnl.mean(), 2),
        "profit_factor": round(wins.sum() / -losses.sum(), 3) if losses.sum() < 0 else None,
        "max_drawdown_usd": round((equity.cummax().clip(lower=0) - equity).max(), 2),
        "exits": trades["reason"].value_counts().to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--tz", default="UTC", help="zone of naive timestamps in the CSV")
    ap.add_argument("--commission", type=float, default=0.0, help="USD per contract round trip")
    ap.add_argument("--slippage-ticks", type=float, default=0.0)
    ap.add_argument("--out", default="bb_rejection_v17_trades.csv")
    args = ap.parse_args()

    p = Params(commission_rt=args.commission, slippage_ticks=args.slippage_ticks)
    trades = backtest(load_bars(args.csv, args.tz), p)
    trades.to_csv(args.out, index=False)
    for key, val in summarize(trades).items():
        print(f"{key:>18}: {val}")
    print(f"\ntrades written to {args.out}")


if __name__ == "__main__":
    main()
