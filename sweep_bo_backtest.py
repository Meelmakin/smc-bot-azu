"""
D1 sweep -> H4 external breakout backtest
Setup: pip install yfinance pandas numpy
Run:   python sweep_bo_backtest.py

Rules (bull; bear is mirrored):
 1. D1 candle sweeps previous D1 low and closes back above it  -> bias.
 2. Level = high of the H4 candle that made the sweep low.
 3. Next-Day Rule: within the NEXT D1 candle, first H4 close above level = external BO.
    Skipped if an H4 low trades below the sweep low first.
 4. Entry: "close" = BO candle close, "level" = limit at level (retrace).
 5. SL = sweep low - buffer.  TP = R x risk.
 Conservative: SL and TP in same bar -> SL. Spread charged in R.
Day/H4 boundaries use 21:00 UTC (FX rollover) to match the alert bot's candle times.
"""
import numpy as np
import pandas as pd
import yfinance as yf

# ticker: (pip size, spread in pips)
PAIRS = {
    "EURUSD=X": (0.0001, 1.0), "GBPUSD=X": (0.0001, 1.2), "USDJPY=X": (0.01, 1.0),
    "AUDUSD=X": (0.0001, 1.2), "USDCAD=X": (0.0001, 1.5), "USDCHF=X": (0.0001, 1.5),
    "NZDUSD=X": (0.0001, 1.5), "EURGBP=X": (0.0001, 1.5), "EURJPY=X": (0.01, 1.5),
    "GBPJPY=X": (0.01, 2.0), "AUDJPY=X": (0.01, 1.8), "EURAUD=X": (0.0001, 2.0),
    "GBPAUD=X": (0.0001, 2.5), "GBPCAD=X": (0.0001, 2.5), "EURCAD=X": (0.0001, 2.0),
    "AUDCAD=X": (0.0001, 2.0), "AUDNZD=X": (0.0001, 2.5), "CADJPY=X": (0.01, 2.0),
    "CHFJPY=X": (0.01, 2.5), "NZDJPY=X": (0.01, 2.5),
}

OFFSET = "21h"          # D1/H4 boundary (UTC)
SL_BUFFER_PIPS = 2      # beyond the sweep extreme
LIMIT_WAIT_H = 24       # how long a limit entry stays valid (1H bars)
MAX_HOLD_H = 24 * 5     # force-close after this many 1H bars
MIN_RISK_SPREADS = 3    # skip if risk < 3x spread
R_TARGETS = [2, 3, 4]
MODES = ["close", "level"]


def load(ticker):
    df = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.columns = ["o", "h", "l", "c"]
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df


def resample(df, rule):
    return df.resample(rule, offset=OFFSET).agg(
        {"o": "first", "h": "max", "l": "min", "c": "last"}).dropna()


def find_setups(d1, h4):
    """Yield dicts: direction, level, sweep_ext, bo_end, bo_close."""
    out = []
    for k in range(1, len(d1) - 1):
        prev, cur = d1.iloc[k - 1], d1.iloc[k]
        for s in (1, -1):
            if s == 1:
                swept = cur.l < prev.l and cur.c > prev.l
            else:
                swept = cur.h > prev.h and cur.c < prev.h
            if not swept:
                continue
            t0, t1 = d1.index[k], d1.index[k] + pd.Timedelta(days=1)
            day = h4[(h4.index >= t0) & (h4.index < t1)]
            if day.empty:
                continue
            if s == 1:
                rej = day.loc[day.l.idxmin()]
                level, ext = rej.h, cur.l
            else:
                rej = day.loc[day.h.idxmax()]
                level, ext = rej.l, cur.h
            n0, n1 = d1.index[k + 1], d1.index[k + 1] + pd.Timedelta(days=1)
            nxt = h4[(h4.index >= n0) & (h4.index < n1)]
            for ts, b in nxt.iterrows():
                if (s == 1 and b.l < ext) or (s == -1 and b.h > ext):
                    break  # invalidated before breakout
                if (s == 1 and b.c > level) or (s == -1 and b.c < level):
                    out.append(dict(dir=s, level=level, ext=ext,
                                    bo_end=ts + pd.Timedelta(hours=4), bo_close=b.c))
                    break
    return out


def simulate(h1, st, mode, rr, pip, spread_pips):
    s = st["dir"]
    sl = st["ext"] - s * SL_BUFFER_PIPS * pip
    spread = spread_pips * pip
    i0 = h1.index.searchsorted(st["bo_end"])
    if i0 >= len(h1):
        return None
    hi, lo, cl = h1.h.values, h1.l.values, h1.c.values

    if mode == "close":
        entry, j, start = st["bo_close"], i0, i0
    else:
        entry, j = st["level"], None
        for i in range(i0, min(i0 + LIMIT_WAIT_H, len(h1))):
            if (s == 1 and lo[i] <= entry) or (s == -1 and hi[i] >= entry):
                j = i
                break
        if j is None:
            return None
        start = j + 1
        # fill bar also hit the stop -> loss (conservative)
        if (s == 1 and lo[j] <= sl) or (s == -1 and hi[j] >= sl):
            risk = abs(entry - sl)
            return dict(t=h1.index[j], r=-1 - spread / risk)

    risk = abs(entry - sl)
    if risk < MIN_RISK_SPREADS * spread or risk <= 0:
        return None
    tp = entry + s * rr * risk
    end = min(start + MAX_HOLD_H, len(h1))
    r = None
    for i in range(start, end):
        if (s == 1 and lo[i] <= sl) or (s == -1 and hi[i] >= sl):
            r = -1.0
            break
        if (s == 1 and hi[i] >= tp) or (s == -1 and lo[i] <= tp):
            r = float(rr)
            break
    if r is None:
        r = s * (cl[end - 1] - entry) / risk
    return dict(t=h1.index[j if j is not None else i0], r=r - spread / risk)


def summarize(name, trades):
    if not trades:
        print(f"{name}: no trades")
        return
    df = pd.DataFrame(trades).sort_values("t")
    half = len(df) // 2
    a, b = df.r.iloc[:half].mean(), df.r.iloc[half:].mean()
    print(f"{name:<14} n={len(df):<5} win={100 * (df.r > 0).mean():5.1f}%  "
          f"avgR={df.r.mean():+.3f}  totalR={df.r.sum():+7.1f}  "
          f"1st half={a:+.3f}  2nd half={b:+.3f}")


def main():
    res = {(m, rr): [] for m in MODES for rr in R_TARGETS}
    for tk, (pip, sp) in PAIRS.items():
        h1 = load(tk)
        if h1 is None:
            print("skip", tk)
            continue
        d1, h4 = resample(h1, "1D"), resample(h1, "4h")
        setups = find_setups(d1, h4)
        print(f"{tk}: {len(setups)} setups")
        for m in MODES:
            for rr in R_TARGETS:
                for st in setups:
                    r = simulate(h1, st, m, rr, pip, sp)
                    if r:
                        r["pair"] = tk
                        res[(m, rr)].append(r)
    print("\n=== RESULTS (net of spread) ===")
    for m in MODES:
        for rr in R_TARGETS:
            summarize(f"{m} @ {rr}R", res[(m, rr)])


if __name__ == "__main__":
    main()
