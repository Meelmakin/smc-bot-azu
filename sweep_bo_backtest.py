"""
D1 sweep -> H4 external breakout backtest  (v2)
Fixes: D1 candles now start 21:00 UTC (same as H4).
Adds:  exit breakdown (TP/SL/TIME) and filter grid:
  lb    = sweep must take out the lowest low / highest high of the prior N D1 candles
  trend = 1H EMA50 must agree with the trade direction at the breakout
  cap   = max risk as a multiple of D1 ATR(14)
Setup: pip install yfinance pandas numpy
"""
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

PAIRS = {
    "EURUSD=X": (0.0001, 1.0), "GBPUSD=X": (0.0001, 1.2), "USDJPY=X": (0.01, 1.0),
    "AUDUSD=X": (0.0001, 1.2), "USDCAD=X": (0.0001, 1.5), "USDCHF=X": (0.0001, 1.5),
    "NZDUSD=X": (0.0001, 1.5), "EURGBP=X": (0.0001, 1.5), "EURJPY=X": (0.01, 1.5),
    "GBPJPY=X": (0.01, 2.0), "AUDJPY=X": (0.01, 1.8), "EURAUD=X": (0.0001, 2.0),
    "GBPAUD=X": (0.0001, 2.5), "GBPCAD=X": (0.0001, 2.5), "EURCAD=X": (0.0001, 2.0),
    "AUDCAD=X": (0.0001, 2.0), "AUDNZD=X": (0.0001, 2.5), "CADJPY=X": (0.01, 2.0),
    "CHFJPY=X": (0.01, 2.5), "NZDJPY=X": (0.01, 2.5),
}

OFFSET = "21h"
SL_BUFFER_PIPS = 2
LIMIT_WAIT_H = 24
MAX_HOLD_H = 24 * 5
MIN_RISK_SPREADS = 3
R_TARGETS = [2, 3]
MODES = ["close", "level"]

# name, sweep lookback (D1 candles), trend filter, risk cap in D1 ATR (None = off)
CONFIGS = [
    ("A base",            1, False, None),
    ("B week sweep",      5, False, None),
    ("C trend",           1, True,  None),
    ("D cap0.5",          1, False, 0.5),
    ("E cap0.3",          1, False, 0.3),
    ("F week+cap0.5",     5, False, 0.5),
    ("G trend+cap0.5",    1, True,  0.5),
    ("H week+trend+cap",  5, True,  0.5),
]


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


def find_setups(d1, h4, h1, ema, lb):
    lowref = d1.l.shift(1).rolling(lb).min()
    highref = d1.h.shift(1).rolling(lb).max()
    tr = pd.concat([d1.h - d1.l, (d1.h - d1.c.shift()).abs(),
                    (d1.l - d1.c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().shift(1)
    out = []
    for k in range(lb, len(d1) - 1):
        cur = d1.iloc[k]
        if np.isnan(atr.iloc[k]):
            continue
        for s in (1, -1):
            if s == 1:
                swept = cur.l < lowref.iloc[k] and cur.c > lowref.iloc[k]
            else:
                swept = cur.h > highref.iloc[k] and cur.c < highref.iloc[k]
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
                    break
                if (s == 1 and b.c > level) or (s == -1 and b.c < level):
                    bo_end = ts + pd.Timedelta(hours=4)
                    idx = h1.index.searchsorted(bo_end) - 1
                    if idx < 0:
                        break
                    c1, e1 = h1.c.iloc[idx], ema.iloc[idx]
                    trend_ok = (c1 > e1) if s == 1 else (c1 < e1)
                    out.append(dict(dir=s, level=level, ext=ext, bo_end=bo_end,
                                    bo_close=b.c, atr=atr.iloc[k], trend_ok=bool(trend_ok)))
                    break
    return out


def simulate(h1, st, mode, rr, pip, spread_pips, max_risk):
    s = st["dir"]
    sl = st["ext"] - s * SL_BUFFER_PIPS * pip
    spread = spread_pips * pip
    i0 = h1.index.searchsorted(st["bo_end"])
    if i0 >= len(h1):
        return None
    hi, lo, cl = h1.h.values, h1.l.values, h1.c.values
    entry = st["bo_close"] if mode == "close" else st["level"]
    risk = abs(entry - sl)
    if risk <= 0 or risk < MIN_RISK_SPREADS * spread:
        return None
    if max_risk is not None and risk > max_risk:
        return None
    cost = spread / risk

    if mode == "close":
        j, start = i0, i0
    else:
        j = None
        for i in range(i0, min(i0 + LIMIT_WAIT_H, len(h1))):
            if (s == 1 and lo[i] <= entry) or (s == -1 and hi[i] >= entry):
                j = i
                break
        if j is None:
            return None
        if (s == 1 and lo[j] <= sl) or (s == -1 and hi[j] >= sl):
            return dict(t=h1.index[j], r=-1 - cost, why="SL")
        start = j + 1

    tp = entry + s * rr * risk
    end = min(start + MAX_HOLD_H, len(h1))
    r, why = None, "TIME"
    for i in range(start, end):
        if (s == 1 and lo[i] <= sl) or (s == -1 and hi[i] >= sl):
            r, why = -1.0, "SL"
            break
        if (s == 1 and hi[i] >= tp) or (s == -1 and lo[i] <= tp):
            r, why = float(rr), "TP"
            break
    if r is None:
        r = s * (cl[end - 1] - entry) / risk
    return dict(t=h1.index[j if j is not None else i0], r=r - cost, why=why)


def summarize(name, trades):
    if len(trades) < 30:
        print(f"{name:<34} n={len(trades)} (too few)")
        return
    df = pd.DataFrame(trades).sort_values("t")
    half = len(df) // 2
    a, b = df.r.iloc[:half].mean(), df.r.iloc[half:].mean()
    w = df.why.value_counts(normalize=True) * 100
    flag = " <<" if df.r.mean() > 0 and a > 0 and b > 0 and len(df) >= 200 else ""
    print(f"{name:<34} n={len(df):<5} avgR={df.r.mean():+.3f} tot={df.r.sum():+7.1f} "
          f"h1={a:+.3f} h2={b:+.3f} TP={w.get('TP', 0):4.1f}% "
          f"SL={w.get('SL', 0):4.1f}% TIME={w.get('TIME', 0):4.1f}%{flag}")


def main():
    res = {}
    for tk, (pip, sp) in PAIRS.items():
        h1 = load(tk)
        if h1 is None:
            print("skip", tk)
            continue
        d1, h4 = resample(h1, "24h"), resample(h1, "4h")
        ema = h1.c.ewm(span=50, adjust=False).mean()
        cache = {lb: find_setups(d1, h4, h1, ema, lb) for lb in (1, 5)}
        print(f"{tk}: {len(cache[1])} / {len(cache[5])} setups")
        for name, lb, trend, cap in CONFIGS:
            for st in cache[lb]:
                if trend and not st["trend_ok"]:
                    continue
                mr = cap * st["atr"] if cap else None
                for m in MODES:
                    for rr in R_TARGETS:
                        r = simulate(h1, st, m, rr, pip, sp, mr)
                        if r:
                            res.setdefault((name, m, rr), []).append(r)
    print("\n=== RESULTS (net of spread; << = positive overall and both halves) ===")
    for name, *_ in CONFIGS:
        for m in MODES:
            for rr in R_TARGETS:
                summarize(f"{name} | {m} | {rr}R", res.get((name, m, rr), []))
        print()


if __name__ == "__main__":
    main()
 
