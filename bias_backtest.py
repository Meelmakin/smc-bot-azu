"""
bias_backtest.py -- how often do the bias bot's alerts follow through?

The bot gives bias, not entries, so each BOS / RETEST alert is scored as if
you entered at the close of the alert candle:
  stop   = the bot's own invalidation level (rejection candle extreme)
  target = T x risk, for T in 1R, 2R, 3R
  A bar touching the stop counts as a stop, even if it also touched the
  target (conservative). No target/stop within MAXBARS 1H bars -> timeout,
  marked to market at that bar's close.
  Cost = COST_PCT of price per trade, converted to R (spread + slippage).

No look-ahead: bias_bot.events_for() only uses completed candles, and each
trade is walked forward only from the bar AFTER the alert candle.

Run (needs internet; Yahoo gives ~730 days of 1H data):
    python bias_backtest.py                  # all pairs, baseline + variants
    python bias_backtest.py --pairs EURUSD,GBPUSD,XAUUSD --refresh
    python bias_backtest.py --cache-dir bt_cache --no-variants
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import bias_bot as bb

TARGETS = (1.0, 2.0, 3.0)
STAGES = ("BOS", "RETEST")
SESSIONS = [(0, 7, "Asia 00-07"), (7, 12, "London 07-12"), (12, 16, "Overlap 12-16"),
            (16, 21, "NY late 16-21"), (21, 24, "Late 21-24")]


# ---------------------------------------------------------------- data
def load(pair, ticker, refresh, cache_dir):
    path = os.path.join(cache_dir, f"{pair}.csv")
    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    df = bb.fetch_1h(ticker, period="730d")
    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(path)
    return df


# ---------------------------------------------------------------- scoring
def outcome(buy, ref, stop, T, hi, lo, cl, i0, maxbars):
    """(R before costs, kind) or None if the trade is still open at the end of data."""
    risk = abs(ref - stop)
    target = ref + T * risk if buy else ref - T * risk
    end = min(len(hi), i0 + maxbars)
    for k in range(i0, end):
        hit_stop = lo[k] <= stop if buy else hi[k] >= stop
        hit_tp = hi[k] >= target if buy else lo[k] <= target
        if hit_stop:                       # ambiguous bar -> stop first
            return -1.0, "stop"
        if hit_tp:
            return T, "target"
    if i0 + maxbars > len(hi):
        return None                        # not enough data to finish this trade
    r = (cl[end - 1] - ref) / risk if buy else (ref - cl[end - 1]) / risk
    return max(-1.0, min(T, r)), "timeout"


def collect(data, stages, cost_pct, maxbars):
    rows = []
    for pair, df in data.items():
        idx = df.index
        hi, lo, cl = df["High"].values, df["Low"].values, df["Close"].values
        for e in bb.events_for(df, pair):
            if e["stage"] not in stages:
                continue
            i0 = idx.searchsorted(e["time"])
            if i0 < 1 or i0 >= len(df):
                continue
            buy = e["direction"] == "BUY"
            ref, stop = float(cl[i0 - 1]), e["invalidation"]
            risk = ref - stop if buy else stop - ref
            if risk <= 0:                  # price already beyond the invalidation level
                continue
            cost_r = cost_pct * ref / risk
            for T in TARGETS:
                res = outcome(buy, ref, stop, T, hi, lo, cl, i0, maxbars)
                if res is None:
                    continue
                r, kind = res
                rows.append(dict(pair=pair, stage=e["stage"], direction=e["direction"],
                                 kind=e["kind"], time=e["time"], hour=e["time"].hour,
                                 T=T, R=r - cost_r, outcome=kind, risk_pct=100 * risk / ref))
    return pd.DataFrame(rows)


def metrics(R):
    """R = chronological list of net R results."""
    arr = np.asarray(R, dtype=float)
    n = len(arr)
    if n == 0:
        return None
    gp, gl = arr[arr > 0].sum(), -arr[arr < 0].sum()
    eq = np.concatenate([[0.0], np.cumsum(arr)])
    dd = (np.maximum.accumulate(eq) - eq).max()
    streak = best = 0
    for x in arr:
        streak = streak + 1 if x < 0 else 0
        best = max(best, streak)
    return dict(n=n, win=100 * (arr > 0).mean(), exp=arr.mean(), total=arr.sum(),
                pf=(gp / gl) if gl > 0 else float("inf"), dd=dd, streak=best)


def fmt(m, extra=""):
    if m is None:
        return "no trades"
    pf = f"{m['pf']:.2f}" if np.isfinite(m["pf"]) else "inf"
    return (f"n={m['n']:<4d} win={m['win']:5.1f}%  exp={m['exp']:+.2f}R  total={m['total']:+7.1f}R  "
            f"PF={pf:<5s} maxDD={m['dd']:5.1f}R  maxLossStreak={m['streak']:<2d}{extra}")


def session_of(h):
    for a, b, name in SESSIONS:
        if a <= h < b:
            return name
    return "?"


def sub(t, **conds):
    out = t
    for k, v in conds.items():
        out = out[out[k] == v]
    return out.sort_values("time")


# ---------------------------------------------------------------- reports
def headline(t):
    print("\n=== HEADLINE (all pairs, net of costs) ===")
    for stage in STAGES:
        for T in TARGETS:
            s = sub(t, stage=stage, T=T)
            to = 100 * (s["outcome"] == "timeout").mean() if len(s) else 0
            print(f"{stage:<6s} {T:.0f}R  " + fmt(metrics(s["R"]), f"  timeout={to:.0f}%"))
    print("BOS and RETEST rows of the same bias overlap: don't add them together.")


def breakdowns(t, stage="RETEST", T=2.0):
    s = sub(t, stage=stage, T=T)
    if s.empty:
        return
    s = s.assign(session=s["hour"].map(session_of))
    for col, title in (("pair", "BY PAIR"), ("session", "BY SESSION (alert time, UTC)"),
                       ("kind", "BY KEY LEVEL TYPE"), ("direction", "BY DIRECTION")):
        print(f"\n--- {stage} {T:.0f}R {title} ---")
        for key, g in s.groupby(col):
            print(f"{str(key):<14s} " + fmt(metrics(g.sort_values('time')['R'])))


def variants(data, cost_pct, maxbars, cut):
    print(f"\n=== VARIANTS (RETEST, 2R) -- in-sample = before {cut:%Y-%m-%d}, out-of-sample = after ===")
    old = (bb.REQUIRE_SWEEP, bb.REQUIRE_D1_TREND)
    try:
        for name, sw, tr in (("baseline", False, False), ("+ sweep", True, False),
                             ("+ D1 trend", False, True), ("+ both", True, True)):
            bb.REQUIRE_SWEEP, bb.REQUIRE_D1_TREND = sw, tr
            s = sub(collect(data, ("RETEST",), cost_pct, maxbars), stage="RETEST", T=2.0)
            if s.empty:
                print(f"{name:<11s} no trades")
                continue
            ins, oos = s[s["time"] < cut], s[s["time"] >= cut]
            mi, mo = metrics(ins["R"]), metrics(oos["R"])
            m = metrics(s["R"])
            print(f"{name:<11s} all: n={m['n']:<4d} win={m['win']:4.1f}% exp={m['exp']:+.2f}R | "
                  f"in-sample: n={mi['n'] if mi else 0:<4d} exp={mi['exp'] if mi else float('nan'):+.2f}R | "
                  f"out-of-sample: n={mo['n'] if mo else 0:<4d} exp={mo['exp'] if mo else float('nan'):+.2f}R")
    finally:
        bb.REQUIRE_SWEEP, bb.REQUIRE_D1_TREND = old
    print("A filter only counts if it helps BOTH halves. A gain in one half is likely noise.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", help="comma list of pair names, e.g. EURUSD,XAUUSD")
    ap.add_argument("--cost", type=float, default=0.0002, help="round-trip cost as fraction of price")
    ap.add_argument("--maxbars", type=int, default=240, help="1H bars before a trade times out")
    ap.add_argument("--cache-dir", default="bt_cache")
    ap.add_argument("--refresh", action="store_true", help="re-download data")
    ap.add_argument("--no-variants", action="store_true")
    ap.add_argument("--out", default="bias_backtest_trades.csv")
    a = ap.parse_args()

    want = set(a.pairs.upper().split(",")) if a.pairs else None
    data = {}
    for ticker, name in bb.PAIRS.items():
        if want and name not in want:
            continue
        try:
            data[name] = load(name, ticker, a.refresh, a.cache_dir)
            print(f"loaded {name}: {len(data[name])} 1H bars, "
                  f"{data[name].index[0]:%Y-%m-%d} to {data[name].index[-1]:%Y-%m-%d}")
        except Exception as ex:
            print(f"skip {name}: {type(ex).__name__}: {ex}")
    if not data:
        sys.exit("no data loaded")

    t = collect(data, STAGES, a.cost, a.maxbars)
    if t.empty:
        sys.exit("no alerts found in this data")
    t.to_csv(a.out, index=False)
    print(f"\nassumptions: cost={a.cost:.4%} of price per trade, timeout={a.maxbars} 1H bars, "
          f"stop = bot invalidation level, same-bar stop+target counts as stop")
    headline(t)
    breakdowns(t)
    start = min(d.index[0] for d in data.values())
    end = max(d.index[-1] for d in data.values())
    if not a.no_variants:
        variants(data, a.cost, a.maxbars, start + 0.6 * (end - start))
    n2 = len(sub(t, stage="RETEST", T=2.0))
    if n2 < 100:
        print(f"\nWARNING: only {n2} RETEST trades. Under ~100-200 trades, results can be luck. "
              "Don't trade or pay for anything based on this alone.")
    print(f"\nall trades saved to {a.out}")


if __name__ == "__main__":
    main()
