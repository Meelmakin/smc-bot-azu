"""
bias_retest.py -- the bias alert PLUS a retest entry, scored like a trader would take it.

Assumed entry model (confirm it is how you really enter; change the settings if not):
  1. The bias alert fires (an external breakout: the BOS stage of bias_bot).
  2. Wait up to WAIT_BARS 1H candles for price to come back to the breakout level and for a 1H
     candle to touch it and CLOSE back in the bias direction (BUY: low <= level, close > level).
     Enter at the close of that candle. If a 1H candle closes beyond the rejection extreme first,
     the bias is dead and no trade is taken.
  3. Stop just past that candle's wick (BUF_ATR x ATR(1H) of room). Trades whose stop is tighter than
     MIN_RISK_ATR x ATR(1H) are skipped, because the spread would eat them.
  4. Targets of 1R, 2R, 3R and 5R. A candle that touches both stop and target counts as a stop.
  5. Cost = COST_PCT of price per trade, converted to R (so tight stops pay more cost).
Nothing here looks ahead: each decision uses candles that have already closed.

Run:  python bias_retest.py --style slk
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import bias_backtest as bt
import bias_bot as bb

TARGETS = (1.0, 2.0, 3.0, 5.0)
WAIT_BARS = 48
BUF_ATR = 0.10
MIN_RISK_ATR = 0.30
COST_PCT = 0.0002
MAXBARS = 240
SESSIONS = bt.SESSIONS


def find_entry(buy, L, inval, op, hi, lo, cl, i0, wait_bars=WAIT_BARS):
    """Index of the first 1H candle (from i0) that touches L and closes back in the bias direction.
    Returns (index or None, reason)."""
    for j in range(i0, min(len(cl), i0 + wait_bars)):
        if (cl[j] < inval) if buy else (cl[j] > inval):
            return None, "bias invalidated before a retest"
        touched = lo[j] <= L if buy else hi[j] >= L
        back = cl[j] > L if buy else cl[j] < L
        if touched and back:
            return j, "entered"
    return None, "no retest within the wait"


def simulate(e, df, a1, cost_pct=COST_PCT, maxbars=MAXBARS, wait_bars=WAIT_BARS,
             buf_atr=BUF_ATR, min_risk_atr=MIN_RISK_ATR, targets=TARGETS):
    """Take one bias alert through the retest entry. Returns a dict (status + R per target)."""
    idx = df.index
    op, hi, lo, cl = (df[c].values for c in ("Open", "High", "Low", "Close"))
    buy = e["direction"] == "BUY"
    out = {"pair": e["pair"], "direction": e["direction"], "alert_time": e["time"],
           "score": e.get("score", 0), "kind": e.get("kind", "")}
    i0 = idx.searchsorted(e["time"])
    if i0 < 1 or i0 >= len(df):
        out["status"] = "no data after the alert"
        return out
    j, why = find_entry(buy, e["level"], e["invalidation"], op, hi, lo, cl, i0, wait_bars)
    if j is None:
        out["status"] = why
        return out
    a = a1[j]
    if not np.isfinite(a) or a <= 0:
        out["status"] = "no ATR yet"
        return out
    ref = float(cl[j])
    stop = float(lo[j] - buf_atr * a) if buy else float(hi[j] + buf_atr * a)
    risk = ref - stop if buy else stop - ref
    if risk <= 0 or risk < min_risk_atr * a:
        out["status"] = "stop too tight"
        return out
    out.update(status="entered", entry_time=idx[j] + pd.Timedelta(hours=1), ref=ref, stop=stop,
               risk_pct=100 * risk / ref, risk_atr=risk / a, cost_r=cost_pct * ref / risk, j=j)
    for T in targets:
        o = bt.outcome(buy, ref, stop, T, hi, lo, cl, j + 1, maxbars)
        out[f"R_{T:g}"] = None if o is None else o[0] - out["cost_r"]
    return out


def collect(data, stage="BOS", **kw):
    rows = []
    for pair, df in data.items():
        a1 = bb.atr(df).values
        for e in bb.events_for(df, pair):
            if e["stage"] == stage:
                rows.append(simulate(e, df, a1, **kw))
    return pd.DataFrame(rows)


def session_of(h):
    return bt.session_of(h)


def table(t, col="R_{}", targets=TARGETS, label=""):
    for T in targets:
        s = t[col.format(f"{T:g}")].dropna() if col.format(f"{T:g}") in t else pd.Series(dtype=float)
        print(f"{label}{T:>4g}R  " + bt.fmt(bt.metrics(s.tolist())))


def random_control(entered, data, cost_pct=COST_PCT, maxbars=MAXBARS, reps=100, seed=0, targets=TARGETS):
    """Random entries with the same pair, direction, stop size (in %) as every executed trade."""
    rng = np.random.default_rng(seed)
    arrays = {p: (d["High"].values, d["Low"].values, d["Close"].values) for p, d in data.items()}
    res = {T: [] for T in targets}
    for _ in range(reps):
        per = {T: [] for T in targets}
        for _, r in entered.iterrows():
            hi, lo, cl = arrays[r["pair"]]
            if len(cl) < maxbars + 300:
                continue
            i0 = int(rng.integers(100, len(cl) - maxbars - 1))
            buy = r["direction"] == "BUY"
            ref = float(cl[i0 - 1])
            risk = ref * r["risk_pct"] / 100.0
            stop = ref - risk if buy else ref + risk
            for T in targets:
                o = bt.outcome(buy, ref, stop, T, hi, lo, cl, i0, maxbars)
                if o is not None:
                    per[T].append(o[0] - cost_pct * ref / risk)
        for T in targets:
            if per[T]:
                res[T].append(float(np.mean(per[T])))
    return res


def report(t, data, cut, reps=100):
    n_all = len(t)
    print(f"\nbias alerts tested: {n_all}")
    for status, n in t["status"].value_counts().items():
        print(f"  {status:<36s} {n:>5d}  ({100 * n / n_all:.0f}%)")
    ent = t[t["status"] == "entered"].copy()
    if ent.empty:
        print("no trades were taken")
        return
    ent["alert_time"] = pd.to_datetime(ent["alert_time"])
    ent = ent.sort_values("entry_time")
    print(f"\nstop size: median {ent['risk_pct'].median():.2f}% of price = {ent['risk_atr'].median():.2f} x ATR(1H); "
          f"median cost = {ent['cost_r'].median():.2f}R per trade")

    print("\n=== HEADLINE (all trades, net of costs) ===")
    table(ent)
    print("Win rate falls as the target grows; only expectancy (exp) tells you if it pays.")

    print("\n=== IN-SAMPLE vs OUT-OF-SAMPLE ===")
    print(f"in-sample = entries before {cut:%Y-%m-%d}")
    for T in TARGETS:
        col = f"R_{T:g}"
        a = ent[ent["entry_time"] < cut][col].dropna().tolist()
        b = ent[ent["entry_time"] >= cut][col].dropna().tolist()
        ma, mb = bt.metrics(a), bt.metrics(b)
        print(f"{T:>4g}R  in-sample n={len(a):<4d} exp={ma['exp'] if ma else float('nan'):+.2f}R | "
              f"out-of-sample n={len(b):<4d} exp={mb['exp'] if mb else float('nan'):+.2f}R")

    print("\n=== BY MINIMUM BIAS SCORE (2R and 3R) ===")
    for k in (0, 3, 4, 5):
        s = ent[ent["score"] >= k]
        m2, m3 = bt.metrics(s["R_2"].dropna().tolist()), bt.metrics(s["R_3"].dropna().tolist())
        if m2:
            print(f"score >= {k}: n={m2['n']:<4d} 2R: win={m2['win']:4.1f}% exp={m2['exp']:+.2f}R | "
                  f"3R: win={m3['win']:4.1f}% exp={m3['exp']:+.2f}R")

    ent["cls"] = ent["pair"].map(lambda p: "crypto" if p.endswith("USDT") else (
        "index" if p in ("US30", "US100", "US500", "GER40") else (
            "metal" if p in ("XAUUSD", "XAGUSD") else "fx")))
    ent["session"] = ent["alert_time"].dt.hour.map(session_of)
    for col, title in (("cls", "ASSET CLASS"), ("session", "SESSION OF THE ALERT (UTC)"), ("direction", "DIRECTION")):
        print(f"\n--- BY {title} (3R) ---")
        for key, g in ent.groupby(col):
            print(f"{str(key):<14s} " + bt.fmt(bt.metrics(g.sort_values('entry_time')['R_3'].dropna().tolist())))

    print(f"\n=== CONTROL: random entries with the same pair / direction / stop size ({reps} random portfolios) ===")
    ctrl = random_control(ent, data, reps=reps)
    for T in TARGETS:
        v = np.array(ctrl[T])
        s = ent[f"R_{T:g}"].dropna()
        if len(v) == 0 or len(s) == 0:
            continue
        obs = s.mean()
        p = float((v >= obs).mean())
        print(f"{T:>4g}R  retest entries exp={obs:+.2f}R | random mean={v.mean():+.2f}R "
              f"(5th-95th pct {np.percentile(v, 5):+.2f} to {np.percentile(v, 95):+.2f}) | "
              f"random portfolios that did as well: {100 * p:.0f}%")
    print("If a large share of random portfolios do as well, the entry adds no skill beyond luck.")
    n = len(ent)
    if n < 200:
        print(f"\nWARNING: only {n} trades. Under ~200 trades the result can easily be luck.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs")
    ap.add_argument("--tf", default="D1H4", choices=sorted(bb.PRESETS))
    ap.add_argument("--style", default="slk", choices=sorted(bb.STYLES))
    ap.add_argument("--stage", default="BOS", choices=["BOS", "RETEST"])
    ap.add_argument("--wait", type=int, default=WAIT_BARS)
    ap.add_argument("--buf", type=float, default=BUF_ATR, help="stop room past the wick, in ATR(1H)")
    ap.add_argument("--minrisk", type=float, default=MIN_RISK_ATR, help="skip stops tighter than this many ATR(1H)")
    ap.add_argument("--cost", type=float, default=COST_PCT)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--cache-dir", default="bt_cache")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default="bias_retest_trades.csv")
    a = ap.parse_args()

    bb.use_preset(a.tf)
    bb.use_style(a.style)
    want = set(a.pairs.upper().split(",")) if a.pairs else None
    data = {}
    for ticker, name in bb.PAIRS.items():
        if want and name not in want:
            continue
        try:
            data[name] = bt.load(name, ticker, a.refresh, a.cache_dir)
        except Exception as ex:
            print(f"skip {name}: {type(ex).__name__}: {ex}")
    if not data:
        sys.exit("no data loaded")
    print(f"loaded {len(data)} instruments; style={a.style}, timeframes={a.tf}, stage={a.stage}")
    print(f"assumptions: wait {a.wait}h for a 1H retest candle, stop = wick +{a.buf} ATR, skip stops under "
          f"{a.minrisk} ATR, cost {a.cost:.4%} of price, same-bar stop+target = stop")

    t = collect(data, a.stage, wait_bars=a.wait, buf_atr=a.buf, min_risk_atr=a.minrisk, cost_pct=a.cost)
    if t.empty:
        sys.exit("no bias alerts found")
    t.to_csv(a.out, index=False)
    start = min(d.index[0] for d in data.values())
    end = max(d.index[-1] for d in data.values())
    report(t, data, start + 0.6 * (end - start), reps=a.reps)
    print(f"\nall results saved to {a.out}")


if __name__ == "__main__":
    main()
