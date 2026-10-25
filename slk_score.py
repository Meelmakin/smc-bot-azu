"""
slk_score.py -- score the slkradar bot's alerts against real price data.

Input : slk_alerts.csv (alerts transcribed from your screenshots, times in WAT = UTC+1).
Prices: Yahoo 1H data (needs internet), cached in slk_cache/.

How each alert is scored (the bot gives bias, not entries):
  stop   = the extreme of the rejection candle named in the alert (high for SELL, low for BUY),
           computed from the price data between the "Rejected on" time and +24h (+168h for W1).
  ENTRY A: at the close of the last 1H bar before the alert was sent.
  ENTRY B: a limit order at the alert's breakout level, valid for 48 1H bars (pullback entry).
           Alerts with no breakout level, or where price never pulls back, are not counted.
  target = T x risk for T in 0.25R ... 3R. A bar that touches both stop and target counts as a stop.
  cost   = COST_PCT of price per trade, converted to R.
A random-entry control uses the same pairs, directions and stop sizes at random times.

Limits: about 30 alerts picked by one person, many in the same market move. This can show the
method is NOT obviously broken or lucky. It cannot prove a win rate.

Run:  python slk_score.py            (cache: slk_cache/, results: slk_scored.csv)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import bias_backtest as bt
import bias_bot as bb

WAT = pd.Timedelta(hours=1)               # WAT = UTC+1
TARGETS = bt.TARGETS
COST_PCT = 0.0002
MAXBARS = 240
PULLBACK_BARS = 48
TICKERS = {                                # names that bias_bot.PAIRS does not cover
    "BCHUSDT": "BCH-USD", "ETCUSDT": "ETC-USD", "ARBUSDT": "ARB11841-USD",
    "ETHUSD": "ETH-USD", "NAS100": "NQ=F", "UK100": "^FTSE", "JPN225": "^N225",
    "USOIL": "CL=F", "XAGUSD": "SI=F", "XAUUSD": "GC=F",
}
INDICES = {"NAS100", "UK100", "JPN225", "US30", "US100", "US500", "GER40"}
COMMODITIES = {"USOIL", "XAGUSD", "XAUUSD"}


def ticker_for(pair):
    if pair in TICKERS:
        return TICKERS[pair]
    for t, n in bb.PAIRS.items():
        if n == pair:
            return t
    return None


def asset_class(pair):
    if pair.endswith("USDT") or pair in ("ETHUSD", "BTCUSD"):
        return "crypto"
    if pair in INDICES:
        return "index"
    if pair in COMMODITIES:
        return "commodity"
    return "fx"


# ---------------------------------------------------------------- input
def load_alerts(path):
    df = pd.read_csv(path, comment="#")
    for c in ("rej_open_wat", "bo_time_wat", "alert_wat"):
        t = pd.to_datetime(df[c], errors="coerce")
        df[c.replace("_wat", "_utc")] = (t - WAT).dt.tz_localize("UTC")
    df["htf_hours"] = df["htf_hours"].fillna(24).astype(int)
    df["bo_level"] = pd.to_numeric(df["bo_level"], errors="coerce")
    df["direction"] = df["direction"].str.upper()
    return df


def load_prices(pair, cache_dir="slk_cache", refresh=False):
    t = ticker_for(pair)
    if t is None:
        raise RuntimeError(f"no Yahoo ticker known for {pair}")
    path = os.path.join(cache_dir, f"{pair}.csv")
    if os.path.exists(path) and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    df = bb.fetch_1h(t, period="730d")
    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(path)
    return df


# ---------------------------------------------------------------- scoring
def rejection_extreme(df, r0, hours, direction):
    seg = df[(df.index >= r0) & (df.index < r0 + pd.Timedelta(hours=hours))]
    if len(seg) < 3:
        return None
    return float(seg["High"].max()) if direction == "SELL" else float(seg["Low"].min())


def score_alert(a, df, cost_pct=COST_PCT, maxbars=MAXBARS, pullback_bars=PULLBACK_BARS):
    idx = df.index
    op, hi, lo, cl = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    buy = a["direction"] == "BUY"
    out = {"id": a["id"], "pair": a["pair"], "direction": a["direction"], "kind": a["kind"],
           "alert_utc": a["alert_utc"], "status": "ok"}
    stop = rejection_extreme(df, a["rej_open_utc"], a["htf_hours"], a["direction"])
    if stop is None:
        out["status"] = "no price data for the rejection candle"
        return out
    i0 = idx.searchsorted(a["alert_utc"])
    if i0 < 1 or i0 >= len(df):
        out["status"] = "no price data at alert time"
        return out
    ref = float(cl[i0 - 1])
    risk = ref - stop if buy else stop - ref
    if risk <= 0:
        out["status"] = "price already beyond the stop at alert time"
        return out
    out.update(ref=ref, stop=stop, risk_pct=100 * risk / ref)
    for T in TARGETS:
        o = bt.outcome(buy, ref, stop, T, hi, lo, cl, i0, maxbars)
        out[f"A_{T:g}"] = None if o is None else o[0] - cost_pct * ref / risk

    L = a["bo_level"]
    use_level = bool(np.isfinite(L))
    if use_level and asset_class(a["pair"]) not in ("fx", "crypto"):
        # CFD prices on indices/commodities differ from Yahoo's futures/cash prices, so a level
        # copied from the alert can sit far from Yahoo's price. Don't compare them.
        out["level_note"] = "level not used: price basis differs from Yahoo"
        use_level = False
    if use_level and "bo_time_utc" in a and pd.notna(a["bo_time_utc"]):
        j = idx.searchsorted(a["bo_time_utc"] + pd.Timedelta(hours=3))
        if j < len(df) and abs(cl[j] - L) / L > 0.015:
            out["level_note"] = f"level not used: {100 * abs(cl[j] - L) / L:.1f}% away from Yahoo price"
            use_level = False
    if use_level:
        out["moved_R"] = ((ref - L) if buy else (L - ref)) / risk     # how far price ran before the alert
        k = None
        for j in range(i0, min(len(df), i0 + pullback_bars)):
            if (lo[j] <= L) if buy else (hi[j] >= L):
                k = j
                break
        out["B_filled"] = False
        if k is not None:
            # a limit order fills at the level, or at the open if price gapped through it
            fillp = min(L, op[k]) if buy else max(L, op[k])
            rb = (fillp - stop) if buy else (stop - fillp)
            if rb > 0:
                out["B_filled"] = True
                cost_r = cost_pct * fillp / rb
                stopped = (lo[k] <= stop) if buy else (hi[k] >= stop)
                for T in TARGETS:
                    if stopped:
                        out[f"B_{T:g}"] = -1.0 - cost_r
                        continue
                    o = bt.outcome(buy, fillp, stop, T, hi, lo, cl, k + 1, maxbars)
                    out[f"B_{T:g}"] = None if o is None else o[0] - cost_r
    return out


# ---------------------------------------------------------------- statistics
def wilson(wins, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def summarize(R):
    R = np.asarray([r for r in R if r is not None and np.isfinite(r)], dtype=float)
    n = len(R)
    if n == 0:
        return None
    wins = int((R > 0).sum())
    lo, hi = wilson(wins, n)
    gp, gl = R[R > 0].sum(), -R[R < 0].sum()
    se = R.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    return dict(n=n, wins=wins, win=100 * wins / n, lo=100 * lo, hi=100 * hi, exp=R.mean(), se=se,
                total=R.sum(), pf=(gp / gl) if gl > 0 else float("inf"))


def fmt(s):
    if s is None:
        return "no trades"
    pf = f"{s['pf']:.2f}" if np.isfinite(s["pf"]) else "inf"
    return (f"n={s['n']:<3d} win={s['win']:4.1f}% (95% range {s['lo']:.0f}-{s['hi']:.0f}%)  "
            f"exp={s['exp']:+.2f}R +/-{1.96 * s['se']:.2f}  total={s['total']:+.1f}R  PF={pf}")


def random_control(scored, data, cost_pct=COST_PCT, maxbars=MAXBARS, reps=200, seed=0):
    """Mean R per target for `reps` random portfolios: same pair, direction and stop size per alert."""
    rng = np.random.default_rng(seed)
    ok = scored[scored["status"] == "ok"]
    arrays = {p: (d["High"].values, d["Low"].values, d["Close"].values) for p, d in data.items()}
    res = {T: [] for T in TARGETS}
    for _ in range(reps):
        per = {T: [] for T in TARGETS}
        for _, r in ok.iterrows():
            hi, lo, cl = arrays[r["pair"]]
            if len(cl) < maxbars + 300:
                continue
            i0 = int(rng.integers(100, len(cl) - maxbars - 1))
            buy = r["direction"] == "BUY"
            ref = float(cl[i0 - 1])
            risk = ref * r["risk_pct"] / 100.0
            stop = ref - risk if buy else ref + risk
            for T in TARGETS:
                o = bt.outcome(buy, ref, stop, T, hi, lo, cl, i0, maxbars)
                if o is not None:
                    per[T].append(o[0] - cost_pct * ref / risk)
        for T in TARGETS:
            if per[T]:
                res[T].append(float(np.mean(per[T])))
    return res


# ---------------------------------------------------------------- report
def report(alerts, scored, data, reps=200):
    print(f"\nalerts in file: {len(alerts)}   scored: {int((scored['status'] == 'ok').sum())}")
    bad = scored[scored["status"] != "ok"]
    for _, r in bad.iterrows():
        print(f"  not scored: {r['id']} {r['pair']} -> {r['status']}")
    ok = scored[scored["status"] == "ok"].copy()
    if ok.empty:
        print("nothing to score")
        return

    print("\n=== EVERY ALERT (R after costs; '-' = trade still open when the data ended) ===")
    print(f"{'id':<4} {'pair':<8} {'dir':<4} {'kind':<8} {'stop%':>6} {'ran(R)':>7} "
          f"{'0.25R':>6} {'0.5R':>6} {'1R':>6} {'2R':>6} {'3R':>6}")
    for _, r in ok.iterrows():
        def c(T):
            v = r.get(f"A_{T:g}")
            return "-" if v is None or not np.isfinite(v) else f"{v:+.2f}"
        mv = r.get("moved_R")
        mvs = "" if mv is None or not np.isfinite(mv) else f"{mv:+.2f}"
        print(f"{r['id']:<4} {r['pair']:<8} {r['direction']:<4} {r['kind']:<8} {r['risk_pct']:>6.2f} {mvs:>7} "
              + " ".join(f"{c(T):>6}" for T in TARGETS))
    mv = ok["moved_R"].dropna() if "moved_R" in ok else pd.Series(dtype=float)
    if len(mv):
        print(f"'ran(R)' = how far price had already moved past the breakout level when the alert arrived, "
              f"in stop-distances. Median {mv.median():+.2f}R, {int((mv > 1).sum())} of {len(mv)} were already over 1R.")

    obs = {}
    print("\n=== ENTRY A: at the close of the last 1H bar before the alert ===")
    for T in TARGETS:
        s = summarize(ok[f"A_{T:g}"].tolist())
        obs[T] = s["exp"] if s else float("nan")
        print(f"{T:>5g}R  " + fmt(s))

    print("\n=== ENTRY B: limit order at the breakout level after the alert (waits up to 48 hours) ===")
    if "level_note" in ok:
        for note, n_ in ok["level_note"].dropna().value_counts().items():
            print(f"  {n_} alert(s): {note}")
    if "B_filled" not in ok:
        print("  no alert had a usable breakout level")
    else:
        filled = ok[ok["B_filled"] == True]  # noqa: E712
        have = int(ok["moved_R"].notna().sum()) if "moved_R" in ok else 0   # scored alerts that gave a level
        print(f"filled {len(filled)} of {have} alerts that gave a level; the rest never pulled back "
              f"(or the trade was cut off).")
        for T in TARGETS:
            col = f"B_{T:g}"
            if col in filled:
                print(f"{T:>5g}R  " + fmt(summarize(filled[col].tolist())))

    print(f"\n=== CONTROL: random entries, same pairs / directions / stop sizes ({reps} random portfolios) ===")
    ctrl = random_control(ok, data, reps=reps)
    for T in TARGETS:
        v = np.array(ctrl[T])
        if len(v) == 0:
            continue
        p = float((v >= obs[T]).mean()) if np.isfinite(obs[T]) else float("nan")
        print(f"{T:>5g}R  alerts exp={obs[T]:+.2f}R | random mean={v.mean():+.2f}R "
              f"(5th-95th pct {np.percentile(v, 5):+.2f} to {np.percentile(v, 95):+.2f}) | "
              f"random portfolios that did as well: {100 * p:.0f}%")
    print("If most random portfolios do as well, the alerts show no timing skill in this sample.")

    print("\n=== INDEPENDENCE: alerts on the same day in the same asset class are one market move ===")
    ok["day"] = ok["alert_utc"].dt.strftime("%Y-%m-%d")
    ok["cls"] = ok["pair"].map(asset_class)
    for T in (1.0, 2.0):
        g = ok.groupby(["day", "cls"])[f"A_{T:g}"].apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
        g = g.dropna()
        print(f"{T:g}R: {len(ok)} alerts = {len(g)} clusters; cluster-average exp = {g.mean():+.2f}R")
    print("\nBY KIND (1R, entry A):")
    for k, grp in ok.groupby("kind"):
        print(f"  {k:<9s} " + fmt(summarize(grp['A_1'].tolist())))
    n = len(ok)
    print(f"\nWARNING: n={n}. With this few alerts, even a real 80% win rate would show anywhere from "
          f"{wilson(int(round(0.8 * n)), n)[0] * 100:.0f}% to {wilson(int(round(0.8 * n)), n)[1] * 100:.0f}%. "
          "Treat this as a check, not proof.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", default="slk_alerts.csv")
    ap.add_argument("--cache-dir", default="slk_cache")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--out", default="slk_scored.csv")
    a = ap.parse_args()

    alerts = load_alerts(a.alerts)
    data, rows = {}, []
    for pair in sorted(alerts["pair"].unique()):
        try:
            data[pair] = load_prices(pair, a.cache_dir, a.refresh)
            print(f"loaded {pair}: {len(data[pair])} 1H bars to {data[pair].index[-1]:%Y-%m-%d}")
        except Exception as ex:
            print(f"skip {pair}: {type(ex).__name__}: {ex}")
    for _, r in alerts.iterrows():
        if r["pair"] not in data:
            rows.append({"id": r["id"], "pair": r["pair"], "direction": r["direction"], "kind": r["kind"],
                         "alert_utc": r["alert_utc"], "status": "no price data for this pair"})
            continue
        rows.append(score_alert(r, data[r["pair"]]))
    scored = pd.DataFrame(rows)
    scored.to_csv(a.out, index=False)
    report(alerts, scored, data, reps=a.reps)
    print(f"\nscored alerts saved to {a.out}")


if __name__ == "__main__":
    main()
