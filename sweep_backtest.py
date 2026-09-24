"""Backtest: liquidity sweep of previous day / previous week high or low (1H).

Rules were fixed BEFORE running (nothing is tuned on the results):
- Levels: previous day high/low (PDH/PDL) and previous week high/low (PWH/PWL), UTC.
- Sweep: the FIRST time price trades beyond a level in the current day/week, and that
  1H candle closes back inside. Wick beyond the level must be >= 0.05 ATR.
  High swept -> SELL. Low swept -> BUY. If the first breach closes beyond the level,
  it is a real breakout and there is no trade.
- Entry: close of the sweep candle (variant D waits for the next candle to close in
  the trade direction without taking out the sweep extreme).
- Stop: beyond the sweep candle extreme + 0.1 ATR. Take profit: 1.5R, 2R, 3R tested.

Uses ~2 years of 1H data (Yahoo). No spread/slippage. SL and TP in the same candle
counts as a loss. Results are split into year 1 and year 2 so you can see whether
anything holds up.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

import scanner as S

RRS = (1.5, 2.0, 3.0)
LOCK_RR = 2.0
MAX_BARS = 96          # trade time limit in 1H bars (4 days)
PEN = 0.05             # minimum wick beyond the level, in ATR
SL_BUF = 0.10          # stop buffer in ATR
MIN_RISK = 0.30        # minimum stop distance in ATR
SESSION = (7, 20)      # UTC hours for variant C

VARIANTS = {
    "A Daily levels only": dict(levels=("PDH", "PDL"), session=False, confirm=False),
    "B Daily + weekly levels": dict(levels=("PDH", "PDL", "PWH", "PWL"), session=False, confirm=False),
    "C B + London/NY only": dict(levels=("PDH", "PDL", "PWH", "PWL"), session=True, confirm=False),
    "D B + wait for confirmation": dict(levels=("PDH", "PDL", "PWH", "PWL"), session=False, confirm=True),
}


def prepare(h1):
    idx = h1.index
    day = idx.floor("D")
    wk = day - pd.to_timedelta(idx.dayofweek, unit="D")
    df = pd.DataFrame({"h": h1["high"].values, "l": h1["low"].values, "day": day, "wk": wk})
    dhl = df.groupby("day").agg(h=("h", "max"), l=("l", "min"))
    whl = df.groupby("wk").agg(h=("h", "max"), l=("l", "min"))
    lv = {
        "PDH": (day.map(dhl["h"].shift(1)).values.astype(float), day.asi8),
        "PDL": (day.map(dhl["l"].shift(1)).values.astype(float), day.asi8),
        "PWH": (wk.map(whl["h"].shift(1)).values.astype(float), wk.asi8),
        "PWL": (wk.map(whl["l"].shift(1)).values.astype(float), wk.asi8),
    }
    return dict(h1=h1, lv=lv, o=h1["open"].values, h=h1["high"].values,
                l=h1["low"].values, c=h1["close"].values, a=S.atr(h1).values,
                hour=idx.hour.values)


def find_events(P):
    h, l, c, a, lv = P["h"], P["l"], P["c"], P["a"], P["lv"]
    breached, events = set(), []
    for i in range(60, len(c)):
        ai = a[i]
        if not np.isfinite(ai) or ai <= 0:
            continue
        for name in ("PWH", "PWL", "PDH", "PDL"):
            arr, keys = lv[name]
            level = arr[i]
            if not np.isfinite(level):
                continue
            k = (name, keys[i])
            if k in breached:
                continue
            if name.endswith("H"):
                if h[i] > level:
                    breached.add(k)
                    if c[i] < level and h[i] - level >= PEN * ai:
                        events.append(dict(i=i, side="SELL", name=name, level=level, extreme=h[i]))
            else:
                if l[i] < level:
                    breached.add(k)
                    if c[i] > level and level - l[i] >= PEN * ai:
                        events.append(dict(i=i, side="BUY", name=name, level=level, extreme=l[i]))
    return events


def outcome(h, l, c, i, d, entry, sl, rr):
    risk = abs(entry - sl)
    tp = entry + rr * risk if d == 1 else entry - rr * risk
    end = min(len(h), i + 1 + MAX_BARS)
    for j in range(i + 1, end):
        if d == 1:
            if l[j] <= sl:
                return -1.0, j
            if h[j] >= tp:
                return rr, j
        else:
            if h[j] >= sl:
                return -1.0, j
            if l[j] <= tp:
                return rr, j
    j = end - 1
    if j <= i:
        return None, i
    r = (c[j] - entry) / risk if d == 1 else (entry - c[j]) / risk
    return float(r), j


def simulate(name, P, events, cfg):
    h, l, c, a, hour = P["h"], P["l"], P["c"], P["a"], P["hour"]
    idx, n = P["h1"].index, len(c)
    trades, lock = [], -1
    for ev in events:
        i = ev["i"]
        if i <= lock or ev["name"] not in cfg["levels"]:
            continue
        if cfg["session"] and not (SESSION[0] <= hour[i] < SESSION[1]):
            continue
        d = 1 if ev["side"] == "BUY" else -1
        e = i
        if cfg["confirm"]:
            j = i + 1
            if j >= n:
                continue
            if d == -1 and not (c[j] < c[i] and h[j] <= ev["extreme"]):
                continue
            if d == 1 and not (c[j] > c[i] and l[j] >= ev["extreme"]):
                continue
            e = j
        entry, ae = c[e], a[e]
        if not np.isfinite(ae) or ae <= 0:
            continue
        if d == -1:
            sl = max(ev["extreme"] + SL_BUF * ae, entry + MIN_RISK * ae)
        else:
            sl = min(ev["extreme"] - SL_BUF * ae, entry - MIN_RISK * ae)
        res = {}
        for rr in RRS:
            r, j = outcome(h, l, c, e, d, entry, sl, rr)
            if r is None:
                res = None
                break
            res[rr] = r
            if rr == LOCK_RR:
                lock = j
        if res:
            trades.append({"t": idx[e], "pair": name, "R": res})
    return trades


def stats(rs):
    n = len(rs)
    if n == 0:
        return None
    wins = sum(1 for r in rs if r > 0)
    cum = peak = dd = streak = worst = 0.0
    for r in rs:
        cum += r
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
        streak = streak + 1 if r <= 0 else 0
        worst = max(worst, streak)
    return dict(n=n, wr=100 * wins / n, tot=cum, exp=cum / n, dd=dd, streak=int(worst),
                noise=2 * float(np.std(rs)) / np.sqrt(n))


def main():
    results = {v: [] for v in VARIANTS}
    errors = []
    for name, ticker in S.PAIRS.items():
        try:
            h1 = S.fetch(ticker, "1h", "729d")
            if len(h1) < 500:
                raise RuntimeError("not enough data")
            P = prepare(h1)
            events = find_events(P)
            for v, cfg in VARIANTS.items():
                results[v] += simulate(name, P, events, cfg)
            print(name, "done", len(events), "sweeps")
        except Exception as e:
            errors.append(f"{name}: {e}")
            print("ERROR", name, e)
    for v in results:
        results[v].sort(key=lambda x: x["t"])

    allt = [t["t"] for tr in results.values() for t in tr]
    lines = ["SWEEP BACKTEST (~2 years, 1H, no spread, nothing tuned)",
             f"Pairs: {len(S.PAIRS)}", ""]
    if not allt:
        lines.append("No trades found.")
    else:
        cut = min(allt) + (max(allt) - min(allt)) / 2
        for v, tr in results.items():
            if not tr:
                lines += [f"{v}: no trades", ""]
                continue
            days = max(1, (tr[-1]["t"] - tr[0]["t"]).days)
            lines.append(f"{v}: {len(tr)} trades (~{len(tr) / days:.1f}/day)")
            for rr in RRS:
                s = stats([t["R"][rr] for t in tr])
                y1 = stats([t["R"][rr] for t in tr if t["t"] < cut])
                y2 = stats([t["R"][rr] for t in tr if t["t"] >= cut])
                yy = " | year1 {:+.2f}, year2 {:+.2f}".format(
                    y1["exp"] if y1 else 0, y2["exp"] if y2 else 0)
                lines.append(f"  {rr:g}R: win {s['wr']:.0f}%, {s['tot']:+.0f}R, "
                             f"{s['exp']:+.2f}R/trade (±{s['noise']:.2f}), DD {s['dd']:.0f}R{yy}")
            lines.append("")
    if errors:
        lines += ["Errors:"] + errors[:6] + [""]
    lines += ["An idea counts only if BOTH years are positive and it is",
              "above +0.2R/trade. Spread will cost roughly 0.05-0.1R more."]
    text = "\n".join(lines)
    print(text)
    S.send(text[:3900])


if __name__ == "__main__":
    main()
