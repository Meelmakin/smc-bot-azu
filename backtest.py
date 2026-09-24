"""Backtest several rule sets of the break & retest scanner (no lookahead).

Imports scanner.py, so it uses your PAIRS list and strategy settings.
Yahoo only serves ~60 days of 15M data, so this is a SMALL sample.
Assumes: entry at the signal candle close, no spread/slippage,
SL and TP hit in the same candle counts as a LOSS (conservative),
one trade at a time per pair (locked until the 3R trade resolves),
trades still open after 24h are closed at market.
"""
from datetime import timedelta

import numpy as np

import scanner as S

RRS = (2.0, 3.0, 4.0)
LOCK_RR = 3.0
MAX_BARS = 96          # 24h of 15M candles
WARMUP = 100           # candles before first check
WINDOW = 200           # candles passed to the signal finder

VARIANTS = {
    "A Old rules (all levels)": dict(swing=True, session=False, trend=False),
    "B Big levels only": dict(swing=False, session=False, trend=False),
    "C Big levels + London/NY": dict(swing=False, session=True, trend=False),
    "D Big + session + 1H trend": dict(swing=False, session=True, trend=True),
    "E All levels + session + trend": dict(swing=True, session=True, trend=True),
}


def outcome(m15, i, side, entry, sl, rr):
    risk = abs(entry - sl)
    tp = entry + rr * risk if side == "BUY" else entry - rr * risk
    hi, lo, cl = m15["high"].values, m15["low"].values, m15["close"].values
    end = min(len(m15), i + 1 + MAX_BARS)
    for j in range(i + 1, end):
        if side == "BUY":
            if lo[j] <= sl:
                return -1.0, j
            if hi[j] >= tp:
                return rr, j
        else:
            if hi[j] >= sl:
                return -1.0, j
            if lo[j] <= tp:
                return rr, j
    j = end - 1
    if j <= i:
        return None, i
    r = (cl[j] - entry) / risk if side == "BUY" else (entry - cl[j]) / risk
    return float(r), j


def run_variant(name, m15, h1, a_arr, n1_arr, above, cache, cfg):
    hours = m15.index.hour
    trades, lock_until = [], -1
    for i in range(WARMUP, len(m15)):
        if i <= lock_until:
            continue
        if cfg["session"] and not S.in_session(hours[i]):
            continue
        a = a_arr[i]
        n1 = n1_arr[i]
        if not np.isfinite(a) or a <= 0 or n1 < 60:
            continue
        key = (cfg["swing"], n1)
        if key not in cache:
            cache[key] = S.key_levels(h1.iloc[:n1], a, swings=cfg["swing"])
        levels = cache[key]
        sides = ("long", "short")
        if cfg["trend"]:
            sides = ("long",) if above[n1 - 1] else ("short",)
        sig = S.find_signal(m15.iloc[max(0, i - WINDOW + 1): i + 1], levels, sides, a,
                            bool(above[n1 - 1]))
        if not sig:
            continue
        side, entry, sl = sig["side"], float(sig["entry"]), float(sig["sl"])
        res = {}
        for rr in RRS:
            r, j = outcome(m15, i, side, entry, sl, rr)
            if r is None:
                res = None
                break
            res[rr] = r
            if rr == LOCK_RR:
                lock_until = j
        if res:
            trades.append({"t": m15.index[i], "pair": name, "R": res,
                           "score": sig["score"], "grade": sig["grade"]})
    return trades


def backtest_pair(name, ticker, results):
    m15 = S.fetch(ticker, "15m", "59d")
    h1 = S.fetch(ticker, "1h", "59d")
    a_arr = S.atr(m15).values
    n1_arr = h1.index.searchsorted(
        m15.index + timedelta(minutes=15) - timedelta(hours=1), side="right")
    ema = h1["close"].ewm(span=S.EMA_LEN, adjust=False).mean()
    above = (h1["close"] > ema).values
    cache = {}
    for vname, cfg in VARIANTS.items():
        results[vname] += run_variant(name, m15, h1, a_arr, n1_arr, above, cache, cfg)


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
    noise = 2 * float(np.std(rs)) / np.sqrt(n)
    return dict(n=n, wr=100 * wins / n, tot=cum, exp=cum / n, dd=dd,
                streak=int(worst), noise=noise)


def main():
    results = {v: [] for v in VARIANTS}
    errors = []
    for name, ticker in S.PAIRS.items():
        try:
            backtest_pair(name, ticker, results)
            print(name, "done")
        except Exception as e:
            errors.append(f"{name}: {e}")
            print("ERROR", name, e)
    for v in results:
        results[v].sort(key=lambda x: x["t"])

    lines = ["BACKTEST v2 (last ~59 days, 15M, no spread)",
             f"Pairs: {len(S.PAIRS)}", ""]
    best, best_exp = None, -9
    for v, tr in results.items():
        if not tr:
            lines.append(f"{v}: no trades")
            continue
        days = max(1, (tr[-1]["t"] - tr[0]["t"]).days)
        lines.append(f"{v}: {len(tr)} trades (~{len(tr) / days:.1f}/day)")
        for rr in RRS:
            s = stats([t["R"][rr] for t in tr])
            lines.append(f"  {rr:.0f}R: win {s['wr']:.0f}%, {s['tot']:+.0f}R, "
                         f"{s['exp']:+.2f}R/trade (±{s['noise']:.2f}), "
                         f"DD {s['dd']:.0f}R, streak {s['streak']}")
            if rr == LOCK_RR and s["exp"] > best_exp:
                best, best_exp = v, s["exp"]
        lines.append("")
    if best:
        lines.append(f"Best at 3R: {best}")
        lines.append("By pair (3R):")
        for name in S.PAIRS:
            rs = [t["R"][LOCK_RR] for t in results[best] if t["pair"] == name]
            s = stats(rs)
            if s:
                lines.append(f"{name}: {s['n']} trades, win {s['wr']:.0f}%, {s['tot']:+.1f}R")
    lines += ["", "QUALITY GRADE CHECK (3R)."]
    lines += ["Does a higher grade really do better?"]
    for v in ("B Big levels only", "D Big + session + 1H trend"):
        tr = results.get(v, [])
        lines.append(v + ":")
        for g in ("A", "B", "C"):
            rs = [t["R"][LOCK_RR] for t in tr if t["grade"] == g]
            s = stats(rs)
            if s:
                lines.append(f"  {g}: {s['n']} trades, win {s['wr']:.0f}%, "
                             f"{s['exp']:+.2f}R/trade (±{s['noise']:.2f})")
            else:
                lines.append(f"  {g}: no trades")
    if errors:
        lines += ["", "Errors:"] + errors[:6]
    lines += ["", "(±) = rough noise margin. If R/trade is inside it, the result",
              "is not distinguishable from luck. Small sample."]
    text = "\n".join(lines)
    print(text)
    S.send(text[:3900])


if __name__ == "__main__":
    main()
    
