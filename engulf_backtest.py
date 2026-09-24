"""Backtest of the manual method: daily bias -> 4H engulf -> retest -> 1H engulf.

Rules as coded (change the settings below if you trade it differently):
1. Daily bias: previous daily candle rejected a side (wick >= 40% of its range and
   closed in the other half). Rejection from lows = bullish, from highs = bearish.
2. 4H direction (line chart idea): the 4H engulf candle closes above the 4H EMA50
   for buys, below it for sells.
3. 4H engulf: bullish/bearish engulfing candle in the bias direction.
4. Retest: price trades back to the ENGULFED candle (its open, the near edge of its
   body). Setup dies if a 1H candle closes beyond the engulfed candle's far end,
   or after 3 days.
5. Entry: 1H engulfing candle in the same direction, at or after the retest.
   Entry at that candle's close.
6. Stop: beyond the lowest low (buy) / highest high (sell) from the retest to the
   entry candle, plus a small buffer. Take profit: 2R and 3R tested.

Uses ~2 years of 1H data from Yahoo. 4H and daily candles are built from 1H (UTC).
No spread/slippage. SL and TP in the same candle counts as a loss.
"""
from datetime import timedelta

import numpy as np

import scanner as S

RRS = (2.0, 3.0)
LOCK_RR = 3.0
MAX_WAIT = 72        # 1H bars a 4H setup stays valid (3 days)
MAX_BARS = 120       # trade time limit in 1H bars (5 days)
SL_BUF = 0.10        # stop buffer in ATR
MIN_RISK = 0.30      # minimum stop distance in ATR
EMA4 = 50            # 4H EMA length for the direction filter
WICK = 0.40          # daily rejection wick as a share of the candle range

VARIANTS = {
    "A Your method (all rules)": dict(daily=True, trend=True, retest=True),
    "B Without daily bias": dict(daily=False, trend=True, retest=True),
    "C Without waiting for retest": dict(daily=True, trend=True, retest=False),
    "D Engulf pattern only": dict(daily=False, trend=False, retest=True),
}


def bull_engulf(o1, c1, o0, c0):
    return c1 < o1 and c0 > o0 and c0 >= o1 and o0 <= c1


def bear_engulf(o1, c1, o0, c0):
    return c1 > o1 and c0 < o0 and c0 <= o1 and o0 >= c1


def daily_bias(o, h, l, c):
    rng = h - l
    if rng <= 0:
        return 0
    lw = (min(o, c) - l) / rng
    uw = (h - max(o, c)) / rng
    mid = (h + l) / 2
    if lw >= WICK and c >= mid:
        return 1
    if uw >= WICK and c <= mid:
        return -1
    return 0


def outcome(hi, lo, cl, i, d, entry, sl, rr):
    risk = abs(entry - sl)
    tp = entry + rr * risk if d == 1 else entry - rr * risk
    end = min(len(hi), i + 1 + MAX_BARS)
    for j in range(i + 1, end):
        if d == 1:
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
    r = (cl[j] - entry) / risk if d == 1 else (entry - cl[j]) / risk
    return float(r), j


AGG = {"open": "first", "high": "max", "low": "min", "close": "last"}


def prepare(h1):
    h4 = h1.resample("4h").agg(AGG).dropna()
    daily = h1.resample("1D").agg(AGG).dropna()
    return dict(
        h1=h1, h4=h4, daily=daily,
        o=h1["open"].values, h=h1["high"].values, l=h1["low"].values, c=h1["close"].values,
        a=S.atr(h1).values,
        h4pos={ts: k for k, ts in enumerate(h4.index)},
        ema4=h4["close"].ewm(span=EMA4, adjust=False).mean().values,
    )


def run(name, P, cfg):
    h1, h4, daily = P["h1"], P["h4"], P["daily"]
    o, h, l, c, a = P["o"], P["h"], P["l"], P["c"], P["a"]
    h4o, h4h, h4l, h4c = (h4[x].values for x in ("open", "high", "low", "close"))
    do, dh, dl, dc = (daily[x].values for x in ("open", "high", "low", "close"))
    idx = h1.index
    trades, setup, lock_until = [], None, -1

    for i in range(60, len(h1)):
        # --- manage an existing setup (bars after the 4H close) ---
        if setup is not None and i > setup["born"]:
            d = setup["d"]
            if (i - setup["born"] > MAX_WAIT
                    or (d == 1 and c[i] < setup["inv"])
                    or (d == -1 and c[i] > setup["inv"])):
                setup = None
            else:
                if not setup["retested"]:
                    if (d == 1 and l[i] <= setup["zone"]) or (d == -1 and h[i] >= setup["zone"]):
                        setup["retested"], setup["ridx"] = True, i
                if setup["retested"]:
                    eng = (bull_engulf(o[i - 1], c[i - 1], o[i], c[i]) if d == 1
                           else bear_engulf(o[i - 1], c[i - 1], o[i], c[i]))
                    if eng and np.isfinite(a[i]) and a[i] > 0:
                        st = setup["ridx"] if setup["ridx"] is not None else i - 1
                        entry = c[i]
                        if d == 1:
                            sl = min(l[st:i + 1]) - SL_BUF * a[i]
                            sl = min(sl, entry - MIN_RISK * a[i])
                        else:
                            sl = max(h[st:i + 1]) + SL_BUF * a[i]
                            sl = max(sl, entry + MIN_RISK * a[i])
                        res = {}
                        for rr in RRS:
                            r, j = outcome(h, l, c, i, d, entry, sl, rr)
                            if r is None:
                                res = None
                                break
                            res[rr] = r
                            if rr == LOCK_RR:
                                lock_until = j
                        if res:
                            trades.append({"t": idx[i], "pair": name, "R": res})
                        setup = None

        # --- does a 4H candle close at the end of this 1H bar? ---
        t_end = idx[i] + timedelta(hours=1)
        if t_end.hour % 4 == 0 and t_end.minute == 0 and i > lock_until:
            k = P["h4pos"].get(t_end - timedelta(hours=4))
            if k is None or k < 1:
                continue
            o1, c1, o0, c0 = h4o[k - 1], h4c[k - 1], h4o[k], h4c[k]
            d = 1 if bull_engulf(o1, c1, o0, c0) else (-1 if bear_engulf(o1, c1, o0, c0) else 0)
            if not d:
                continue
            if cfg["trend"]:
                ema = P["ema4"][k]
                if not ((c0 > ema) if d == 1 else (c0 < ema)):
                    continue
            if cfg["daily"]:
                dp = daily.index.searchsorted(t_end.normalize(), side="left") - 1
                if dp < 0 or daily_bias(do[dp], dh[dp], dl[dp], dc[dp]) != d:
                    continue
            inv = h4l[k - 1] if d == 1 else h4h[k - 1]
            setup = dict(d=d, zone=o1, inv=inv, born=i,
                         retested=not cfg["retest"], ridx=None)
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
            for v, cfg in VARIANTS.items():
                results[v] += run(name, P, cfg)
            print(name, "done", len(h1), "bars")
        except Exception as e:
            errors.append(f"{name}: {e}")
            print("ERROR", name, e)
    for v in results:
        results[v].sort(key=lambda x: x["t"])

    lines = ["ENGULF METHOD BACKTEST (~2 years, 1H entries, no spread)",
             f"Pairs: {len(S.PAIRS)}", ""]
    for v, tr in results.items():
        if not tr:
            lines += [f"{v}: no trades", ""]
            continue
        days = max(1, (tr[-1]["t"] - tr[0]["t"]).days)
        lines.append(f"{v}: {len(tr)} trades (~{len(tr) / days:.1f}/day)")
        for rr in RRS:
            s = stats([t["R"][rr] for t in tr])
            lines.append(f"  {rr:.0f}R: win {s['wr']:.0f}%, {s['tot']:+.0f}R, "
                         f"{s['exp']:+.2f}R/trade (±{s['noise']:.2f}), "
                         f"DD {s['dd']:.0f}R, streak {s['streak']}")
        mid = tr[len(tr) // 2]["t"]
        f1 = stats([t["R"][LOCK_RR] for t in tr if t["t"] < mid])
        f2 = stats([t["R"][LOCK_RR] for t in tr if t["t"] >= mid])
        if f1 and f2:
            lines.append(f"  3R first half {f1['exp']:+.2f}R/trade, second half {f2['exp']:+.2f}R/trade")
        lines.append("")
    a = results.get("A Your method (all rules)", [])
    if a:
        lines.append("By pair, your method (3R):")
        for name in S.PAIRS:
            s = stats([t["R"][LOCK_RR] for t in a if t["pair"] == name])
            if s:
                lines.append(f"{name}: {s['n']} trades, win {s['wr']:.0f}%, {s['tot']:+.1f}R")
    if errors:
        lines += ["", "Errors:"] + errors[:6]
    lines += ["", "(±) = rough noise margin. Treat under +0.2R/trade as noise.",
              "If both halves are positive, the result is more trustworthy."]
    text = "\n".join(lines)
    print(text)
    S.send(text[:3900])


if __name__ == "__main__":
    main()
