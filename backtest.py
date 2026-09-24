"""Backtest the break & retest scanner on recent history (no lookahead).

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


def backtest_pair(name, ticker):
    m15 = S.fetch(ticker, "15m", "59d")
    h1 = S.fetch(ticker, "1h", "59d")
    a_all = S.atr(m15)
    trades, lock_until = [], -1
    levels, last_n1 = [], -1
    for i in range(WARMUP, len(m15)):
        if i <= lock_until:
            continue
        a = a_all.iloc[i]
        if not np.isfinite(a) or a <= 0:
            continue
        t_end = m15.index[i] + timedelta(minutes=15)
        n1 = h1.index.searchsorted(t_end - timedelta(hours=1), side="right")
        if n1 < 60:
            continue
        if n1 != last_n1:
            levels = S.key_levels(h1.iloc[:n1], a)
            last_n1 = n1
        sig = S.find_signal(m15.iloc[max(0, i - WINDOW + 1): i + 1], levels)
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
            trades.append({"t": m15.index[i], "pair": name, "side": side, "R": res})
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
    return n, 100 * wins / n, cum, cum / n, dd, int(worst)


def main():
    all_trades, errors = [], []
    for name, ticker in S.PAIRS.items():
        try:
            t = backtest_pair(name, ticker)
            all_trades += t
            print(name, len(t), "trades")
        except Exception as e:
            errors.append(f"{name}: {e}")
            print("ERROR", name, e)
    all_trades.sort(key=lambda x: x["t"])

    lines = ["BACKTEST (last ~59 days, 15M, no spread)",
             f"Pairs: {len(S.PAIRS)} | Trades: {len(all_trades)}", ""]
    for rr in RRS:
        s = stats([t["R"][rr] for t in all_trades])
        if s is None:
            lines.append(f"{rr:.0f}R: no trades")
            continue
        n, wr, tot, exp, dd, streak = s
        lines.append(f"{rr:.0f}R: {n} trades, win {wr:.0f}%, {tot:+.1f}R total, "
                     f"{exp:+.2f}R/trade, max drawdown {dd:.1f}R, worst loss streak {streak}")
    lines += ["", f"By pair ({LOCK_RR:.0f}R):"]
    for name in S.PAIRS:
        rs = [t["R"][LOCK_RR] for t in all_trades if t["pair"] == name]
        s = stats(rs)
        if s:
            lines.append(f"{name}: {s[0]} trades, win {s[1]:.0f}%, {s[2]:+.1f}R")
    if errors:
        lines += ["", "Errors:"] + errors[:6]
    lines += ["", "Small sample. Past results do not guarantee future results."]
    text = "\n".join(lines)
    print(text)
    S.send(text[:3900])


if __name__ == "__main__":
    main()
  
