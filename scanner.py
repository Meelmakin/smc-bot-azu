"""Key level break & retest scanner -> Telegram alerts.

Levels (1H data): previous day H/L, previous week H/L, recent swing H/L.
Signal (15M data): close breaks a level -> price leaves -> returns to the level
-> rejection candle closes back in break direction on the LAST closed candle.
Data: Yahoo Finance (works from GitHub Actions; Binance is blocked there).
"""
import os
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import requests

PAIRS = {
    "GBPUSD": "GBPUSD=X",
    "EURUSD": "EURUSD=X",
    "USDJPY": "JPY=X",
    "XAUUSD": "GC=F",
    "US30": "YM=F",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
}

RR = 2.0             # take-profit as multiple of risk
LOOKBACK = 24        # 15M candles to look back for the break (~6 hours)
BREAK_BUF = 0.10     # close must clear level by this many ATR
RETEST_TOL = 0.25    # retest wick must come within this many ATR of level
MIN_LEAVE = 0.50     # price must move this many ATR away before retest
SL_BUF = 0.10        # stop buffer in ATR
STALE_MIN = 20       # ignore data if last closed candle is older than this


def env(*names):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def send(text):
    token = env("TELEGRAM_TOKEN", "TELEGRAM_BOT_TOKEN", "BOT_TOKEN")
    chat = env("CHAT_ID", "TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("Missing Telegram secrets (TELEGRAM_TOKEN / CHAT_ID)")
        print(text)
        return False
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data={"chat_id": chat, "text": text},
        timeout=20,
    )
    print("telegram:", r.status_code, r.text[:120])
    return r.ok


def fetch(ticker, interval, period):
    import yfinance as yf
    df = yf.Ticker(ticker).history(interval=interval, period=period, auto_adjust=False)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(n).mean()


def key_levels(h1, a):
    """Return list of (price, label)."""
    levels = []
    daily = h1.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    if len(daily) >= 2:
        levels += [(daily["high"].iloc[-2], "PDH"), (daily["low"].iloc[-2], "PDL")]
    weekly = h1.resample("W-SUN").agg({"high": "max", "low": "min"}).dropna()
    if len(weekly) >= 2:
        levels += [(weekly["high"].iloc[-2], "PWH"), (weekly["low"].iloc[-2], "PWL")]
    # swing fractals on 1H (5 left / 5 right), last ~5 days
    k = 5
    recent = h1.tail(24 * 5)
    hi, lo = recent["high"].values, recent["low"].values
    for i in range(k, len(recent) - k):
        if hi[i] == hi[i - k:i + k + 1].max():
            levels.append((hi[i], "Swing H"))
        if lo[i] == lo[i - k:i + k + 1].min():
            levels.append((lo[i], "Swing L"))
    # cluster: drop levels closer than 0.3 ATR to an already kept one
    kept = []
    for p, lab in sorted(levels, key=lambda x: x[0]):
        if not kept or abs(p - kept[-1][0]) > 0.3 * a:
            kept.append((p, lab))
    return kept


def find_signal(m15, levels):
    """m15 must contain only CLOSED candles. Returns dict or None."""
    a = atr(m15).iloc[-1]
    if not np.isfinite(a) or a <= 0:
        return None
    n = len(m15)
    last = n - 1
    o, h, l, c = (m15[x].values for x in ("open", "high", "low", "close"))
    start = max(1, n - LOOKBACK)

    for price, label in levels:
        for side in ("long", "short"):
            # find break candle
            b = None
            for j in range(start, last):
                if side == "long" and c[j - 1] <= price and c[j] > price + BREAK_BUF * a:
                    b = j
                elif side == "short" and c[j - 1] >= price and c[j] < price - BREAK_BUF * a:
                    b = j
                if b is not None:
                    break
            if b is None or last - b < 2:
                continue
            between = slice(b, last)  # candles after break up to (not incl.) last
            if side == "long":
                if (c[between] < price).any():          # broke back below -> failed
                    continue
                if h[between].max() < price + MIN_LEAVE * a:   # never left the level
                    continue
                touched = l[last] <= price + RETEST_TOL * a
                rejected = c[last] > price and c[last] > o[last]
                if touched and rejected:
                    sl = min(l[last], price) - SL_BUF * a
                    entry = c[last]
                    risk = entry - sl
                    if risk > 0:
                        return dict(side="BUY", label=label, level=price, entry=entry,
                                    sl=sl, tp=entry + RR * risk)
            else:
                if (c[between] > price).any():
                    continue
                if l[between].min() > price - MIN_LEAVE * a:
                    continue
                touched = h[last] >= price - RETEST_TOL * a
                rejected = c[last] < price and c[last] < o[last]
                if touched and rejected:
                    sl = max(h[last], price) + SL_BUF * a
                    entry = c[last]
                    risk = sl - entry
                    if risk > 0:
                        return dict(side="SELL", label=label, level=price, entry=entry,
                                    sl=sl, tp=entry - RR * risk)
    return None


def scan_pair(name, ticker):
    h1 = fetch(ticker, "1h", "1mo")
    m15 = fetch(ticker, "15m", "5d")
    now = datetime.now(timezone.utc)
    # drop the still-forming candle
    if len(m15) and m15.index[-1] + timedelta(minutes=15) > now:
        m15 = m15.iloc[:-1]
    if len(m15) < 40 or len(h1) < 60:
        raise RuntimeError("not enough data")
    last_end = m15.index[-1] + timedelta(minutes=15)
    if now - last_end > timedelta(minutes=STALE_MIN):
        return None, "market closed/stale"
    levels = key_levels(h1, atr(m15).iloc[-1])
    return find_signal(m15, levels), f"{len(levels)} levels"


def main():
    now = datetime.now(timezone.utc)
    manual = os.environ.get("MANUAL", "").lower() == "true"
    signals, errors, notes = [], [], []
    for name, ticker in PAIRS.items():
        try:
            sig, note = scan_pair(name, ticker)
            notes.append(f"{name}: {note}")
            if sig:
                signals.append((name, sig))
        except Exception as e:  # keep scanning other pairs
            errors.append(f"{name}: {e}")
            print("ERROR", name, e)

    for name, s in signals:
        dec = 3 if name in ("USDJPY", "XAUUSD", "US30", "BTCUSD", "ETHUSD") else 5
        send(
            f"{s['side']} {name}  (break & retest of {s['label']})\n"
            f"Level: {s['level']:.{dec}f}\n"
            f"Entry: {s['entry']:.{dec}f}\n"
            f"SL: {s['sl']:.{dec}f}\n"
            f"TP: {s['tp']:.{dec}f}  ({RR:.0f}R)"
        )

    if errors and len(errors) == len(PAIRS):
        send("Scanner ERROR: all pairs failed.\n" + "\n".join(errors[:4]))
    elif manual or (now.hour == 6 and now.minute < 15):
        send(
            f"Scanner alive {now:%Y-%m-%d %H:%M} UTC. Signals: {len(signals)}\n"
            + "\n".join(notes + errors)
        )
    print("done", len(signals), "signals,", len(errors), "errors")


if __name__ == "__main__":
    sys.exit(main())
      
