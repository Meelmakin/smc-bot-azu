"""Key level break & retest scanner -> Telegram alerts (v2, tighter rules).

Levels (1H data): previous day H/L and previous week H/L (swing levels optional).
Filters: London/NY session is now tagged (not hard-blocked), and trade only
with the 1H trend (EMA 50) if TREND_ON.
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
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "NZDUSD": "NZDUSD=X",
    "USDCHF": "CHF=X",
    "GBPJPY": "GBPJPY=X",
    "EURJPY": "EURJPY=X",
    "EURGBP": "EURGBP=X",
    "EURAUD": "EURAUD=X",
    "EURCAD": "EURCAD=X",
    "GBPAUD": "GBPAUD=X",
    "GBPCAD": "GBPCAD=X",
    "GBPCHF": "GBPCHF=X",
    "AUDJPY": "AUDJPY=X",
    "CADJPY": "CADJPY=X",
    "NZDJPY": "NZDJPY=X",
    "AUDCAD": "AUDCAD=X",
    "NAS100": "NQ=F",
    "SPX500": "ES=F",
    "XAGUSD": "SI=F",
    "USOIL": "CL=F",
    "SOLUSD": "SOL-USD",
}

RR = 3.0             # take-profit as multiple of risk
LOOKBACK = 24        # 15M candles to look back for the break (~6 hours)
BREAK_BUF = 0.10     # close must clear level by this many ATR
RETEST_TOL = 0.25    # retest wick must come within this many ATR of level
MIN_LEAVE = 0.50     # price must move this many ATR away before retest
SL_BUF = 0.10        # stop buffer in ATR
STALE_MIN = 20       # ignore data if last closed candle is older than this

USE_SWING = False    # False = only previous day/week high/low
SESSION = (7, 20)    # UTC hours: start (inclusive), end (exclusive) -- used for tagging only
TREND_ON = True      # only trade with the 1H trend
EMA_LEN = 50         # 1H EMA length for the trend filter
MIN_SCORE = 0        # only alert if quality score >= this (0 = send all; 5 = A only)
MAX_DRIFT_R = 0.35    # skip the alert if live price has already moved this many
                      # multiples of risk (R) away from the computed entry


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


def in_session(hour):
    return SESSION[0] <= hour < SESSION[1]


def key_levels(h1, a, swings=None):
    """Return list of (price, label)."""
    if swings is None:
        swings = USE_SWING
    h1 = h1.tail(24 * 16)
    levels = []
    daily = h1.resample("1D").agg({"high": "max", "low": "min"}).dropna()
    if len(daily) >= 2:
        levels += [(daily["high"].iloc[-2], "PDH"), (daily["low"].iloc[-2], "PDL")]
    weekly = h1.resample("W-SUN").agg({"high": "max", "low": "min"}).dropna()
    if len(weekly) >= 2:
        levels += [(weekly["high"].iloc[-2], "PWH"), (weekly["low"].iloc[-2], "PWL")]
    if swings:
        k = 5
        recent = h1.tail(24 * 5)
        hi, lo = recent["high"].values, recent["low"].values
        for i in range(k, len(recent) - k):
            if hi[i] == hi[i - k:i + k + 1].max():
                levels.append((hi[i], "Swing H"))
            if lo[i] == lo[i - k:i + k + 1].min():
                levels.append((lo[i], "Swing L"))
    kept = []
    for p, lab in sorted(levels, key=lambda x: x[0]):
        if not kept or abs(p - kept[-1][0]) > 0.3 * a:
            kept.append((p, lab))
    return kept


def quality(side, label, price, entry, sl, tp, levels, a, o, h, l, c, last, b,
            hour, trend_up):
    """Score 0-6. Returns (score, grade, reasons)."""
    pts, why = 0, []
    buy = side == "BUY"
    if label in ("PWH", "PWL"):
        pts += 1
        why.append("weekly level")
    if trend_up is not None and trend_up == buy:
        pts += 1
        why.append("with 1H trend")
    if 12 <= hour < 16:
        pts += 1
        why.append("London/NY overlap")
    rng = h[last] - l[last]
    if rng > 0:
        wick = (min(o[last], c[last]) - l[last]) if buy else (h[last] - max(o[last], c[last]))
        if wick / rng >= 0.4:
            pts += 1
            why.append("clean rejection wick")
    leave = (h[b:last].max() - price) if buy else (price - l[b:last].min())
    if leave >= 1.0 * a:
        pts += 1
        why.append("strong move after break")
    risk = abs(entry - sl)
    ahead = [abs(p - entry) for p, _ in levels if (p > entry if buy else p < entry)]
    if not ahead or min(ahead) >= 2 * risk:
        pts += 1
        why.append("2R+ room to next level")
    grade = "A" if pts >= 5 else ("B" if pts >= 3 else "C")
    return pts, grade, why


def find_signal(m15, levels, sides=("long", "short"), a=None, trend_up=None):
    """m15 must contain only CLOSED candles. Returns dict or None."""
    if a is None:
        a = atr(m15).iloc[-1]
    if not np.isfinite(a) or a <= 0:
        return None
    n = len(m15)
    last = n - 1
    o, h, l, c = (m15[x].values for x in ("open", "high", "low", "close"))
    start = max(1, n - LOOKBACK)
    tol = RETEST_TOL * a
    hour = m15.index[-1].hour

    for price, label in levels:
        for side in sides:
            # cheap check first: does the LAST candle retest and reject the level?
            if side == "long":
                if not (l[last] <= price + tol and c[last] > price and c[last] > o[last]):
                    continue
            else:
                if not (h[last] >= price - tol and c[last] < price and c[last] < o[last]):
                    continue
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
            between = slice(b, last)
            if side == "long":
                if (c[between] < price).any():
                    continue
                if h[between].max() < price + MIN_LEAVE * a:
                    continue
                sl = min(l[last], price) - SL_BUF * a
                entry = price  # limit order resting at the level, not the
                                # rejection-candle close (better fill, tighter risk)
                risk = entry - sl
                if risk > 0:
                    tp = entry + RR * risk
                    sc, gr, why = quality("BUY", label, price, entry, sl, tp, levels,
                                          a, o, h, l, c, last, b, hour, trend_up)
                    return dict(side="BUY", label=label, level=price, entry=entry,
                                sl=sl, tp=tp, score=sc, grade=gr, why=why)
            else:
                if (c[between] > price).any():
                    continue
                if l[between].min() > price - MIN_LEAVE * a:
                    continue
                sl = max(h[last], price) + SL_BUF * a
                entry = price  # limit order resting at the level
                risk = sl - entry
                if risk > 0:
                    tp = entry - RR * risk
                    sc, gr, why = quality("SELL", label, price, entry, sl, tp, levels,
                                          a, o, h, l, c, last, b, hour, trend_up)
                    return dict(side="SELL", label=label, level=price, entry=entry,
                                sl=sl, tp=tp, score=sc, grade=gr, why=why)
    return None


def live_price(ticker):
    """Best-effort current price, independent of the 15m candle series used
    for the signal. Falls back through a couple of methods since yfinance's
    fast_info isn't always populated for futures tickers on GitHub Actions."""
    import yfinance as yf
    t = yf.Ticker(ticker)
    try:
        p = t.fast_info.get("last_price")
        if p:
            return float(p)
    except Exception:
        pass
    try:
        df = t.history(interval="1m", period="1d", auto_adjust=False)
        if len(df):
            return float(df["close"].iloc[-1])
    except Exception:
        pass
    return None


def scan_pair(name, ticker):
    h1 = fetch(ticker, "1h", "1mo")
    m15 = fetch(ticker, "15m", "5d")
    now = datetime.now(timezone.utc)
    if len(m15) and m15.index[-1] + timedelta(minutes=15) > now:
        m15 = m15.iloc[:-1]
    if len(h1) and h1.index[-1] + timedelta(hours=1) > now:
        h1 = h1.iloc[:-1]
    if len(m15) < 40 or len(h1) < 60:
        raise RuntimeError("not enough data")
    last_end = m15.index[-1] + timedelta(minutes=15)
    if now - last_end > timedelta(minutes=STALE_MIN):
        return None, "market closed/stale"

    session_tag = "in-session" if in_session(m15.index[-1].hour) else "off-session"
    a = atr(m15).iloc[-1]
    levels = key_levels(h1, a)
    ema = h1["close"].ewm(span=EMA_LEN, adjust=False).mean()
    trend_up = bool(h1["close"].iloc[-1] > ema.iloc[-1])
    sides = ("long", "short")
    if TREND_ON:
        sides = ("long",) if trend_up else ("short",)
    trend = "up" if trend_up else "down"
    sig = find_signal(m15, levels, sides, a, trend_up)
    if sig:
        sig["session"] = session_tag
    return sig, f"{len(levels)} levels, 1H trend {trend}, {session_tag}"


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
        except Exception as e:
            errors.append(f"{name}: {e}")
            print("ERROR", name, e)

    for name, s in signals:
        if s["score"] < MIN_SCORE:
            continue

        risk = abs(s["entry"] - s["sl"])
        live = live_price(PAIRS[name])
        if live is not None and risk > 0:
            drift_r = abs(live - s["entry"]) / risk
            if drift_r > MAX_DRIFT_R:
                print(f"SKIP {name}: price drifted {drift_r:.2f}R from entry "
                      f"(entry {s['entry']:.5f}, live {live:.5f}) - no longer executable")
                continue
        elif live is None:
            print(f"WARN {name}: could not confirm live price, sending anyway")

        dec = 3 if name.endswith("JPY") or name in (
            "XAUUSD", "XAGUSD", "US30", "NAS100", "SPX500", "USOIL",
            "BTCUSD", "ETHUSD", "SOLUSD") else 5

        buy = s["side"] == "BUY"
        entry_limit = s["entry"]          # order resting at the level
        entry_market = live if live is not None else entry_limit  # enter now instead

        # SL stays anchored to structure; TP recalculated per entry so each
        # option still targets the same reward multiple off its own risk.
        risk_market = abs(entry_market - s["sl"])
        tp_market = (entry_market + RR * risk_market if buy
                     else entry_market - RR * risk_market) if risk_market > 0 else s["tp"]

        market_block = (
            f"OR market now: {entry_market:.{dec}f}  |  "
            f"SL {s['sl']:.{dec}f}  |  TP {tp_market:.{dec}f}\n"
            if live is not None and abs(entry_market - entry_limit) > 1e-9 else ""
        )

        send(
            f"{s['side']} {name}  (break & retest of {s['label']})\n"
            f"Session: {s.get('session', 'n/a')}\n"
            f"Quality: {s['grade']} ({s['score']}/6){'  ⭐' if s['grade'] == 'A' else ''}\n"
            f"Why: {', '.join(s['why']) or 'basic setup only'}\n"
            f"Entry (limit @ level): {entry_limit:.{dec}f}  |  "
            f"SL {s['sl']:.{dec}f}  |  TP {s['tp']:.{dec}f}  ({RR:.0f}R)\n"
            f"{market_block}"
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
                
