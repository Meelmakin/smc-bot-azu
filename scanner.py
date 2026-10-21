"""Key level break & retest scanner -> Telegram alerts (v3, Stage 1).

Strategy rules are UNCHANGED from v2. Stage 1 fixes delivery and reliability:
  * Telegram token can no longer leak into logs or tracebacks.
  * Data layer: validated candles, retries, pluggable source (DATA_SOURCE).
  * Timing diagnostics: how old the data is and how many candles behind.
  * Scans the last SCAN_BACK closed candles, so a late/skipped run isn't lost.
  * One alert per setup (state file) instead of re-alerting every run.
  * Every alert carries a status, an expiry and an invalidation rule.
    Fills are never assumed: this code cannot know if you were filled.

Env: TELEGRAM_TOKEN + CHAT_ID (secrets), STATE_PATH, DATA_SOURCE, DRY_RUN=1.
"""
import json
import os
import re
import sys
import time
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

# ---- strategy settings (unchanged from v2) ---------------------------------
RR = 3.0             # take-profit as multiple of risk
LOOKBACK = 24        # 15M candles to look back for the break (~6 hours)
BREAK_BUF = 0.10     # close must clear level by this many ATR
RETEST_TOL = 0.25    # retest wick must come within this many ATR of level
MIN_LEAVE = 0.50     # price must move this many ATR away before retest
SL_BUF = 0.10        # stop buffer in ATR
STALE_MIN = 20       # ignore data if last closed candle is older than this

USE_SWING = False    # False = only previous day/week high/low
SESSION = (7, 20)    # UTC hours: start (inclusive), end (exclusive) -- tagging only
TREND_ON = True      # only trade with the 1H trend
EMA_LEN = 50         # 1H EMA length for the trend filter
MIN_SCORE = 0        # only alert if quality score >= this (0 = send all; 5 = A only)

MIN_RISK_ATR = 0.30        # stop must be at least this many ATR(15m) from entry
COST_PCT = 0.0002          # rough round-trip spread+slippage as a fraction of price
MIN_RISK_COST_MULT = 3.0   # stop must be at least this many times that cost

ENTRY_AT_LEVEL = False   # False = enter at the signal candle's close (what backtest.py
                         #         measures); True = limit order at the level (never backtested)

# ---- Stage 1 settings (starting values, NOT backtested: tune them) ---------
MAX_DRIFT_R = 0.35       # live price within this many R of entry = entry still valid
MIN_RR = 2.0             # live R:R must stay >= this for an ACTIVE entry
SCAN_BACK = 2            # check the last N closed 15m candles (covers a late run)
EXPIRY_CANDLES = 8       # setup expires this many 15m candles after the signal closes
SEND_STATUSES = {"ACTIVE", "PENDING", "UNVERIFIED"}   # add "EXPIRED"/"INVALIDATED" to see them
STATE_PATH = os.environ.get("STATE_PATH", "state/signals.json")
DATA_SOURCE = os.environ.get("DATA_SOURCE", "yahoo")
DRY_RUN = os.environ.get("DRY_RUN") == "1"
FETCH_RETRIES = 2
STATE_KEEP_DAYS = 7
ERROR_ALERT_COOLDOWN_H = 6

SECRET_ENV = ("TELEGRAM_TOKEN", "TELEGRAM_BOT_TOKEN", "BOT_TOKEN",
              "CHAT_ID", "TELEGRAM_CHAT_ID")
M15 = timedelta(minutes=15)


# ---------------------------------------------------------------------------
# logging / secrets
# ---------------------------------------------------------------------------
def redact(x):
    """Remove secrets from anything that might be printed."""
    s = str(x)
    for n in SECRET_ENV:
        v = os.environ.get(n)
        if v:
            s = s.replace(v, "***")
    return re.sub(r"bot\d+:[A-Za-z0-9_-]+", "bot***", s)


def log(*parts):
    print(" ".join(redact(p) for p in parts), flush=True)


def env(*names):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def send(text):
    """Send to Telegram. Returns True only if Telegram accepted it."""
    token = env("TELEGRAM_TOKEN", "TELEGRAM_BOT_TOKEN", "BOT_TOKEN")
    chat = env("CHAT_ID", "TELEGRAM_CHAT_ID")
    if DRY_RUN:
        log("[DRY_RUN] would send:\n" + text)
        return False
    if not token or not chat:
        log("Missing Telegram secrets (TELEGRAM_TOKEN / CHAT_ID)")
        log(text)
        return False
    for attempt in range(2):
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat, "text": text},
                timeout=20,
            )
        except Exception as e:  # exception text can contain the URL (= the token)
            log("telegram: request failed:", type(e).__name__, e)
        else:
            if r.ok:
                log("telegram: ok")
                return True
            log("telegram: HTTP", r.status_code, r.text[:120])
        if attempt == 0:
            time.sleep(2)
    return False


# ---------------------------------------------------------------------------
# data layer
# ---------------------------------------------------------------------------
def fetch_yahoo(ticker, interval, period):
    import yfinance as yf
    df, err = None, "unknown"
    for attempt in range(FETCH_RETRIES + 1):
        try:
            df = yf.Ticker(ticker).history(interval=interval, period=period,
                                           auto_adjust=False)
            if len(df):
                break
            err = "empty response"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        df = None
        if attempt < FETCH_RETRIES:
            time.sleep(1.5 * (attempt + 1))
    if df is None:
        raise RuntimeError(f"yahoo {ticker} {interval} failed: {redact(err)}")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df


# Plug other sources in here (broker feed, etc.): fn(ticker, interval, period) -> OHLC df, UTC index
SOURCES = {"yahoo": fetch_yahoo}


def clean_candles(df):
    """Sort, de-duplicate, drop impossible bars. Raises if nothing is left."""
    if df is None or len(df) == 0:
        raise RuntimeError("no candles")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    bad = (df["high"] < df["low"]) | (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    if bad.any():
        log(f"data: dropped {int(bad.sum())} invalid bar(s)")
        df = df[~bad]
    if len(df) == 0:
        raise RuntimeError("no valid candles")
    return df


def fetch(ticker, interval, period):
    src = SOURCES.get(DATA_SOURCE)
    if src is None:
        raise RuntimeError(f"unknown DATA_SOURCE {DATA_SOURCE!r}")
    return clean_candles(src(ticker, interval, period))


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(n).mean()


def in_session(hour):
    return SESSION[0] <= hour < SESSION[1]


def min_risk(entry, a):
    """Smallest stop distance worth trading."""
    return max(MIN_RISK_ATR * a, MIN_RISK_COST_MULT * COST_PCT * entry)


def floor15(dt):
    return dt.replace(minute=dt.minute - dt.minute % 15, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# strategy (logic unchanged from v2; find_signal now also returns break_time)
# ---------------------------------------------------------------------------
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
                entry = price if ENTRY_AT_LEVEL else float(c[last])
                risk = entry - sl
                if risk > 0 and risk >= min_risk(entry, a):
                    tp = entry + RR * risk
                    sc, gr, why = quality("BUY", label, price, entry, sl, tp, levels,
                                          a, o, h, l, c, last, b, hour, trend_up)
                    return dict(side="BUY", label=label, level=price, entry=entry,
                                sl=sl, tp=tp, score=sc, grade=gr, why=why,
                                break_time=m15.index[b].isoformat())
            else:
                if (c[between] > price).any():
                    continue
                if l[between].min() > price - MIN_LEAVE * a:
                    continue
                sl = max(h[last], price) + SL_BUF * a
                entry = price if ENTRY_AT_LEVEL else float(c[last])
                risk = sl - entry
                if risk > 0 and risk >= min_risk(entry, a):
                    tp = entry - RR * risk
                    sc, gr, why = quality("SELL", label, price, entry, sl, tp, levels,
                                          a, o, h, l, c, last, b, hour, trend_up)
                    return dict(side="SELL", label=label, level=price, entry=entry,
                                sl=sl, tp=tp, score=sc, grade=gr, why=why,
                                break_time=m15.index[b].isoformat())
    return None


# ---------------------------------------------------------------------------
# live price, status, alert text
# ---------------------------------------------------------------------------
def live_price(ticker):
    """Best-effort current price, independent of the 15m candle series."""
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


def rr_at(s, price):
    """Reward:risk if entering at `price` with the setup's SL/TP. None if invalid."""
    buy = s["side"] == "BUY"
    risk = price - s["sl"] if buy else s["sl"] - price
    reward = s["tp"] - price if buy else price - s["tp"]
    if risk <= 0 or reward <= 0:
        return None
    return reward / risk


def classify(s, live, now):
    """(status, reason). There is deliberately no 'filled' status: nothing in
    this bot can see your broker, so a fill can only be confirmed by you."""
    buy = s["side"] == "BUY"
    if now >= s["expires_at"]:
        return "EXPIRED", f"past expiry ({s['expires_at']:%H:%M} UTC)"
    if live is None:
        return "UNVERIFIED", "live price unavailable, check price before entering"
    if (live <= s["sl"]) if buy else (live >= s["sl"]):
        return "INVALIDATED", "price already at/through the stop level"
    if (live >= s["tp"]) if buy else (live <= s["tp"]):
        return "INVALIDATED", "target already reached before entry"
    risk = abs(s["entry"] - s["sl"])
    fav = ((live - s["entry"]) if buy else (s["entry"] - live)) / risk
    if fav < -MAX_DRIFT_R:
        return "INVALIDATED", f"price {abs(fav):.2f}R against entry (retest failing)"
    rr = rr_at(s, live)
    if fav > MAX_DRIFT_R or rr is None or rr < MIN_RR:
        return "PENDING", f"price {fav:+.2f}R from entry, wait for pullback, do not chase"
    return "ACTIVE", f"live price within {MAX_DRIFT_R}R of entry"


def decimals(name):
    return 3 if name.endswith("JPY") or name in (
        "XAUUSD", "XAGUSD", "US30", "NAS100", "SPX500", "USOIL",
        "BTCUSD", "ETHUSD", "SOLUSD") else 5


ICON = {"ACTIVE": "🟢", "PENDING": "🟡", "UNVERIFIED": "⚪",
        "EXPIRED": "⏱", "INVALIDATED": "🔴"}
STATUS_TEXT = {
    "ACTIVE": "entry executable now (fill NOT confirmed)",
    "PENDING": "entry not valid right now",
    "UNVERIFIED": "could not check live price",
    "EXPIRED": "do not enter",
    "INVALIDATED": "do not enter",
}


def format_alert(name, s, status, reason, live, now):
    dec = decimals(name)
    age = int((now - s["signal_time"]).total_seconds() // 60)
    entry_label = "limit @ level" if ENTRY_AT_LEVEL else "signal close"
    rr_live = rr_at(s, live) if live is not None else None
    rr_txt = f" (1:{rr_live:.1f} at live)" if rr_live else ""
    star = "  ⭐" if s["grade"] == "A" else ""
    return "\n".join([
        f"{ICON[status]} {s['side']} {name}  (break & retest of {s['label']})",
        f"Status: {status} - {STATUS_TEXT[status]}",
        f"Reason: {reason}",
        "Timeframe: 15m entry, 1H trend and levels",
        f"Signal candle closed: {s['signal_time']:%Y-%m-%d %H:%M} UTC ({age} min ago)",
        f"Valid until: {s['expires_at']:%H:%M} UTC",
        f"Session: {s.get('session', 'n/a')}",
        f"Quality: {s['grade']} ({s['score']}/6){star}",
        f"Why: {', '.join(s['why']) or 'basic setup only'}",
        f"Entry ({entry_label}): {s['entry']:.{dec}f}",
        f"Live: {live:.{dec}f}" if live is not None else "Live: unavailable",
        f"SL {s['sl']:.{dec}f}  |  TP {s['tp']:.{dec}f}  |  R:R 1:{RR:.1f}{rr_txt}",
        f"Invalid if: price reaches SL, or TP before entry, or moves >{MAX_DRIFT_R}R "
        f"against entry, or after {s['expires_at']:%H:%M} UTC",
    ])


# ---------------------------------------------------------------------------
# state (dedupe)
# ---------------------------------------------------------------------------
def load_state():
    try:
        with open(STATE_PATH) as f:
            st = json.load(f)
        if not isinstance(st, dict):
            raise ValueError("state is not an object")
    except FileNotFoundError:
        st = {}
    except Exception as e:
        log("state unreadable, starting fresh:", type(e).__name__, e)
        st = {}
    if not isinstance(st.get("seen"), dict):
        st["seen"] = {}
    st.setdefault("last_error_alert", None)
    return st


def save_state(st, now):
    cutoff = (now - timedelta(days=STATE_KEEP_DAYS)).isoformat()
    st["seen"] = {k: v for k, v in st["seen"].items() if v.get("ts", "") >= cutoff}
    d = os.path.dirname(STATE_PATH)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1, sort_keys=True)
    os.replace(tmp, STATE_PATH)


def signal_id(name, s):
    """One id per break of one level, so the same setup alerts only once."""
    return f"{name}|{s['side']}|{s['label']}|{s['level']:.6g}|{s['break_time']}"


# ---------------------------------------------------------------------------
# scanning
# ---------------------------------------------------------------------------
def scan_pair(name, ticker, now=None):
    """Returns (signals, note, meta). Signals are newest-first."""
    now = now or datetime.now(timezone.utc)
    t0 = time.time()
    h1 = fetch(ticker, "1h", "1mo")
    m15 = fetch(ticker, "15m", "5d")
    meta = {"fetch_s": time.time() - t0, "age_min": None, "behind": None}
    if len(m15) and m15.index[-1] + M15 > now:       # drop the still-forming candle
        m15 = m15.iloc[:-1]
    if len(h1) and h1.index[-1] + timedelta(hours=1) > now:
        h1 = h1.iloc[:-1]
    if len(m15) < 40 or len(h1) < 60:
        raise RuntimeError("not enough data")

    last_end = m15.index[-1] + M15
    meta["age_min"] = (now - last_end).total_seconds() / 60
    # how many closed candles Yahoo has NOT published yet (0 = fully up to date)
    meta["behind"] = max(0, round((floor15(now) - last_end) / M15))
    if now - last_end > timedelta(minutes=STALE_MIN):
        return [], "market closed/stale", meta

    a_now = atr(m15).iloc[-1]
    # NOTE: levels come from the latest 1H data even when scanning one candle back.
    # That only matters if a UTC day/week rolls between the two candles.
    levels = key_levels(h1, a_now)
    ema = h1["close"].ewm(span=EMA_LEN, adjust=False).mean()
    trend_up = bool(h1["close"].iloc[-1] > ema.iloc[-1])
    sides = ("long", "short")
    if TREND_ON:
        sides = ("long",) if trend_up else ("short",)

    sigs = []
    for k in range(SCAN_BACK):
        sub = m15.iloc[: len(m15) - k]
        if len(sub) < 40:
            break
        sig = find_signal(sub, levels, sides, atr(sub).iloc[-1], trend_up)
        if sig:
            end = sub.index[-1] + M15
            sig["session"] = "in-session" if in_session(sub.index[-1].hour) else "off-session"
            sig["signal_time"] = end
            sig["expires_at"] = end + EXPIRY_CANDLES * M15
            sigs.append(sig)
    note = f"{len(levels)} levels, 1H trend {'up' if trend_up else 'down'}"
    return sigs, note, meta


def main():
    now = datetime.now(timezone.utc)
    state = load_state()
    seen = state["seen"]
    errors, metas, n_alerts = [], [], 0

    for name, ticker in PAIRS.items():
        try:
            sigs, note, meta = scan_pair(name, ticker, now)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {redact(e)}")
            log("ERROR", name, type(e).__name__, e)
            continue
        metas.append((name, meta))
        log(f"{name}: {note} | last candle closed {meta['age_min']:.0f} min ago, "
            f"{meta['behind']} candle(s) behind, fetch {meta['fetch_s']:.1f}s")

        for s in sigs:
            if s["score"] < MIN_SCORE:
                continue
            sid = signal_id(name, s)
            if sid in seen:
                log(f"skip {name}: duplicate setup already handled ({seen[sid].get('status')})")
                continue
            live = live_price(ticker)
            status, reason = classify(s, live, now)
            log(f"{name}: {s['side']} {s['label']} -> {status} ({reason})")
            sent = False
            if status in SEND_STATUSES:
                sent = send(format_alert(name, s, status, reason, live, now))
                n_alerts += int(sent)
                if not sent:
                    continue          # not marked seen: retry on the next run
            seen[sid] = {"ts": now.isoformat(), "status": status, "sent": sent}

    if metas:
        worst = max(metas, key=lambda x: x[1]["age_min"])
        behind = sum(1 for _, m in metas if m["behind"])
        log(f"timing: {len(metas)} pairs ok, {behind} with delayed candles, "
            f"oldest data {worst[1]['age_min']:.0f} min ({worst[0]})")

    if errors and len(errors) == len(PAIRS):
        last = state.get("last_error_alert")
        due = (not last) or (now - datetime.fromisoformat(last) >
                             timedelta(hours=ERROR_ALERT_COOLDOWN_H))
        if due and send("Scanner ERROR: all pairs failed.\n" + "\n".join(errors[:4])):
            state["last_error_alert"] = now.isoformat()

    if not DRY_RUN:
        try:
            save_state(state, now)
        except Exception as e:
            log("could not save state:", type(e).__name__, e)
    log("done", n_alerts, "alerts,", len(errors), "errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
