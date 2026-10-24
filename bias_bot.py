"""
bias_bot.py -- BIAS-ONLY alerts (v2). It never gives entries; you take those.

Method (all from your charts):
  1. KEY LEVELS from DAILY candle bodies:
       A-shape  : bullish candle then bearish candle, level = top of the "A"
       V-shape  : bearish candle then bullish candle, level = bottom of the "V"
       Open-close: two same-colour candles, level = where one closes and the
                   next opens
  2. REJECTION: a later daily candle wicks into the level and closes back on
     the side it came from.  Rejected from BELOW -> SELL, from ABOVE -> BUY.
  3. 4H BREAK OF STRUCTURE in the bias direction (close beyond the last 4H
     swing low for sell / swing high for buy).
  4. 4H RETEST of the broken level, rejected by a 4H close.

Alerts: BIAS (step 2), BOS (step 3), RETEST (step 4).
Invalidation: 4H close beyond the rejection candle's extreme.

v2 changes (rules are unchanged unless you switch the options below on):
  * Telegram token can no longer leak into logs/tracebacks; send failures
    no longer crash the run.
  * Alerts fire at BOS and RETEST (was RETEST only).
  * Event window widened to 3h (dedupe stops repeats), so a late run or
    delayed Yahoo data no longer loses an alert permanently.
  * Alerts for events whose bias has since been invalidated are skipped.
  * Sent-alert memory lives in SENT_FILE (default state/sent_bias_alerts.json);
    the workflow must keep that folder between runs (see README snippet).
  * Per-pair data lag is logged.
  * Optional switches REQUIRE_SWEEP / REQUIRE_D1_TREND (both OFF = old rules).

Env vars: TELEGRAM_TOKEN, CHAT_ID  (optional: SENT_FILE, DRY_RUN=1, TEST_MESSAGE=true)
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

PAIRS = {
    # forex majors and crosses
    "EURUSD=X": "EURUSD", "GBPUSD=X": "GBPUSD", "USDJPY=X": "USDJPY",
    "AUDUSD=X": "AUDUSD", "NZDUSD=X": "NZDUSD", "USDCAD=X": "USDCAD",
    "USDCHF=X": "USDCHF", "EURGBP=X": "EURGBP", "EURJPY=X": "EURJPY",
    "GBPJPY=X": "GBPJPY", "AUDJPY=X": "AUDJPY", "EURAUD=X": "EURAUD",
    "GBPAUD=X": "GBPAUD", "EURCAD=X": "EURCAD", "EURCHF=X": "EURCHF",
    "EURNZD=X": "EURNZD", "GBPCAD=X": "GBPCAD", "GBPCHF=X": "GBPCHF",
    "AUDCAD=X": "AUDCAD", "AUDNZD=X": "AUDNZD", "CADJPY=X": "CADJPY",
    "NZDJPY=X": "NZDJPY", "CHFJPY=X": "CHFJPY",
    # metals, indices, crypto
    "GC=F": "XAUUSD", "SI=F": "XAGUSD",
    "^DJI": "US30", "^NDX": "US100", "^GSPC": "US500", "^GDAXI": "GER40",
    "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT",
}

FETCH_PERIOD = "60d"       # 1h candles; daily + 4H are built from these
ATR_N = 14
MIN_BODY_ATR = 0.2         # both candles of a level need a real body
TOUCH_TOL_ATR = 0.1        # how close a wick must get to count as a touch
LEVEL_MAX_AGE = 40         # daily candles a level stays valid
BIAS_COOLDOWN = 3          # daily candles between same-direction biases
FRACTAL_N = 2
BOS_MAX_BARS = 30          # 4H bars to wait for the BOS (5 days)
RETEST_MAX_BARS = 24       # 4H bars to wait for the retest (4 days)
RETEST_TOL_ATR = 0.15

EVENT_WINDOW_MIN = 180     # only alert on events newer than this (dedupe stops repeats)
ALERT_STAGES = {"BOS", "RETEST"}   # add "BIAS" for the earliest (noisiest) heads-up

# Optional extra conditions seen in the slkradar alert. OFF = your original rules.
# Not yet backtested: bias_backtest.py compares them against the baseline.
REQUIRE_SWEEP = False      # rejection candle must take out the previous daily extreme
REQUIRE_D1_TREND = False   # BUY only above the daily EMA, SELL only below it
D1_TREND_EMA = 20          # my assumption for "D1 trend"; tune with the backtest

SENT_FILE = os.environ.get("SENT_FILE", "state/sent_bias_alerts.json")
SENT_KEEP = 500
FETCH_RETRIES = 2
DRY_RUN = os.environ.get("DRY_RUN") == "1"
SECRET_ENV = ("TELEGRAM_TOKEN", "CHAT_ID")

H4 = pd.Timedelta(hours=4)
D1 = pd.Timedelta(days=1)
H1 = pd.Timedelta(hours=1)


# ---------------------------------------------------------------- logging
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


# ---------------------------------------------------------------- data
def fetch_1h(ticker, period=None):
    import yfinance as yf
    df, err = None, "unknown"
    for attempt in range(FETCH_RETRIES + 1):
        try:
            d = yf.download(ticker, period=period or FETCH_PERIOD, interval="1h",
                            progress=False, auto_adjust=False)
            if len(d):
                df = d
                break
            err = "empty response"
        except Exception as ex:
            err = f"{type(ex).__name__}: {ex}"
        if attempt < FETCH_RETRIES:
            time.sleep(1.5 * (attempt + 1))
    if df is None:
        raise RuntimeError(f"{ticker} fetch failed: {redact(err)}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    return df[~df.index.duplicated(keep="last")].sort_index()


def resample_ohlc(df, rule):
    return df.resample(rule).agg(
        {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()


def complete_only(df, span, now):
    return df[df.index + span <= now]


def atr(df, n=ATR_N):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ---------------------------------------------------------------- step 1-2
def daily_biases(D):
    """Key level rejections on completed daily candles."""
    o, h, l, c = (D[k].values for k in ("Open", "High", "Low", "Close"))
    a = atr(D).values
    ema = D["Close"].ewm(span=D1_TREND_EMA, adjust=False).mean().values
    n = len(D)
    out = []
    last_idx = {"BUY": -99, "SELL": -99}

    def passes(direction, j):
        if REQUIRE_SWEEP:
            if direction == "BUY" and not (l[j] < l[j - 1] and c[j] > l[j - 1]):
                return False
            if direction == "SELL" and not (h[j] > h[j - 1] and c[j] < h[j - 1]):
                return False
        if REQUIRE_D1_TREND:
            if direction == "BUY" and not c[j] > ema[j]:
                return False
            if direction == "SELL" and not c[j] < ema[j]:
                return False
        return True

    for j in range(2, n):
        if np.isnan(a[j - 1]):
            continue
        tol = TOUCH_TOL_ATR * a[j - 1]
        best = None
        for i in range(max(1, j - LEVEL_MAX_AGE), j):
            if np.isnan(a[i]):
                continue
            if abs(c[i - 1] - o[i - 1]) < MIN_BODY_ATR * a[i] or \
               abs(c[i] - o[i]) < MIN_BODY_ATR * a[i]:
                continue
            bull0, bull1 = c[i - 1] > o[i - 1], c[i] > o[i]
            if bull0 and not bull1:
                kind, level = "A-shape", max(c[i - 1], o[i])
            elif (not bull0) and bull1:
                kind, level = "V-shape", min(c[i - 1], o[i])
            else:
                kind, level = "open-close", (c[i - 1] + o[i]) / 2.0
            prev = c[j - 1]
            # price must have been on the approach side for two closes
            if j - 2 > i and (c[j - 2] - level) * (prev - level) <= 0:
                continue
            if prev < level:                      # came from below
                if not (h[j] >= level - tol and c[j] < level):
                    continue
                direction, came = "SELL", "below"
            elif prev > level:                    # came from above
                if not (l[j] <= level + tol and c[j] > level):
                    continue
                direction, came = "BUY", "above"
            else:
                continue
            if not passes(direction, j):
                continue
            key = (0 if kind != "open-close" else 1, -i)
            if best is None or key < best[0]:
                best = (key, kind, level, direction, came)
        if best is None:
            continue
        _, kind, level, direction, came = best
        if j - last_idx[direction] <= BIAS_COOLDOWN:
            continue
        last_idx[direction] = j
        out.append({
            "stage": "BIAS", "direction": direction, "kind": kind, "came": came,
            "level": float(level), "time": D.index[j] + D1,
            "rej_high": float(h[j]), "rej_low": float(l[j]),
            "invalidation": float(h[j] if direction == "SELL" else l[j]),
        })
    return out


# ---------------------------------------------------------------- step 3-4
def swings(H, n=FRACTAL_N):
    win = 2 * n + 1
    sh = (H["High"] == H["High"].rolling(win, center=True).max()).values
    sl = (H["Low"] == H["Low"].rolling(win, center=True).min()).values
    return sh, sl


def track(bias, H, a4, sh, sl):
    """Follow one bias forward bar by bar: BOS, then retest."""
    hi, lo, cl = H["High"].values, H["Low"].values, H["Close"].values
    n = len(H)
    d = bias["direction"]
    t0 = H.index.searchsorted(bias["time"])
    if t0 >= n:
        return []
    last_sh = last_sl = None
    for k in range(max(0, t0 - 80), max(0, t0 - FRACTAL_N)):
        if sh[k]:
            last_sh = hi[k]
        if sl[k]:
            last_sl = lo[k]

    events, bos = [], None
    for t in range(t0, min(n, t0 + BOS_MAX_BARS)):
        k = t - 1 - FRACTAL_N          # swing newly confirmed by bar t-1
        if t > t0 and k >= 0:
            if sh[k]:
                last_sh = hi[k]
            if sl[k]:
                last_sl = lo[k]
        if d == "SELL":
            if cl[t] > bias["rej_high"]:
                return events
            if last_sl is not None and cl[t] < last_sl:
                bos = (t, last_sl); break
        else:
            if cl[t] < bias["rej_low"]:
                return events
            if last_sh is not None and cl[t] > last_sh:
                bos = (t, last_sh); break
    if bos is None:
        return events

    tb, lvl = bos
    base = {"direction": d, "level": float(lvl),
            "invalidation": bias["invalidation"], "kind": bias["kind"],
            "key_level": bias["level"], "came": bias["came"]}
    events.append({**base, "stage": "BOS", "time": H.index[tb] + H4})
    for t in range(tb + 1, min(n, tb + 1 + RETEST_MAX_BARS)):
        tol = RETEST_TOL_ATR * a4[t]
        if np.isnan(tol):
            tol = 0.0
        if d == "SELL":
            if hi[t] >= lvl - tol and cl[t] < lvl:
                events.append({**base, "stage": "RETEST", "time": H.index[t] + H4}); break
            if cl[t] >= lvl:
                break
        else:
            if lo[t] <= lvl + tol and cl[t] > lvl:
                events.append({**base, "stage": "RETEST", "time": H.index[t] + H4}); break
            if cl[t] <= lvl:
                break
    return events


def mark_invalidated(events, H):
    """Flag events whose bias was invalidated by a later 4H close."""
    cl = H["Close"]
    for e in events:
        after = cl[H.index >= e["time"]]
        if e["direction"] == "SELL":
            e["invalidated"] = bool((after > e["invalidation"]).any())
        else:
            e["invalidated"] = bool((after < e["invalidation"]).any())


def events_for(df1h, pair, now=None):
    """All BIAS / BOS / RETEST events for one pair, in time order.
    Only completed candles are used, so this is safe live and in a backtest."""
    if df1h is None or df1h.empty:
        return []
    data_end = df1h.index[-1] + H1
    now = data_end if now is None else min(now, data_end)
    D = complete_only(resample_ohlc(df1h, "1D"), D1, now)
    H = complete_only(resample_ohlc(df1h, "4h"), H4, now)
    if len(D) < ATR_N + 5 or len(H) < 40:
        return []
    a4 = atr(H).values
    sh, sl = swings(H)
    ev = []
    for b in daily_biases(D):
        ev.append(b)
        ev.extend(track(b, H, a4, sh, sl))
    for e in ev:
        e["pair"] = pair
    mark_invalidated(ev, H)
    return sorted(ev, key=lambda e: e["time"])


# ---------------------------------------------------------------- alerts
def event_key(e):
    return f"{e['pair']}|{e['stage']}|{e['direction']}|{e['time'].isoformat()}"


def load_sent():
    try:
        with open(SENT_FILE) as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()
    except Exception as ex:
        log("sent file unreadable, starting fresh:", type(ex).__name__, ex)
        return set()


def save_sent(sent):
    try:
        d = os.path.dirname(SENT_FILE)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = SENT_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(sorted(sent)[-SENT_KEEP:], f)
        os.replace(tmp, SENT_FILE)
    except Exception as ex:
        log(f"Could not save {SENT_FILE}:", type(ex).__name__, ex)


def send_telegram(token, chat_id, text):
    """True only if Telegram accepted the message. Never raises, never prints the token."""
    if DRY_RUN:
        log("[DRY_RUN] would send:\n" + text)
        return False
    for attempt in range(2):
        try:
            r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                              data={"chat_id": chat_id, "text": text,
                                    "parse_mode": "Markdown"}, timeout=15)
        except Exception as ex:   # the exception text contains the URL (= the token)
            log("telegram: request failed:", type(ex).__name__, ex)
        else:
            if r.status_code == 200:
                return True
            log("telegram: HTTP", r.status_code, r.text[:120])
        if attempt == 0:
            time.sleep(2)
    return False


def format_alert(e, now=None):
    sell = e["direction"] == "SELL"
    emoji = "🔴" if sell else "🟢"
    way = "down" if sell else "up"
    side = "above" if sell else "below"
    p = e["pair"]
    when = f"4H candle closed {e['time']:%d %b %H:%M} UTC"
    if now is not None:
        when += f" ({int((now - e['time']).total_seconds() // 60)} min ago)"
    if e["stage"] == "BIAS":
        return (f"{emoji} *{p} -- {e['direction']} BIAS set*\n"
                f"Daily rejection from {e['kind']} key level {e['level']:.5f} (price came from {e['came']})\n"
                f"Next: wait for a 4H BOS {way}, then a retest.\n"
                f"Bias invalid if 4H closes {side} {e['invalidation']:.5f}")
    if e["stage"] == "BOS":
        return (f"{emoji} *{p} -- 4H BOS {way}* ({e['direction']} bias)\n"
                f"{when}\n"
                f"Daily {e['kind']} key level {e['key_level']:.5f} rejected (came from {e['came']})\n"
                f"Broke 4H level {e['level']:.5f}\n"
                f"Next: wait for the retest of {e['level']:.5f}.\n"
                f"Bias invalid if 4H closes {side} {e['invalidation']:.5f}")
    return (f"{emoji} *{p} -- 4H RETEST, {e['direction']} bias*\n"
            f"{when}\n"
            f"Daily {e['kind']} key level {e['key_level']:.5f} rejected (came from {e['came']})\n"
            f"4H BOS {way}, then retest of {e['level']:.5f} rejected.\n"
            f"Look for YOUR entry now. This is not an entry signal.\n"
            f"Bias invalid if 4H closes {side} {e['invalidation']:.5f}")


def hours_behind(df1h, now):
    """Closed 1H bars Yahoo has not published yet (0 = up to date)."""
    done = df1h[df1h.index + H1 <= now]
    if done.empty:
        return None
    return max(0, round((now.floor("h") - (done.index[-1] + H1)) / H1))


def main():
    token, chat_id = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("CHAT_ID")
    if not DRY_RUN and (not token or not chat_id):
        log("Missing TELEGRAM_TOKEN or CHAT_ID env vars.")
        sys.exit(1)
    if os.environ.get("TEST_MESSAGE", "").lower() == "true":
        ok = send_telegram(token, chat_id, "✅ bias bot test message: Telegram works.")
        sys.exit(0 if ok else 1)

    now = pd.Timestamp(datetime.now(timezone.utc))
    sent, count, errs, lagging = load_sent(), 0, 0, 0
    for ticker, name in PAIRS.items():
        try:
            df = fetch_1h(ticker)
            events = events_for(df, name, now)
        except Exception as ex:
            errs += 1
            log(f"[{name}] error: {type(ex).__name__}: {ex}")
            continue
        behind = hours_behind(df, now)
        if behind:
            lagging += 1
            log(f"[{name}] data is {behind} hourly bar(s) behind")
        for e in events:
            if e["stage"] not in ALERT_STAGES:
                continue
            age_min = (now - e["time"]).total_seconds() / 60
            if age_min < 0 or age_min > EVENT_WINDOW_MIN:
                continue
            k = event_key(e)
            if k in sent:
                continue
            if e.get("invalidated"):
                log(f"[{name}] skip {e['stage']} {e['direction']}: bias already invalidated")
                continue
            if send_telegram(token, chat_id, format_alert(e, now)):
                sent.add(k)
                count += 1
                log(f"Alert: {name} {e['stage']} {e['direction']} @ {e['time']}")
    if not DRY_RUN:
        save_sent(sent)
    log(f"Run complete. {count} alert(s), {errs} error(s), {lagging} pair(s) with delayed data, "
        f"at {now.isoformat()}.")
    if errs == len(PAIRS):
        sys.exit(1)


if __name__ == "__main__":
    main()
