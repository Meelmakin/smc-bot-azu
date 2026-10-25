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
  * Optional switches REQUIRE_SWEEP / REQUIRE_D1_TREND / REQUIRE_FRESH (all OFF = old rules).
  * Each setup gets a 0-5 score; MIN_SCORE and MAX_SETUPS_PER_DAY (default 3) limit alerts.
    The cap is first come, first served: a live bot cannot know a better one comes later.

STRICT MODE (default ON, set STRICT_MODE=0 to turn off): an alert is sent only if ALL five
checks of the plan pass (fresh level, sweep, higher-timeframe trend, tight invalidation, clean
retest wick). If one is missing, nothing is sent.

STYLE (default slk): 'slk' = 5pm-New-York candles for FX/metals, 14-month key levels, and breakouts
that happen during the rejection day (alert at the daily close). 'legacy' = the original rules.

Env vars: TELEGRAM_TOKEN, CHAT_ID  (optional: STYLE, STRICT_MODE, SENT_FILE, DRY_RUN=1, TEST_MESSAGE=true,
          TIMEFRAMES=D1H4 (default) or H4H1 for 4H levels with 1H structure)
"""

import contextlib
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
ALERT_STAGES = {"RETEST"}   # confirmed setups only; add "BOS" for earlier heads-ups (more alerts)

# Optional extra conditions seen in the slkradar alert. OFF = your original rules.
# Not yet backtested: bias_backtest.py compares them against the baseline.
REQUIRE_SWEEP = False      # rejection candle must take out the previous daily extreme
REQUIRE_D1_TREND = False   # BUY only above the daily EMA, SELL only below it
D1_TREND_EMA = 20          # my assumption for "D1 trend"; tune with the backtest
REQUIRE_FRESH = False      # key level must be untouched between its formation and the rejection

# Setup score (0-5, one point each; weights are equal and UNTESTED guesses):
#   fresh level, swept the previous candle's extreme, with the higher-timeframe trend,
#   tight invalidation (<= RISK_ATR_MAX lower-timeframe ATRs from the alert close),
#   clean rejection wick on the retest candle (>= 40% of the candle's range).
RISK_ATR_MAX = 5.0
MIN_SCORE = 0              # only alert setups scoring at least this (backtest picks the value)
MAX_SETUPS_PER_DAY = 3     # at most this many RETEST alerts per UTC day (first come first served)

SENT_FILE = os.environ.get("SENT_FILE", "state/sent_bias_alerts.json")
SENT_KEEP = 500
FETCH_RETRIES = 2
DRY_RUN = os.environ.get("DRY_RUN") == "1"
SECRET_ENV = ("TELEGRAM_TOKEN", "CHAT_ID")

H4 = pd.Timedelta(hours=4)
D1 = pd.Timedelta(days=1)
H1 = pd.Timedelta(hours=1)

# Timeframe presets. In the code below, D1 means "higher-timeframe candle span" and H4
# means "lower-timeframe candle span"; use_preset() re-points them, so the names stay
# but the meaning follows the preset. D1H4 = original (daily levels, 4H structure).
# H4H1 = faster version (4H levels, 1H structure). The faster one is UNTESTED live.
PRESETS = {
    "D1H4": dict(htf="1D", ltf="4h", htf_span=D1, ltf_span=H4, bos_max=30, retest_max=24,
                 htf_name="Daily", ltf_name="4H"),
    "H4H1": dict(htf="4h", ltf="1h", htf_span=H4, ltf_span=H1, bos_max=48, retest_max=36,
                 htf_name="4H", ltf_name="1H"),
}
HTF_RULE, LTF_RULE = "1D", "4h"
HTF_NAME, LTF_NAME = "Daily", "4H"
PRESET = "D1H4"


def use_preset(name):
    """Switch timeframes: 'D1H4' (default) or 'H4H1'."""
    global D1, H4, HTF_RULE, LTF_RULE, HTF_NAME, LTF_NAME, BOS_MAX_BARS, RETEST_MAX_BARS, PRESET
    if name not in PRESETS:
        raise ValueError(f"unknown timeframe preset {name!r}; choose from {sorted(PRESETS)}")
    p = PRESETS[name]
    D1, H4 = p["htf_span"], p["ltf_span"]
    HTF_RULE, LTF_RULE = p["htf"], p["ltf"]
    HTF_NAME, LTF_NAME = p["htf_name"], p["ltf_name"]
    BOS_MAX_BARS, RETEST_MAX_BARS = p["bos_max"], p["retest_max"]
    PRESET = name


# Styles. "legacy" = the original rules. "slk" = closer to what the slkradar alerts show:
#   * NY_ANCHOR: FX and metals build daily/4H candles on the 5pm New York close (like a broker
#     chart). Crypto and indices stay on UTC.
#   * LEVEL_MAX_AGE: key levels stay valid for ~14 months instead of 40 days (needs ~2 years of 1H data).
#   * SAME_DAY_BOS: a breakout that happens DURING the rejection day counts, and its alert waits for
#     the daily close (no look-ahead: the rejection is only known when that daily candle closes).
STYLES = {
    "legacy": dict(ny=False, age=40, same_day=False, fetch="60d"),
    "slk": dict(ny=True, age=300, same_day=True, fetch="730d"),
}
STYLE = "legacy"
NY_ANCHOR = False
SAME_DAY_BOS = False
NY_OFFSETS = {"1D": "17h", "4h": "1h"}      # candle starts at 17:00 New York (daily) / 01:00 (4H grid)
NON_NY_NAMES = {"US30", "US100", "US500", "GER40", "NAS100", "UK100", "JPN225", "SPX500",
                "ETHUSD", "BTCUSD", "SOLUSD"}


def use_style(name):
    global STYLE, NY_ANCHOR, SAME_DAY_BOS, LEVEL_MAX_AGE, FETCH_PERIOD
    if name not in STYLES:
        raise ValueError(f"unknown style {name!r}; choose from {sorted(STYLES)}")
    st = STYLES[name]
    NY_ANCHOR, SAME_DAY_BOS = st["ny"], st["same_day"]
    LEVEL_MAX_AGE, FETCH_PERIOD = st["age"], st["fetch"]
    STYLE = name


@contextlib.contextmanager
def style_ctx(name):
    """Apply a style for the duration of a run, then restore what was set before."""
    old = (STYLE, NY_ANCHOR, SAME_DAY_BOS, LEVEL_MAX_AGE, FETCH_PERIOD)
    use_style(name)
    try:
        yield
    finally:
        globals().update(STYLE=old[0], NY_ANCHOR=old[1], SAME_DAY_BOS=old[2],
                         LEVEL_MAX_AGE=old[3], FETCH_PERIOD=old[4])


def uses_ny_anchor(pair):
    """FX and metals follow the 5pm New York close; crypto and indices do not."""
    return not (pair.endswith("USDT") or pair in NON_NY_NAMES)


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


def resample_ohlc(df, rule, ny=False):
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if ny and rule in NY_OFFSETS:
        # Work on New York wall-clock time so bins start at 17:00 local all year (DST-aware).
        # FX is closed over the DST switch (Sunday 02:00 local), so no bar falls in an ambiguous hour.
        local = df.copy()
        local.index = df.index.tz_convert("America/New_York").tz_localize(None)
        # "1D" is not a fixed-length frequency in pandas, which makes it ignore `offset`; "24h" is
        # identical on a naive (no-DST) clock and honours it.
        freq = "24h" if rule == "1D" else rule
        out = local.resample(freq, offset=NY_OFFSETS[rule]).agg(agg).dropna()
        out.index = out.index.tz_localize("America/New_York", ambiguous="NaT",
                                          nonexistent="NaT").tz_convert("UTC")
        return out[out.index.notna()]
    return df.resample(rule).agg(agg).dropna()


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

    def swept(direction, j):
        if direction == "BUY":
            return bool(l[j] < l[j - 1] and c[j] > l[j - 1])
        return bool(h[j] > h[j - 1] and c[j] < h[j - 1])

    def trend_ok(direction, j):
        return bool(c[j] > ema[j]) if direction == "BUY" else bool(c[j] < ema[j])

    def fresh_level(i, j, level):
        """True if no candle after the level formed (i) and before the rejection (j) touched it."""
        return not any(l[k] <= level <= h[k] for k in range(i + 1, j))

    def passes(direction, j):
        if REQUIRE_SWEEP and not swept(direction, j):
            return False
        if REQUIRE_D1_TREND and not trend_ok(direction, j):
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
            if REQUIRE_FRESH and not fresh_level(i, j, level):
                continue
            key = (0 if kind != "open-close" else 1, -i)
            if best is None or key < best[0]:
                best = (key, kind, level, direction, came, i)
        if best is None:
            continue
        _, kind, level, direction, came, bi = best
        if j - last_idx[direction] <= BIAS_COOLDOWN:
            continue
        last_idx[direction] = j
        out.append({
            "stage": "BIAS", "direction": direction, "kind": kind, "came": came,
            "level": float(level), "time": D.index[j] + D1, "open": D.index[j],
            "rej_high": float(h[j]), "rej_low": float(l[j]),
            "invalidation": float(h[j] if direction == "SELL" else l[j]),
            "fresh": fresh_level(bi, j, level), "swept": swept(direction, j),
            "trend_ok": trend_ok(direction, j),
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
    op = H["Open"].values
    n = len(H)
    d = bias["direction"]
    t0 = H.index.searchsorted(bias["open"] if SAME_DAY_BOS else bias["time"])
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
            "key_level": bias["level"], "came": bias["came"],
            "fresh": bias["fresh"], "swept": bias["swept"], "trend_ok": bias["trend_ok"],
            "bias_time": bias["time"]}

    def when(t):
        """Alert time: the lower-timeframe candle's close, but never before the rejection candle closes."""
        return max(H.index[t] + H4, bias["time"])

    def risk_atr(t):
        """Distance from this bar's close to the invalidation level, in lower-timeframe ATRs."""
        if np.isnan(a4[t]) or a4[t] <= 0:
            return None
        return float(abs(cl[t] - bias["invalidation"]) / a4[t])

    def wick(t):
        """Rejection wick as a fraction of the candle's range (upper for SELL, lower for BUY)."""
        rng = hi[t] - lo[t]
        if rng <= 0:
            return 0.0
        w = (hi[t] - max(op[t], cl[t])) if d == "SELL" else (min(op[t], cl[t]) - lo[t])
        return float(w / rng)

    events.append({**base, "stage": "BOS", "time": when(tb), "risk_atr": risk_atr(tb)})
    for t in range(tb + 1, min(n, tb + 1 + RETEST_MAX_BARS)):
        tol = RETEST_TOL_ATR * a4[t]
        if np.isnan(tol):
            tol = 0.0
        if d == "SELL":
            if hi[t] >= lvl - tol and cl[t] < lvl:
                events.append({**base, "stage": "RETEST", "time": when(t),
                               "risk_atr": risk_atr(t), "retest_wick": wick(t)}); break
            if cl[t] >= lvl:
                break
        else:
            if lo[t] <= lvl + tol and cl[t] > lvl:
                events.append({**base, "stage": "RETEST", "time": when(t),
                               "risk_atr": risk_atr(t), "retest_wick": wick(t)}); break
            if cl[t] <= lvl:
                break
    return events


SCORE_LABELS = ("fresh level", "swept previous extreme", "with higher-timeframe trend",
                "tight invalidation", "clean rejection wick")


def score_flags(e):
    ra = e.get("risk_atr")
    return [bool(e.get("fresh")), bool(e.get("swept")), bool(e.get("trend_ok")),
            ra is not None and ra <= RISK_ATR_MAX,
            (e.get("retest_wick") or 0.0) >= 0.4]


def setup_score(e):
    return int(sum(score_flags(e)))


def score_reasons(e):
    return [lab for lab, ok in zip(SCORE_LABELS, score_flags(e)) if ok]


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
    ny = NY_ANCHOR and uses_ny_anchor(pair)
    D = complete_only(resample_ohlc(df1h, HTF_RULE, ny), D1, now)
    H = complete_only(resample_ohlc(df1h, LTF_RULE, ny), H4, now)
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
    for e in ev:
        e["score"] = setup_score(e)
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


def score_line(e):
    if "score" not in e:
        return ""
    why = ", ".join(score_reasons(e)) or "none of the extra checks"
    return f"Setup score: {e['score']}/5 ({why})\n"


def format_alert(e, now=None):
    sell = e["direction"] == "SELL"
    emoji = "🔴" if sell else "🟢"
    way = "down" if sell else "up"
    side = "above" if sell else "below"
    p = e["pair"]
    ltf, htf = LTF_NAME, HTF_NAME
    when = f"{ltf} candle closed {e['time']:%d %b %H:%M} UTC"
    if now is not None:
        when += f" ({int((now - e['time']).total_seconds() // 60)} min ago)"
    if e["stage"] == "BIAS":
        return (f"{emoji} *{p} -- {e['direction']} BIAS set*\n"
                f"{htf} rejection from {e['kind']} key level {e['level']:.5f} (price came from {e['came']})\n"
                f"Next: wait for a {ltf} BOS {way}, then a retest.\n"
                f"Bias invalid if {ltf} closes {side} {e['invalidation']:.5f}")
    if e["stage"] == "BOS":
        return (f"{emoji} *{p} -- {ltf} BOS {way}* ({e['direction']} bias)\n"
                f"{when}\n"
                f"{htf} {e['kind']} key level {e['key_level']:.5f} rejected (came from {e['came']})\n"
                f"{score_line(e)}"
                f"Broke {ltf} level {e['level']:.5f}\n"
                f"Next: wait for the retest of {e['level']:.5f}.\n"
                f"Bias invalid if {ltf} closes {side} {e['invalidation']:.5f}")
    return (f"{emoji} *{p} -- {ltf} RETEST, {e['direction']} bias*\n"
            f"{when}\n"
            f"{htf} {e['kind']} key level {e['key_level']:.5f} rejected (came from {e['came']})\n"
            f"{score_line(e)}"
            f"{ltf} BOS {way}, then retest of {e['level']:.5f} rejected.\n"
            f"Look for YOUR entry now. This is not an entry signal.\n"
            f"Bias invalid if {ltf} closes {side} {e['invalidation']:.5f}")


def sent_that_day(sent, e):
    """How many RETEST alerts were already sent for the UTC day of this event."""
    day = e["time"].strftime("%Y-%m-%d")
    n = 0
    for k in sent:
        parts = k.split("|")
        if len(parts) == 4 and parts[1] == "RETEST" and parts[3].startswith(day):
            n += 1
    return n


def hours_behind(df1h, now):
    """Closed 1H bars Yahoo has not published yet (0 = up to date)."""
    done = df1h[df1h.index + H1 <= now]
    if done.empty:
        return None
    return max(0, round((now.floor("h") - (done.index[-1] + H1)) / H1))


@contextlib.contextmanager
def strict_rules(on):
    """STRICT mode: every check of the plan must pass or nothing is sent. Turns on all three
    optional rules and requires the full setup score. Restores the settings afterwards."""
    global REQUIRE_SWEEP, REQUIRE_D1_TREND, REQUIRE_FRESH, MIN_SCORE
    old = (REQUIRE_SWEEP, REQUIRE_D1_TREND, REQUIRE_FRESH, MIN_SCORE)
    if on:
        REQUIRE_SWEEP = REQUIRE_D1_TREND = REQUIRE_FRESH = True
        MIN_SCORE = 5
    try:
        yield
    finally:
        REQUIRE_SWEEP, REQUIRE_D1_TREND, REQUIRE_FRESH, MIN_SCORE = old


def required_score(stage):
    """BOS has no retest candle yet, so its score tops out at 4."""
    return MIN_SCORE if stage == "RETEST" else max(0, MIN_SCORE - 1)


def main():
    use_preset(os.environ.get("TIMEFRAMES", "D1H4"))
    token, chat_id = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("CHAT_ID")
    if not DRY_RUN and (not token or not chat_id):
        log("Missing TELEGRAM_TOKEN or CHAT_ID env vars.")
        sys.exit(1)
    if os.environ.get("TEST_MESSAGE", "").lower() == "true":
        ok = send_telegram(token, chat_id, "✅ bias bot test message: Telegram works.")
        sys.exit(0 if ok else 1)

    strict = os.environ.get("STRICT_MODE", "1") != "0"
    with style_ctx(os.environ.get("STYLE", "slk")), strict_rules(strict):
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
                need = required_score(e["stage"])
                if e.get("score", 0) < need:
                    log(f"[{name}] skip {e['stage']} {e['direction']}: only {e.get('score')} checks passed, "
                        f"{need} needed")
                    continue
                if e["stage"] == "RETEST":
                    if sent_that_day(sent, e) >= MAX_SETUPS_PER_DAY:
                        log(f"[{name}] skip RETEST {e['direction']}: daily cap of {MAX_SETUPS_PER_DAY} reached")
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
