"""
Structure-Quality Engulf Backtester
====================================
Strategy under test:
  1. Find a 1H engulfing candle (bullish or bearish).
  2. Score the swing leg leading into it for "cleanliness" (0-6).
  3. Wait for price to retrace back into the 1H engulf candle's zone.
  4. Enter when a same-direction engulfing candle forms on the 5m inside
     that zone.
  5. Stop = beyond the 1H engulf candle's opposite extreme.
  6. Target = configurable R multiples (tests several at once).

Results are broken out by structure-quality score bucket (3/6, 4/6, 5/6, 6/6)
so you can see whether cleanliness actually separates win rate / expectancy,
the same way you're scoring the break-and-retest setups.

Data source: yfinance. NOTE Yahoo limits:
  - 5m candles: last ~60 days only
  - 1h candles: last ~730 days
This is fine for a first-pass read but keep it in mind vs. your longer
28-pair, 2-year break/retest backtests.

Run: pip install yfinance pandas numpy
     python structure_engulf_backtest.py
"""

import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PAIRS = {
    "US30": "^DJI",
    "XAUUSD": "GC=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "BTCUSDT": "BTC-USD",
    "ETHUSDT": "ETH-USD",
}

R_MULTIPLES = [2, 3]          # test both targets, like your other backtests
LOOKBACK_LEGS = 8             # how many 1H candles back to score "the leg"
MIN_BODY_RATIO = 0.60         # impulse candle body/range must avg >= this
MAX_CONSOL_RATIO = 0.35       # consolidation range vs impulse leg range
MAX_OVERLAP = 0.30            # max allowed overlap into prior candle's body
RETEST_MAX_BARS_5M = 48       # how many 5m bars (=4h) to wait for retest+engulf
ZONE_INVALIDATE_BUFFER = 0.0  # extra buffer beyond zone before killing setup


# ---------------------------------------------------------------------------
# CANDLE HELPERS
# ---------------------------------------------------------------------------

def body(c):
    return abs(c["Close"] - c["Open"])

def rng(c):
    return max(c["High"] - c["Low"], 1e-9)

def is_bull_engulf(prev, cur):
    return (cur["Close"] > cur["Open"] and prev["Close"] < prev["Open"]
            and cur["Close"] >= prev["Open"] and cur["Open"] <= prev["Close"])

def is_bear_engulf(prev, cur):
    return (cur["Close"] < cur["Open"] and prev["Close"] > prev["Open"]
            and cur["Open"] >= prev["Close"] and cur["Close"] <= prev["Open"])


# ---------------------------------------------------------------------------
# STRUCTURE QUALITY SCORE (0-6)
# ---------------------------------------------------------------------------

def structure_score(df, i, direction):
    """Score the leg ending at candle i (the 1H engulf candle)."""
    start = max(0, i - LOOKBACK_LEGS)
    leg = df.iloc[start:i + 1]
    if len(leg) < 3:
        return 0, {}

    checks = {}

    # 1. impulse body-to-range ratio
    body_ratios = [body(c) / rng(c) for _, c in leg.iterrows()]
    checks["body_ratio"] = np.mean(body_ratios) >= MIN_BODY_RATIO

    # 2. consolidation tightness: last 2-3 candles before the engulf vs whole leg
    consol = leg.iloc[-3:-1] if len(leg) >= 4 else leg.iloc[:1]
    consol_range = consol["High"].max() - consol["Low"].min() if len(consol) else 0
    leg_range = leg["High"].max() - leg["Low"].min()
    checks["consolidation"] = (consol_range / max(leg_range, 1e-9)) <= MAX_CONSOL_RATIO

    # 3. overlap check: consecutive candle bodies shouldn't eat >MAX_OVERLAP into prior
    overlaps = []
    rows = leg.reset_index(drop=True)
    for j in range(1, len(rows)):
        prev_c, cur_c = rows.iloc[j - 1], rows.iloc[j]
        prev_top, prev_bot = max(prev_c["Open"], prev_c["Close"]), min(prev_c["Open"], prev_c["Close"])
        cur_top, cur_bot = max(cur_c["Open"], cur_c["Close"]), min(cur_c["Open"], cur_c["Close"])
        overlap_amt = max(0, min(prev_top, cur_top) - max(prev_bot, cur_bot))
        overlaps.append(overlap_amt / max(body(prev_c), 1e-9))
    checks["low_overlap"] = (np.mean(overlaps) <= MAX_OVERLAP) if overlaps else True

    # 4. swing clarity: monotonic highs/lows in engulf direction
    highs, lows = leg["High"].values, leg["Low"].values
    if direction == "bull":
        checks["swing_clarity"] = np.mean(np.diff(highs) >= 0) >= 0.6 and np.mean(np.diff(lows) >= 0) >= 0.6
    else:
        checks["swing_clarity"] = np.mean(np.diff(highs) <= 0) >= 0.6 and np.mean(np.diff(lows) <= 0) >= 0.6

    # 5. engulf candle body strength vs recent average
    avg_body = np.mean([body(c) for _, c in df.iloc[max(0, i - 10):i].iterrows()])
    checks["strong_engulf"] = body(df.iloc[i]) >= avg_body

    # 6. retracement doesn't blow through the opposite extreme of the leg before engulf
    pre_engulf = leg.iloc[:-1]
    if direction == "bull":
        checks["clean_leg_low"] = df.iloc[i]["Low"] >= pre_engulf["Low"].min()
    else:
        checks["clean_leg_low"] = df.iloc[i]["High"] <= pre_engulf["High"].max()

    score = sum(checks.values())
    return score, checks


# ---------------------------------------------------------------------------
# ENGULF + RETEST + BACKTEST
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    pair: str
    direction: str
    score: int
    entry: float
    stop: float
    target: float
    r_mult: float
    outcome_r: float = 0.0


def find_htf_engulfs(df_1h):
    events = []
    for i in range(1, len(df_1h)):
        prev, cur = df_1h.iloc[i - 1], df_1h.iloc[i]
        if is_bull_engulf(prev, cur):
            score, checks = structure_score(df_1h, i, "bull")
            events.append((i, "bull", score, checks))
        elif is_bear_engulf(prev, cur):
            score, checks = structure_score(df_1h, i, "bear")
            events.append((i, "bear", score, checks))
    return events


def simulate_retest(df_5m, zone_low, zone_high, direction, engulf_time):
    """After the 1H engulf candle closes, watch 5m bars for a tap into the
    zone followed by a same-direction 5m engulf inside it."""
    window = df_5m[df_5m.index > engulf_time].iloc[:RETEST_MAX_BARS_5M]
    if len(window) < 2:
        return None

    tapped = False
    for j in range(1, len(window)):
        prev, cur = window.iloc[j - 1], window.iloc[j]
        c_low, c_high = cur["Low"], cur["High"]

        # invalidate if price blows through the zone entirely
        if direction == "bull" and c_low < zone_low - ZONE_INVALIDATE_BUFFER:
            return None
        if direction == "bear" and c_high > zone_high + ZONE_INVALIDATE_BUFFER:
            return None

        in_zone = c_low <= zone_high and c_high >= zone_low
        if in_zone:
            tapped = True

        if tapped:
            if direction == "bull" and is_bull_engulf(prev, cur):
                return cur
            if direction == "bear" and is_bear_engulf(prev, cur):
                return cur
    return None


def run_trade(entry_candle, direction, zone_low, zone_high, df_5m, entry_time, r_mult):
    entry = entry_candle["Close"]
    stop = zone_low if direction == "bull" else zone_high
    risk = abs(entry - stop)
    if risk == 0:
        return None
    target = entry + r_mult * risk if direction == "bull" else entry - r_mult * risk

    future = df_5m[df_5m.index > entry_time]
    for _, c in future.iterrows():
        if direction == "bull":
            if c["Low"] <= stop:
                return -1.0
            if c["High"] >= target:
                return r_mult
        else:
            if c["High"] >= stop:
                return -1.0
            if c["Low"] <= target:
                return r_mult
    return None  # ran out of data before resolving


def _flatten(df):
    """yfinance can return MultiIndex columns (ticker, field) even for a
    single ticker; flatten to plain field names so df["Close"] etc. work."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def backtest_pair(name, ticker):
    df_1h = _flatten(yf.download(ticker, period="730d", interval="1h", progress=False))
    df_5m = _flatten(yf.download(ticker, period="60d", interval="5m", progress=False))
    if df_1h.empty or df_5m.empty:
        print(f"[{name}] no data, skipping")
        return []

    trades = []
    events = find_htf_engulfs(df_1h)
    for i, direction, score, checks in events:
        engulf = df_1h.iloc[i]
        zone_low, zone_high = engulf["Low"], engulf["High"]
        entry_candle = simulate_retest(df_5m, zone_low, zone_high, direction, df_1h.index[i])
        if entry_candle is None:
            continue
        entry_time = entry_candle.name
        for r_mult in R_MULTIPLES:
            outcome = run_trade(entry_candle, direction, zone_low, zone_high, df_5m, entry_time, r_mult)
            if outcome is not None:
                trades.append(Trade(name, direction, score, entry_candle["Close"],
                                     zone_low if direction == "bull" else zone_high,
                                     0, r_mult, outcome))
    return trades


def summarize(all_trades):
    df = pd.DataFrame([t.__dict__ for t in all_trades])
    if df.empty:
        print("No trades generated.")
        return
    print("\n=== Overall ===")
    for r_mult in R_MULTIPLES:
        sub = df[df.r_mult == r_mult]
        if len(sub):
            print(f"R={r_mult}: {len(sub)} trades, win rate "
                  f"{(sub.outcome_r > 0).mean():.2%}, expectancy {sub.outcome_r.mean():.3f}R")

    print("\n=== By structure-quality score ===")
    for score in sorted(df.score.unique()):
        for r_mult in R_MULTIPLES:
            sub = df[(df.score == score) & (df.r_mult == r_mult)]
            if len(sub) >= 5:
                print(f"score {score}/6, R={r_mult}: {len(sub)} trades, "
                      f"win rate {(sub.outcome_r > 0).mean():.2%}, "
                      f"expectancy {sub.outcome_r.mean():.3f}R")

    df.to_csv("structure_engulf_trades.csv", index=False)
    print("\nFull trade log saved to structure_engulf_trades.csv")


if __name__ == "__main__":
    all_trades = []
    for name, ticker in PAIRS.items():
        print(f"Backtesting {name} ({ticker})...")
        all_trades.extend(backtest_pair(name, ticker))
    summarize(all_trades)
  
