"""
match_check.py -- does our bias bot reproduce the real slkradar alerts, and what would it send right now?

PART 1. For every alert in slk_alerts.csv this runs OUR bot (slk style, every quality filter off) on the same
market and looks for the same rejection and the same external breakout. It reports one of:
  MATCH           same rejection candle, key level, breakout level, breakout candle and alert time
  PARTIAL         the same setup was found but some numbers differ (the differences are listed)
  REJECTION ONLY  we found their rejection but no external breakout
  NO REJECTION    we found no rejection on that day in that direction (we list what we did find nearby)
  NOT BUILT       a type we do not produce yet (internal breakout, range rejection)
  NO DATA         price data for that pair/date was not available
D1->H4 and W1->D1 alerts are both compared. Levels within 0.2% and times within 1 hour count as equal.
Prices come from Yahoo, theirs from a broker feed, so small level differences are normal.

PART 2. "What the live bot would have sent in the last N hours": the same rules and the same settings
(PROFILE, REQUIRE_D1_TREND ...) as the scan job, over all instruments. If this lists alerts but nothing
arrived in Telegram, the scan job itself is failing (look in Actions). If it lists nothing, the rules
really produced nothing in that window.

Run:  python match_check.py              (add --no-recent to skip part 2, --recent-hours 72 to change it)
"""
import argparse
import contextlib
import os
import sys

import numpy as np
import pandas as pd

import bias_backtest as bt
import bias_bot as bb
import slk_score as ss

LEVEL_TOL = 0.002
HOUR = pd.Timedelta(hours=1)


@contextlib.contextmanager
def no_filters(style="slk"):
    """Our events exactly as the rules produce them: slk style, every optional filter off."""
    keep = (bb.REQUIRE_SWEEP, bb.REQUIRE_D1_TREND, bb.REQUIRE_FRESH)
    bb.REQUIRE_SWEEP = bb.REQUIRE_D1_TREND = bb.REQUIRE_FRESH = False
    try:
        with bb.style_ctx(style):
            yield
    finally:
        bb.REQUIRE_SWEEP, bb.REQUIRE_D1_TREND, bb.REQUIRE_FRESH = keep


def preset_for(htf_hours):
    return "W1D1" if int(htf_hours) >= 168 else "D1H4"


def our_events(df, pair, preset="D1H4"):
    old_preset = bb.PRESET
    bb.use_preset(preset)
    try:
        with no_filters():
            return bb.events_for(df, pair)
    finally:
        bb.use_preset(old_preset)


def rel(a, b):
    return abs(a - b) / abs(b) if b else float("inf")


def match_alert(a, ev, level_tol=LEVEL_TOL):
    res = {"id": a["id"], "pair": a["pair"], "direction": a["direction"], "notes": []}
    if a["kind"] != "external":
        res["status"] = "NOT BUILT"
        res["notes"].append(f"{a['kind']} alerts are not produced by our bot yet")
        return res
    chart = "W1" if int(a["htf_hours"]) >= 168 else "D1"
    span = pd.Timedelta(days=3 if chart == "D1" else 14)

    d = a["direction"]
    same = [e for e in ev if e["stage"] == "BIAS" and e["direction"] == d
            and abs(e["open"] - a["rej_open_utc"]) <= HOUR]
    if not same:
        near = [e for e in ev if e["stage"] == "BIAS" and abs(e["open"] - a["rej_open_utc"]) <= span]
        res["status"] = "NO REJECTION"
        res["ours_nearby"] = [f"{e['direction']} {e['kind']} KL {e['level']:.5g} (candle {e['open']:%d %b %H:%M})"
                              for e in near]
        return res

    b = min(same, key=lambda e: abs(e["open"] - a["rej_open_utc"]))
    res.update(kl_ours=b["level"], trend_ok=b["trend_ok"], fresh=b["fresh"], swept=b["swept"], shape=b["kind"])
    notes = res["notes"]
    if "key_level" in a and np.isfinite(a["key_level"]) and rel(b["level"], a["key_level"]) > level_tol:
        notes.append(f"key level: ours {b['level']:.5g}, theirs {a['key_level']:.5g}")
    if not b["trend_ok"]:
        notes.append(f"trend: our rule says NOT aligned with {chart} (their alert says aligned)")

    bos = [e for e in ev if e["stage"] == "BOS" and e["direction"] == d and e.get("bias_open") == b["open"]]
    if not bos:
        res["status"] = "REJECTION ONLY"
        notes.append("we found no external breakout for this rejection")
        return res
    e = bos[0]
    res.update(bo_ours=e["level"])
    if np.isfinite(a["bo_level"]) and rel(e["level"], a["bo_level"]) > level_tol:
        notes.append(f"breakout level: ours {e['level']:.5g}, theirs {a['bo_level']:.5g}")
    if abs(e["bar_open"] - a["bo_time_utc"]) > HOUR:
        notes.append(f"breakout candle: ours {e['bar_open']:%d %b %H:%M}, theirs {a['bo_time_utc']:%d %b %H:%M} UTC")
    if abs(e["time"] - a["alert_utc"]) > HOUR:
        notes.append(f"alert time: ours {e['time']:%d %b %H:%M}, theirs {a['alert_utc']:%d %b %H:%M} UTC")
    res["status"] = "MATCH" if not notes else "PARTIAL"
    return res


def recent_report(hours=72, cache_dir="bt_cache", refresh=False):
    """What the scan job would send over the last `hours`, with the same settings it runs with."""
    strict = os.environ.get("STRICT_MODE", "0" if bb.PROFILE in ("slk", "all") else "1") != "0"
    print(f"\n=== WHAT THE LIVE BOT WOULD HAVE SENT IN THE LAST {hours} HOURS ===")
    print(f"settings: profile={bb.PROFILE or 'none'}, strict={strict}, stages={sorted(bb.ALERT_STAGES)}, "
          f"trend required={bb.REQUIRE_D1_TREND}, daily alignment={bb.REQUIRE_DAILY_ALIGN}, "
          f"daily cap={bb.MAX_SETUPS_PER_DAY or 'none'}")
    rows, loaded, failed = [], 0, []
    with bb.style_ctx("slk"), bb.strict_rules(strict):
        for ticker, name in bb.PAIRS.items():
            try:
                df = bt.load(name, ticker, refresh, cache_dir)
            except Exception as ex:
                failed.append(f"{name} ({type(ex).__name__})")
                continue
            loaded += 1
            end = df.index[-1] + HOUR
            for e in bb.events_all(df, name, None, ["D1H4", "W1D1"]):
                if e["stage"] not in bb.ALERT_STAGES or not (end - hours * HOUR <= e["time"] <= end):
                    continue
                if e.get("invalidated") or e.get("score", 0) < bb.required_score(e["stage"]):
                    continue
                if bb.REQUIRE_DAILY_ALIGN and e.get("daily_ok") is False:
                    continue
                rows.append(e)
    rows.sort(key=lambda e: e["time"])
    perday = {}
    shown = []
    for e in rows:                       # the daily cap applies to retests, first come first served
        if e["stage"] == "RETEST" and bb.MAX_SETUPS_PER_DAY > 0:
            day = e["time"].strftime("%Y-%m-%d")
            perday[day] = perday.get(day, 0) + 1
            if perday[day] > bb.MAX_SETUPS_PER_DAY:
                continue
        shown.append(e)
    print(f"instruments loaded: {loaded}" + (f"; could not load: {', '.join(failed)}" if failed else ""))
    for e in shown:
        pair = bb.DISPLAY_NAMES.get(e["pair"], e["pair"])
        print(f"  {e['time']:%a %d %b %H:%M} UTC  {pair:<8} {e['tf']}  {e['stage']:<6} {e['direction']:<4} "
              f"trend {'aligned' if e.get('trend_ok') else 'NOT aligned'}  score {e.get('score', 0)}/5")
    print(f"\nthe live rules would have sent {len(shown)} alert(s) in this window")
    if shown:
        print("If these did NOT arrive in Telegram, the scan job is not running or is failing: "
              "open GitHub -> Actions -> scan and read the newest run.")
    else:
        print("Nothing qualified in this window even with your settings. Loosen them (for example "
              "PROFILE all and no trend requirement) to see more.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", default="slk_alerts.csv")
    ap.add_argument("--cache-dir", default="slk_cache")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--recent-hours", type=int, default=72)
    ap.add_argument("--no-recent", action="store_true")
    a = ap.parse_args()

    alerts = ss.load_alerts(a.alerts)
    frames, events, why = {}, {}, {}
    for pair in sorted(alerts["pair"].unique()):
        try:
            frames[pair] = ss.load_prices(pair, a.cache_dir, a.refresh)
            print(f"loaded {pair}: {len(frames[pair])} 1H bars to {frames[pair].index[-1]:%Y-%m-%d}")
        except Exception as ex:
            why[pair] = f"{type(ex).__name__}: {ex}"
            print(f"skip {pair}: {why[pair]}")
    for _, row in alerts.iterrows():
        key = (row["pair"], preset_for(row["htf_hours"]))
        if row["pair"] in frames and key not in events:
            events[key] = our_events(frames[row["pair"]], row["pair"], key[1])

    results = []
    for _, row in alerts.iterrows():
        key = (row["pair"], preset_for(row["htf_hours"]))
        if key not in events:
            results.append({"id": row["id"], "pair": row["pair"], "direction": row["direction"],
                            "status": "NO DATA", "notes": [why.get(row["pair"], "")]})
            continue
        results.append(match_alert(row, events[key]))

    print("\n=== EVERY ALERT ===")
    for r in results:
        print(f"{r['id']:<4} {r['pair']:<8} {r['direction']:<4} {r['status']}")
        for n in r["notes"]:
            print(f"       - {n}")
        for n in r.get("ours_nearby", []):
            print(f"       we had: {n}")

    counts = pd.Series([r["status"] for r in results]).value_counts()
    print("\n=== SUMMARY ===")
    for k, v in counts.items():
        print(f"{k:<16s} {v}")
    built = [r for r in results if r["status"] not in ("NOT BUILT", "NO DATA")]
    ok = sum(r["status"] == "MATCH" for r in built)
    print(f"\nreproduced exactly: {ok} of {len(built)} comparable alerts")
    tally = {"key level": 0, "breakout level": 0, "breakout candle": 0, "alert time": 0, "trend": 0}
    for r in results:
        for n in r["notes"]:
            for k in tally:
                if n.startswith(k):
                    tally[k] += 1
    print("what differed (number of alerts):", ", ".join(f"{k} {v}" for k, v in tally.items()))
    print("A 'trend' difference means our trend rule disagrees with theirs; a 'key level' difference "
          "usually means a different candle pair or a data-feed difference.")

    if not a.no_recent:
        recent_report(a.recent_hours, refresh=a.refresh)


if __name__ == "__main__":
    main()
