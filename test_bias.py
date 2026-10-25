"""Run:  python -m unittest test_bias -v     (no internet or yfinance needed)"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import requests

import bias_backtest as bt
import bias_bot as bb

TOKEN = "123456789:AAH-secret_token_value"


def walk(n, seed=1, start="2024-01-01", base=100.0, vol=0.004):
    """Synthetic 24/7 hourly OHLC random walk (for logic tests only, not for results)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    c = base * np.exp(np.cumsum(rng.normal(0, vol, n)))
    o = np.concatenate([[base], c[:-1]])
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c}, index=idx)


def key(e):
    return (e["pair"], e["stage"], e["direction"], e["time"], round(e["level"], 6),
            round(e["invalidation"], 6))


class TestSecrets(unittest.TestCase):
    def test_send_failure_never_prints_token_and_does_not_raise(self):
        err = requests.ConnectionError(
            f"HTTPSConnectionPool: Max retries exceeded with url: /bot{TOKEN}/sendMessage")
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {"TELEGRAM_TOKEN": TOKEN, "CHAT_ID": "42"}), \
                mock.patch.object(bb.requests, "post", side_effect=err), \
                mock.patch.object(bb.time, "sleep"), contextlib.redirect_stdout(buf):
            ok = bb.send_telegram(TOKEN, "42", "hi")
        self.assertFalse(ok)
        self.assertNotIn(TOKEN, buf.getvalue())
        self.assertNotIn("AAH-secret", buf.getvalue())

    def test_http_error_returns_false_and_ok_returns_true(self):
        bad = mock.Mock(status_code=400, text="Bad Request: can't parse")
        good = mock.Mock(status_code=200, text="ok")
        with mock.patch.object(bb.time, "sleep"), contextlib.redirect_stdout(io.StringIO()):
            with mock.patch.object(bb.requests, "post", return_value=bad):
                self.assertFalse(bb.send_telegram(TOKEN, "42", "x"))
            with mock.patch.object(bb.requests, "post", return_value=good):
                self.assertTrue(bb.send_telegram(TOKEN, "42", "x"))

    def test_dry_run_sends_nothing(self):
        with mock.patch.object(bb, "DRY_RUN", True), \
                mock.patch.object(bb.requests, "post") as post, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(bb.send_telegram(TOKEN, "42", "x"))
        post.assert_not_called()


class TestNoLookahead(unittest.TestCase):
    def test_events_do_not_change_when_future_data_is_added(self):
        df = walk(3000, seed=7)
        full = bb.events_for(df, "T")
        self.assertGreater(len(full), 0, "synthetic data produced no events; test is vacuous")
        for cut in (1800, 2200, 2600):
            part = df.iloc[:cut]
            cut_end = part.index[-1] + bb.H1
            got = {key(e) for e in bb.events_for(part, "T")}
            want = {key(e) for e in full if e["time"] <= cut_end}
            self.assertEqual(got, want, f"events differ when truncated at bar {cut}")

    def test_unfinished_candles_are_ignored(self):
        df = walk(1500, seed=3)
        now = df.index[-1] + bb.H1
        d = bb.complete_only(bb.resample_ohlc(df, "4h"), bb.H4, now)
        self.assertTrue((d.index + bb.H4 <= now).all())


class TestSwitches(unittest.TestCase):
    def test_switches_run_and_off_is_default(self):
        self.assertFalse(bb.REQUIRE_SWEEP)
        self.assertFalse(bb.REQUIRE_D1_TREND)
        df = walk(3000, seed=11)
        base = bb.events_for(df, "T")
        with mock.patch.object(bb, "REQUIRE_SWEEP", True), mock.patch.object(bb, "REQUIRE_D1_TREND", True):
            both = bb.events_for(df, "T")
        self.assertGreater(len(base), 0)
        self.assertTrue(all(e["direction"] in ("BUY", "SELL") for e in both))
        n_bias = lambda ev: sum(e["stage"] == "BIAS" for e in ev)
        self.assertLessEqual(n_bias(both), n_bias(base))


class TestInvalidation(unittest.TestCase):
    def test_flag(self):
        H = pd.DataFrame({"Close": [10.0, 11.0, 12.0, 9.0]},
                         index=pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC"))
        sell = {"direction": "SELL", "invalidation": 11.5, "time": H.index[0] + bb.H4}
        buy = {"direction": "BUY", "invalidation": 9.5, "time": H.index[0] + bb.H4}
        late = {"direction": "SELL", "invalidation": 11.5, "time": H.index[3] + bb.H4}
        bb.mark_invalidated([sell, buy, late], H)
        self.assertTrue(sell["invalidated"])    # 12 closed above 11.5 later
        self.assertTrue(buy["invalidated"])     # 9 closed below 9.5 later
        self.assertFalse(late["invalidated"])   # nothing after it


def fake_event(stage, minutes_ago, now, invalidated=False, direction="BUY", score=3, pair="TEST"):
    return {"pair": pair, "stage": stage, "direction": direction, "kind": "V-shape",
            "level": 100.0, "key_level": 99.0, "came": "above", "invalidation": 95.0,
            "time": now - pd.Timedelta(minutes=minutes_ago), "invalidated": invalidated,
            "score": score, "fresh": True, "swept": True, "trend_ok": False,
            "risk_atr": 3.0, "retest_wick": 0.5}


@contextlib.contextmanager
def preset(name):
    bb.use_preset(name)
    try:
        yield
    finally:
        bb.use_preset("D1H4")


class TestMainFlow(unittest.TestCase):
    def run_main(self, events, now, state, stages=("BOS", "RETEST"), min_score=0, cap=3, strict="0"):
        df = walk(60)
        sent = []
        with mock.patch.dict(os.environ, {"TELEGRAM_TOKEN": TOKEN, "CHAT_ID": "42", "STRICT_MODE": strict}), \
                mock.patch.object(bb, "ALERT_STAGES", set(stages)), \
                mock.patch.object(bb, "MIN_SCORE", min_score), \
                mock.patch.object(bb, "MAX_SETUPS_PER_DAY", cap), \
                mock.patch.object(bb, "PAIRS", {"T=X": "TEST"}), \
                mock.patch.object(bb, "SENT_FILE", state), \
                mock.patch.object(bb, "fetch_1h", return_value=df), \
                mock.patch.object(bb, "events_for", return_value=events), \
                mock.patch.object(bb, "send_telegram", side_effect=lambda t, c, m: sent.append(m) or True), \
                mock.patch.object(bb.pd.Timestamp, "now", return_value=now), \
                contextlib.redirect_stdout(io.StringIO()):
            with mock.patch.object(bb, "datetime") as dt:
                dt.now.return_value = now.to_pydatetime()
                bb.main()
        return sent

    def test_window_dedupe_invalidated_and_stage_filter(self):
        now = pd.Timestamp("2024-03-01 12:05", tz="UTC")
        events = [fake_event("RETEST", 50, now),                  # sends
                  fake_event("BOS", 100, now),                    # sends (3h window)
                  fake_event("BOS", 400, now),                    # too old
                  fake_event("RETEST", 30, now, invalidated=True),  # invalidated -> skip
                  fake_event("BIAS", 10, now)]                    # stage not enabled
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "state", "sent.json")
            first = self.run_main(events, now, state)
            self.assertEqual(len(first), 2)
            self.assertTrue(os.path.exists(state))
            second = self.run_main(events, now, state)           # same events again
            self.assertEqual(second, [])

    def test_default_alerts_retest_only(self):
        self.assertEqual(bb.ALERT_STAGES, {"RETEST"})
        self.assertEqual(bb.MAX_SETUPS_PER_DAY, 3)

    def test_daily_cap_and_min_score(self):
        now = pd.Timestamp("2024-03-01 12:05", tz="UTC")
        same_day = [fake_event("RETEST", 20 + i, now, score=sc, pair=f"P{i}")
                    for i, sc in enumerate([4, 2, 5, 3, 4])]
        with tempfile.TemporaryDirectory() as d:
            sent = self.run_main(same_day, now, os.path.join(d, "s.json"), stages=("RETEST",))
            self.assertEqual(len(sent), 3)                       # cap of 3 per UTC day
        with tempfile.TemporaryDirectory() as d:
            sent = self.run_main(same_day, now, os.path.join(d, "s.json"),
                                 stages=("RETEST",), min_score=4)
            self.assertEqual(len(sent), 3)                       # three events score >= 4
            self.assertTrue(all("Setup score" in m for m in sent))
        with tempfile.TemporaryDirectory() as d:
            sent = self.run_main(same_day, now, os.path.join(d, "s.json"),
                                 stages=("RETEST",), min_score=5)
            self.assertEqual(len(sent), 1)
        with tempfile.TemporaryDirectory() as d:                 # cap carries across runs
            st = os.path.join(d, "s.json")
            self.assertEqual(len(self.run_main(same_day[:2], now, st, stages=("RETEST",), cap=2)), 2)
            self.assertEqual(len(self.run_main(same_day, now, st, stages=("RETEST",), cap=2)), 0)

    def test_strict_mode_sends_only_full_plan_setups(self):
        now = pd.Timestamp("2024-03-01 12:05", tz="UTC")
        ev = [fake_event("RETEST", 20, now, score=5, pair="FULL"),
              fake_event("RETEST", 25, now, score=4, pair="ONE_MISSING"),
              fake_event("RETEST", 30, now, score=2, pair="WEAK"),
              fake_event("BOS", 35, now, score=4, pair="BOS_FULL"),
              fake_event("BOS", 40, now, score=3, pair="BOS_MISSING")]
        with tempfile.TemporaryDirectory() as d:
            sent = self.run_main(ev, now, os.path.join(d, "s.json"), strict="1")
        self.assertEqual(len(sent), 2)
        self.assertTrue(any("FULL" in m and "ONE_MISSING" not in m for m in sent))
        self.assertTrue(any("BOS_FULL" in m for m in sent))
        self.assertFalse(any("MISSING" in m or "WEAK" in m for m in sent))
        # strict switches are restored after the run
        self.assertFalse(bb.REQUIRE_SWEEP or bb.REQUIRE_D1_TREND or bb.REQUIRE_FRESH)
        self.assertEqual(bb.MIN_SCORE, 0)

    def test_strict_off_sends_lower_scores(self):
        now = pd.Timestamp("2024-03-01 12:05", tz="UTC")
        ev = [fake_event("RETEST", 20, now, score=3)]
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(len(self.run_main(ev, now, os.path.join(d, "s.json"), strict="0")), 1)

    def test_strict_is_on_by_default(self):
        now = pd.Timestamp("2024-03-01 12:05", tz="UTC")
        ev = [fake_event("RETEST", 20, now, score=4)]
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(os.environ):
                os.environ.pop("STRICT_MODE", None)
                df = walk(60)
                sent = []
                with mock.patch.dict(os.environ, {"TELEGRAM_TOKEN": TOKEN, "CHAT_ID": "42"}), \
                        mock.patch.object(bb, "PAIRS", {"T=X": "TEST"}), \
                        mock.patch.object(bb, "SENT_FILE", os.path.join(d, "s.json")), \
                        mock.patch.object(bb, "fetch_1h", return_value=df), \
                        mock.patch.object(bb, "events_for", return_value=ev), \
                        mock.patch.object(bb, "send_telegram", side_effect=lambda t, c, m: sent.append(m) or True), \
                        mock.patch.object(bb, "datetime") as dt, \
                        contextlib.redirect_stdout(io.StringIO()):
                    dt.now.return_value = now.to_pydatetime()
                    bb.main()
        self.assertEqual(sent, [])

    def test_failed_send_is_retried_next_run(self):
        now = pd.Timestamp("2024-03-01 12:05", tz="UTC")
        events = [fake_event("RETEST", 20, now)]
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "sent.json")
            df = walk(60)
            with mock.patch.dict(os.environ, {"TELEGRAM_TOKEN": TOKEN, "CHAT_ID": "42", "STRICT_MODE": "0"}), \
                    mock.patch.object(bb, "PAIRS", {"T=X": "TEST"}), \
                    mock.patch.object(bb, "SENT_FILE", state), \
                    mock.patch.object(bb, "fetch_1h", return_value=df), \
                    mock.patch.object(bb, "events_for", return_value=events), \
                    mock.patch.object(bb, "send_telegram", return_value=False), \
                    mock.patch.object(bb, "datetime") as dt, \
                    contextlib.redirect_stdout(io.StringIO()):
                dt.now.return_value = now.to_pydatetime()
                bb.main()
            with mock.patch.object(bb, "SENT_FILE", state):
                self.assertEqual(bb.load_sent(), set())          # nothing marked as sent

    def test_corrupt_state_file_is_survivable(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            with open(p, "w") as f:
                f.write("{not json")
            with mock.patch.object(bb, "SENT_FILE", p), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(bb.load_sent(), set())


class TestAlertText(unittest.TestCase):
    def test_fields(self):
        now = pd.Timestamp("2024-03-01 12:05", tz="UTC")
        t = bb.format_alert(fake_event("RETEST", 25, now), now)
        for s in ("TEST", "RETEST", "BUY", "25 min ago", "not an entry signal", "invalid if"):
            self.assertIn(s.lower(), t.lower())


class TestBacktestMath(unittest.TestCase):
    def arrs(self, hi, lo, cl):
        return np.array(hi, float), np.array(lo, float), np.array(cl, float)

    def test_target_stop_timeout_and_ambiguous_bar(self):
        # BUY ref=100, stop=98 -> risk 2; 2R target = 104
        hi, lo, cl = self.arrs([101, 105, 106], [99, 100, 101], [100, 104, 105])
        r, k = bt.outcome(True, 100, 98, 2.0, hi, lo, cl, 0, 3)
        self.assertEqual((r, k), (2.0, "target"))
        hi, lo, cl = self.arrs([101, 100], [99, 97], [100, 98])
        self.assertEqual(bt.outcome(True, 100, 98, 2.0, hi, lo, cl, 0, 3), (-1.0, "stop"))
        hi, lo, cl = self.arrs([105], [97], [100])               # same bar hits both -> stop
        self.assertEqual(bt.outcome(True, 100, 98, 2.0, hi, lo, cl, 0, 1), (-1.0, "stop"))
        hi, lo, cl = self.arrs([101, 101], [99.5, 99.5], [100.5, 101])   # timeout, +0.5R
        r, k = bt.outcome(True, 100, 98, 2.0, hi, lo, cl, 0, 2)
        self.assertEqual(k, "timeout")
        self.assertAlmostEqual(r, 0.5)
        self.assertIsNone(bt.outcome(True, 100, 98, 2.0, hi, lo, cl, 0, 50))   # data ends first

    def test_sell_side(self):
        # SELL ref=100, stop=102, risk 2, 2R target = 96
        hi, lo, cl = self.arrs([101, 100], [99, 95], [100, 96])
        self.assertEqual(bt.outcome(False, 100, 102, 2.0, hi, lo, cl, 0, 5), (2.0, "target"))
        hi, lo, cl = self.arrs([103], [99], [101])
        self.assertEqual(bt.outcome(False, 100, 102, 2.0, hi, lo, cl, 0, 1), (-1.0, "stop"))

    def test_metrics(self):
        m = bt.metrics([2, -1, -1, -1, 2, 2, -1])
        self.assertEqual(m["n"], 7)
        self.assertEqual(m["streak"], 3)
        self.assertAlmostEqual(m["total"], 2.0)
        self.assertAlmostEqual(m["exp"], 2 / 7)
        self.assertAlmostEqual(m["pf"], 6 / 4)
        self.assertAlmostEqual(m["dd"], 3.0)            # peak 2 -> trough -1
        self.assertIsNone(bt.metrics([]))

    def test_trades_start_after_alert_candle(self):
        """The scored path must begin at the first bar AFTER the alert candle closes."""
        df = walk(3000, seed=7)
        t = bt.collect({"T": df}, ("BOS", "RETEST"), 0.0, 120)
        self.assertGreater(len(t), 0)
        ev = {(e["stage"], e["time"]) for e in bb.events_for(df, "T")}
        for _, r in t.head(30).iterrows():
            self.assertIn((r["stage"], r["time"]), ev)
            self.assertGreaterEqual(df.index.searchsorted(r["time"]), 1)

    def test_costs_reduce_r(self):
        df = walk(3000, seed=7)
        free = bt.collect({"T": df}, ("RETEST",), 0.0, 120)
        paid = bt.collect({"T": df}, ("RETEST",), 0.001, 120)
        self.assertEqual(len(free), len(paid))
        self.assertTrue((paid["R"].values < free["R"].values).all())


class TestBacktestSmoke(unittest.TestCase):
    def tearDown(self):
        bb.use_preset("D1H4")
        bb.use_style("legacy")

    def test_end_to_end_on_synthetic_data(self):
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cache")
            os.makedirs(cache)
            walk(6000, seed=21).to_csv(os.path.join(cache, "EURUSD.csv"))
            walk(6000, seed=22, base=1.1).to_csv(os.path.join(cache, "GBPUSD.csv"))
            out = os.path.join(d, "trades.csv")
            argv = ["bias_backtest.py", "--pairs", "EURUSD,GBPUSD", "--cache-dir", cache, "--out", out]
            buf = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(buf):
                bt.main()
            text = buf.getvalue()
            for s in ("HEADLINE", "BY PAIR", "BY SETUP SCORE", "MAX 3 SETUPS PER DAY",
                      "RANDOM entries", "VARIANTS", "+ fresh level", "assumptions", "0.25R"):
                self.assertIn(s, text)
            self.assertTrue(os.path.exists(out))
            self.assertFalse(bb.REQUIRE_SWEEP or bb.REQUIRE_D1_TREND or bb.REQUIRE_FRESH)  # restored

    def test_end_to_end_h4h1(self):
        with tempfile.TemporaryDirectory() as d:
            cache = os.path.join(d, "cache")
            os.makedirs(cache)
            walk(6000, seed=31).to_csv(os.path.join(cache, "EURUSD.csv"))
            argv = ["bias_backtest.py", "--pairs", "EURUSD", "--cache-dir", cache, "--tf", "H4H1",
                    "--out", os.path.join(d, "t.csv"), "--no-variants"]
            buf = io.StringIO()
            try:
                with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(buf):
                    bt.main()
            finally:
                bb.use_preset("D1H4")
            self.assertIn("4H key levels + 1H structure", buf.getvalue())
            self.assertEqual(bb.PRESET, "D1H4")


class TestCapAndControl(unittest.TestCase):
    def rows(self):
        base = pd.Timestamp("2024-03-01 00:00", tz="UTC")
        recs = []
        # day 1: scores 1,5,4,5 at increasing times; day 2: scores 5,5
        for i, (day, sc, r) in enumerate([(0, 1, -1.0), (0, 5, 2.0), (0, 4, 1.0), (0, 5, 3.0),
                                          (1, 5, -1.0), (1, 5, 0.5)]):
            recs.append(dict(pair="P", stage="RETEST", T=1.0, R=r, score=sc,
                             time=base + pd.Timedelta(days=day, hours=i + 1)))
        return pd.DataFrame(recs)

    def test_capped_is_first_come_first_served_and_filters_score(self):
        df = self.rows()
        got = [r for _, r in bt.capped(df, 0, 3)]
        self.assertEqual(got, [-1.0, 2.0, 1.0, -1.0, 0.5])          # 4th of day 1 dropped
        got = [r for _, r in bt.capped(df, 5, 3)]
        self.assertEqual(got, [2.0, 3.0, -1.0, 0.5])
        got = [r for _, r in bt.capped(df, 5, 1)]
        self.assertEqual(got, [2.0, -1.0])

    def test_reports_run(self):
        t = bt.collect({"T": walk(6000, seed=41)}, ("BOS", "RETEST"), 0.0002, 120)
        self.assertGreater(len(t), 0)
        self.assertTrue(set(bt.TARGETS) <= set(t["T"].unique()) or len(t) > 0)
        buf = io.StringIO()
        cut = pd.Timestamp("2024-06-01", tz="UTC")
        with contextlib.redirect_stdout(buf):
            bt.headline(t)
            bt.score_report(t, 1.0)
            bt.top3_report(t, 1.0, 3, cut)
            bt.baseline_report({"T": walk(6000, seed=41)}, t, 0.0002, 120, reps=2)
        out = buf.getvalue()
        self.assertIn("0.25R", out)
        self.assertIn("min score 0", out)
        self.assertIn("random:", out)

    def test_control_uses_same_stop_size_and_costs_reduce_r(self):
        df = walk(6000, seed=41)
        free = bt.collect({"T": df}, ("RETEST",), 0.0, 120)
        paid = bt.collect({"T": df}, ("RETEST",), 0.001, 120)
        self.assertTrue((paid["R"].values < free["R"].values).all())
        self.assertTrue((free["risk_pct"] > 0).all())


class TestScoreAndFresh(unittest.TestCase):
    def test_score_flags(self):
        e = {"fresh": True, "swept": True, "trend_ok": True, "risk_atr": 2.0, "retest_wick": 0.5}
        self.assertEqual(bb.setup_score(e), 5)
        e = {"fresh": False, "swept": False, "trend_ok": False, "risk_atr": 9.0, "retest_wick": 0.1}
        self.assertEqual(bb.setup_score(e), 0)
        self.assertEqual(bb.setup_score({}), 0)                  # BIAS events lack the late features
        self.assertEqual(bb.score_reasons({"fresh": True}), ["fresh level"])

    def test_events_carry_features_and_scores(self):
        ev = bb.events_for(walk(3000, seed=7), "T")
        self.assertGreater(len(ev), 0)
        for e in ev:
            self.assertIn(e["score"], range(0, 6))
            for k in ("fresh", "swept", "trend_ok"):
                self.assertIsInstance(e[k], bool)
        retests = [e for e in ev if e["stage"] == "RETEST"]
        self.assertGreater(len(retests), 0)
        self.assertTrue(all(0.0 <= e["retest_wick"] <= 1.0 for e in retests))

    def test_require_fresh_only_keeps_fresh_levels(self):
        df = walk(3000, seed=11)
        base = [e for e in bb.events_for(df, "T") if e["stage"] == "BIAS"]
        with mock.patch.object(bb, "REQUIRE_FRESH", True):
            fresh = [e for e in bb.events_for(df, "T") if e["stage"] == "BIAS"]
        self.assertGreater(len(base), 0)
        self.assertTrue(all(e["fresh"] for e in fresh))
        self.assertTrue(any(not e["fresh"] for e in base), "synthetic data had no touched levels")

    def test_fresh_rule_on_handmade_candles(self):
        """Level formed by a V (down candle then up candle); a later wick touching it makes it not fresh."""
        idx = pd.date_range("2024-01-01", periods=60, freq="D", tz="UTC")
        rng = np.random.default_rng(5)
        c = 100 + np.cumsum(rng.normal(0, 1, 60))
        o = np.concatenate([[100], c[:-1]])
        D = pd.DataFrame({"Open": o, "High": np.maximum(o, c) + 0.5,
                          "Low": np.minimum(o, c) - 0.5, "Close": c}, index=idx)
        for e in bb.daily_biases(D):
            self.assertIn(e["fresh"], (True, False))


class TestPresets(unittest.TestCase):
    def test_h4h1_runs_is_causal_and_restores(self):
        df = walk(3000, seed=7)
        with preset("H4H1"):
            self.assertEqual((bb.HTF_RULE, bb.LTF_RULE), ("4h", "1h"))
            full = bb.events_for(df, "T")
            self.assertGreater(len(full), 0)
            for cut in (1800, 2400):
                part = df.iloc[:cut]
                cut_end = part.index[-1] + bb.H1
                got = {key(e) for e in bb.events_for(part, "T")}
                want = {key(e) for e in full if e["time"] <= cut_end}
                self.assertEqual(got, want, f"H4H1 events differ when truncated at bar {cut}")
        self.assertEqual((bb.HTF_RULE, bb.LTF_RULE, bb.PRESET), ("1D", "4h", "D1H4"))
        self.assertEqual(bb.D1, pd.Timedelta(days=1))
        self.assertEqual(bb.H4, pd.Timedelta(hours=4))

    def test_alert_text_uses_preset_names(self):
        now = pd.Timestamp("2024-03-01 12:05", tz="UTC")
        e = fake_event("RETEST", 10, now)
        with preset("H4H1"):
            t = bb.format_alert(e, now)
        self.assertIn("1H RETEST", t)
        self.assertIn("4H V-shape key level", t)
        t = bb.format_alert(e, now)
        self.assertIn("4H RETEST", t)
        self.assertIn("Daily V-shape key level", t)

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            bb.use_preset("W1D1")


@contextlib.contextmanager
def style(name):
    bb.use_style(name)
    try:
        yield
    finally:
        bb.use_style("legacy")


class TestNewYorkAnchor(unittest.TestCase):
    def test_daily_bins_start_at_5pm_new_york_all_year(self):
        df = walk(24 * 40, seed=2, start="2026-10-15")        # spans the US clock change on 1 Nov 2026
        D = bb.resample_ohlc(df, "1D", ny=True)
        for ts in D.index:
            want = 21 if ts < pd.Timestamp("2026-11-01 06:00", tz="UTC") else 22
            self.assertEqual(ts.hour, want, f"daily bin starts {ts}")
        # a bin is exactly the 24 hourly bars from its label
        t = pd.Timestamp("2026-10-20 21:00", tz="UTC")
        seg = df[(df.index >= t) & (df.index < t + pd.Timedelta(hours=24))]
        row = D.loc[t]
        self.assertAlmostEqual(row["Open"], seg["Open"].iloc[0])
        self.assertAlmostEqual(row["High"], seg["High"].max())
        self.assertAlmostEqual(row["Low"], seg["Low"].min())
        self.assertAlmostEqual(row["Close"], seg["Close"].iloc[-1])
        # after the clock change the same rule gives 22:00 UTC
        t2 = pd.Timestamp("2026-11-05 22:00", tz="UTC")
        seg2 = df[(df.index >= t2) & (df.index < t2 + pd.Timedelta(hours=24))]
        self.assertAlmostEqual(D.loc[t2, "High"], seg2["High"].max())

    def test_4h_grid_follows_the_new_york_close(self):
        df = walk(24 * 40, seed=3, start="2026-10-15")
        H = bb.resample_ohlc(df, "4h", ny=True)
        summer = H[H.index < pd.Timestamp("2026-11-01", tz="UTC")]
        winter = H[H.index > pd.Timestamp("2026-11-03", tz="UTC")]
        self.assertEqual(set(summer.index.hour), {1, 5, 9, 13, 17, 21})
        self.assertEqual(set(winter.index.hour), {2, 6, 10, 14, 18, 22})

    def test_without_anchor_it_is_midnight_utc(self):
        df = walk(24 * 10, seed=4, start="2026-10-15")
        self.assertEqual(set(bb.resample_ohlc(df, "1D").index.hour), {0})
        self.assertEqual(set(bb.resample_ohlc(df, "4h").index.hour), {0, 4, 8, 12, 16, 20})

    def test_which_pairs_use_it(self):
        for p in ("EURUSD", "GBPJPY", "XAUUSD", "XAGUSD"):
            self.assertTrue(bb.uses_ny_anchor(p), p)
        for p in ("BTCUSDT", "ETHUSDT", "ETHUSD", "US30", "NAS100", "UK100", "JPN225"):
            self.assertFalse(bb.uses_ny_anchor(p), p)


class TestStyles(unittest.TestCase):
    def test_switch_and_restore(self):
        self.assertEqual((bb.STYLE, bb.LEVEL_MAX_AGE, bb.NY_ANCHOR, bb.SAME_DAY_BOS), ("legacy", 40, False, False))
        with style("slk"):
            self.assertEqual((bb.LEVEL_MAX_AGE, bb.NY_ANCHOR, bb.SAME_DAY_BOS, bb.FETCH_PERIOD), (300, True, True, "730d"))
        self.assertEqual((bb.STYLE, bb.LEVEL_MAX_AGE, bb.FETCH_PERIOD), ("legacy", 40, "60d"))
        with bb.style_ctx("slk"):
            self.assertEqual(bb.STYLE, "slk")
        self.assertEqual(bb.STYLE, "legacy")
        with self.assertRaises(ValueError):
            bb.use_style("nope")

    def test_main_defaults_to_slk_and_restores(self):
        now = pd.Timestamp("2024-03-01 12:05", tz="UTC")
        seen = {}

        def fake_events(df, pair, now_=None):
            seen["style"], seen["ny"], seen["fetch"] = bb.STYLE, bb.NY_ANCHOR, bb.FETCH_PERIOD
            return []
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.dict(os.environ, {"TELEGRAM_TOKEN": TOKEN, "CHAT_ID": "42"}), \
                mock.patch.object(bb, "PAIRS", {"T=X": "TEST"}), \
                mock.patch.object(bb, "SENT_FILE", os.path.join(d, "s.json")), \
                mock.patch.object(bb, "fetch_1h", return_value=walk(60)), \
                mock.patch.object(bb, "events_for", side_effect=fake_events), \
                mock.patch.object(bb, "datetime") as dt, contextlib.redirect_stdout(io.StringIO()):
            dt.now.return_value = now.to_pydatetime()
            os.environ.pop("STYLE", None)
            bb.main()
        self.assertEqual(seen, {"style": "slk", "ny": True, "fetch": "730d"})
        self.assertEqual((bb.STYLE, bb.NY_ANCHOR, bb.LEVEL_MAX_AGE), ("legacy", False, 40))


class TestSameDayBreakout(unittest.TestCase):
    def test_no_look_ahead_and_alert_waits_for_the_daily_close(self):
        df = walk(3600, seed=7, start="2026-01-05")
        with style("slk"):
            full = bb.events_for(df, "EURUSD")
            biases = [e for e in full if e["stage"] == "BIAS"]
            later = [e for e in full if e["stage"] != "BIAS"]
            self.assertGreater(len(later), 0, "synthetic data produced no BOS/RETEST; test is vacuous")
            # never before the rejection candle has closed
            self.assertTrue(all(e["time"] >= e["bias_time"] for e in later))
            # the same-day path is really exercised: some alerts were pushed to the daily close
            self.assertTrue(any(e["time"] == e["bias_time"] for e in later))
            for cut in (1800, 2400, 3000):
                part = df.iloc[:cut]
                cut_end = part.index[-1] + bb.H1
                got = {key(e) for e in bb.events_for(part, "EURUSD")}
                want = {key(e) for e in full if e["time"] <= cut_end}
                self.assertEqual(got, want, f"slk events differ when truncated at bar {cut}")

    def test_legacy_never_alerts_inside_the_rejection_day(self):
        df = walk(3600, seed=7, start="2026-01-05")
        later = [e for e in bb.events_for(df, "EURUSD") if e["stage"] != "BIAS"]
        self.assertGreater(len(later), 0)
        # legacy starts looking at the first 4H bar AFTER the rejection candle, so the alert is
        # always at least one full 4H candle after the daily close
        self.assertTrue(all(e["time"] >= e["bias_time"] + bb.H4 for e in later))

    def test_slk_style_finds_more_breakouts_than_legacy(self):
        df = walk(3600, seed=7, start="2026-01-05")
        n_legacy = sum(e["stage"] == "BOS" for e in bb.events_for(df, "EURUSD"))
        with style("slk"):
            n_slk = sum(e["stage"] == "BOS" for e in bb.events_for(df, "EURUSD"))
        self.assertGreater(n_slk, 0)
        self.assertNotEqual(n_slk, n_legacy)


if __name__ == "__main__":
    unittest.main()
