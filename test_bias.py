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


def fake_event(stage, minutes_ago, now, invalidated=False, direction="BUY"):
    return {"pair": "TEST", "stage": stage, "direction": direction, "kind": "V-shape",
            "level": 100.0, "key_level": 99.0, "came": "above", "invalidation": 95.0,
            "time": now - pd.Timedelta(minutes=minutes_ago), "invalidated": invalidated}


class TestMainFlow(unittest.TestCase):
    def run_main(self, events, now, state):
        df = walk(60)
        sent = []
        with mock.patch.dict(os.environ, {"TELEGRAM_TOKEN": TOKEN, "CHAT_ID": "42"}), \
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

    def test_failed_send_is_retried_next_run(self):
        now = pd.Timestamp("2024-03-01 12:05", tz="UTC")
        events = [fake_event("RETEST", 20, now)]
        with tempfile.TemporaryDirectory() as d:
            state = os.path.join(d, "sent.json")
            df = walk(60)
            with mock.patch.dict(os.environ, {"TELEGRAM_TOKEN": TOKEN, "CHAT_ID": "42"}), \
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
            for s in ("HEADLINE", "BY PAIR", "VARIANTS", "assumptions"):
                self.assertIn(s, text)
            self.assertTrue(os.path.exists(out))
            self.assertFalse(bb.REQUIRE_SWEEP or bb.REQUIRE_D1_TREND)   # switches restored


if __name__ == "__main__":
    unittest.main()
