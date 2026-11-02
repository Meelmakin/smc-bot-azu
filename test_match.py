"""Run: python -m unittest test_match -v   (no internet needed)"""
import contextlib
import io
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import bias_bot as bb
import match_check as mc
from test_bias import walk


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = walk(3600, seed=7, start="2026-01-05", base=1.1, vol=0.003)
        cls.ev = mc.our_events(cls.df, "EURUSD")
        cls.bos = [e for e in cls.ev if e["stage"] == "BOS"]
        cls.bias = [e for e in cls.ev if e["stage"] == "BIAS"]

    def tearDown(self):
        self.assertEqual((bb.STYLE, bb.PRESET), ("legacy", "D1H4"))      # settings always restored

    def alert(self, e, **over):
        a = {"id": "t1", "pair": "EURUSD", "direction": e["direction"], "kind": "external", "htf_hours": 24,
             "rej_open_utc": e["bias_open"], "key_level": e["key_level"], "bo_level": e["level"],
             "bo_time_utc": e["bar_open"], "alert_utc": e["time"]}
        a.update(over)
        return pd.Series(a)


class TestMatching(Base):
    def setUp(self):
        self.good = next(e for e in self.bos if e["trend_ok"])

    def test_events_carry_the_times_needed(self):
        self.assertGreater(len(self.bos), 0)
        for e in self.bos:
            self.assertIn("bar_open", e)
            self.assertIn("bias_open", e)
            self.assertLessEqual(e["bar_open"], e["time"])

    def test_exact_copy_matches(self):
        r = mc.match_alert(self.alert(self.good), self.ev)
        self.assertEqual((r["status"], r["notes"]), ("MATCH", []))

    def test_each_difference_is_reported(self):
        e = self.good
        r = mc.match_alert(self.alert(e, key_level=e["key_level"] * 1.03), self.ev)
        self.assertEqual(r["status"], "PARTIAL")
        self.assertTrue(r["notes"][0].startswith("key level"))
        r = mc.match_alert(self.alert(e, bo_level=e["level"] * 0.97), self.ev)
        self.assertTrue(any(n.startswith("breakout level") for n in r["notes"]))
        r = mc.match_alert(self.alert(e, bo_time_utc=e["bar_open"] + pd.Timedelta(hours=8)), self.ev)
        self.assertTrue(any(n.startswith("breakout candle") for n in r["notes"]))
        r = mc.match_alert(self.alert(e, alert_utc=e["time"] + pd.Timedelta(hours=5)), self.ev)
        self.assertTrue(any(n.startswith("alert time") for n in r["notes"]))

    def test_small_level_differences_are_tolerated(self):
        e = self.good
        r = mc.match_alert(self.alert(e, key_level=e["key_level"] * 1.0005, bo_level=e["level"] * 0.9995), self.ev)
        self.assertEqual(r["status"], "MATCH")

    def test_wrong_day_or_direction_is_no_rejection(self):
        e = self.good
        r = mc.match_alert(self.alert(e, rej_open_utc=e["bias_open"] + pd.Timedelta(days=40)), self.ev)
        self.assertEqual(r["status"], "NO REJECTION")
        other = "SELL" if e["direction"] == "BUY" else "BUY"
        r = mc.match_alert(self.alert(e, direction=other), self.ev)
        self.assertEqual(r["status"], "NO REJECTION")
        self.assertIn("ours_nearby", r)

    def test_trend_disagreement_is_flagged(self):
        e = next((x for x in self.bos if not x["trend_ok"]), None)
        self.assertIsNotNone(e, "synthetic data had no counter-trend breakout")
        r = mc.match_alert(self.alert(e), self.ev)
        self.assertEqual(r["status"], "PARTIAL")
        self.assertTrue(any(n.startswith("trend") for n in r["notes"]))

    def test_rejection_without_breakout(self):
        have = {e["bias_open"] for e in self.bos}
        e = next((b for b in self.bias if b["open"] not in have), None)
        self.assertIsNotNone(e, "every rejection had a breakout in the synthetic data")
        a = self.alert(self.good, direction=e["direction"], rej_open_utc=e["open"], key_level=e["level"])
        self.assertEqual(mc.match_alert(a, self.ev)["status"], "REJECTION ONLY")

    def test_types_we_do_not_build(self):
        self.assertEqual(mc.match_alert(self.alert(self.good, kind="internal"), self.ev)["status"], "NOT BUILT")
        self.assertEqual(mc.match_alert(self.alert(self.good, kind="range"), self.ev)["status"], "NOT BUILT")

    def test_filters_do_not_hide_events(self):
        with mock.patch.object(bb, "REQUIRE_D1_TREND", True), mock.patch.object(bb, "REQUIRE_FRESH", True):
            ev = mc.our_events(self.df, "EURUSD")
            self.assertTrue(mc.bb.REQUIRE_D1_TREND)                      # restored inside the patch
        self.assertEqual(len(ev), len(self.ev))


class TestMain(Base):
    def run_main(self, loader):
        argv = ["match_check.py", "--no-recent"]
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.object(mc.ss, "load_prices", loader), \
                contextlib.redirect_stdout(buf):
            mc.main()
        return buf.getvalue()

    def test_full_run_on_the_alert_file(self):
        out = self.run_main(lambda pair, cache_dir="x", refresh=False:
                            walk(3600, seed=abs(hash(pair)) % 100, start="2026-07-01", base=100.0, vol=0.003))
        for s in ("EVERY ALERT", "SUMMARY", "reproduced exactly:", "what differed"):
            self.assertIn(s, out)
        self.assertIn("NOT BUILT", out)                                  # internal / range rows

    def test_missing_pair_is_reported_not_fatal(self):
        def loader(pair, cache_dir="x", refresh=False):
            if pair == "GBPJPY":
                raise RuntimeError("no data")
            return walk(3600, seed=3, start="2026-07-01", base=100.0, vol=0.003)
        out = self.run_main(loader)
        self.assertIn("skip GBPJPY", out)
        self.assertIn("NO DATA", out)


class TestWeekly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = walk(24 * 700, seed=7, start="2024-10-25", base=1.1, vol=0.002)
        cls.ev = mc.our_events(cls.df, "EURUSD", "W1D1")
        cls.bos = [e for e in cls.ev if e["stage"] == "BOS"]

    def alert(self, e, **over):
        a = {"id": "w1", "pair": "EURUSD", "direction": e["direction"], "kind": "external", "htf_hours": 168,
             "rej_open_utc": e["bias_open"], "key_level": e["key_level"], "bo_level": e["level"],
             "bo_time_utc": e["bar_open"], "alert_utc": e["time"]}
        a.update(over)
        return pd.Series(a)

    def test_weekly_alerts_are_now_compared(self):
        self.assertEqual(mc.preset_for(168), "W1D1")
        self.assertEqual(mc.preset_for(24), "D1H4")
        self.assertGreater(len(self.bos), 0, "no weekly breakouts in the synthetic data")
        e = next(x for x in self.bos if x["trend_ok"])
        r = mc.match_alert(self.alert(e), self.ev)
        self.assertEqual((r["status"], r["notes"]), ("MATCH", []))
        r = mc.match_alert(self.alert(e, bo_level=e["level"] * 1.05), self.ev)
        self.assertTrue(any(n.startswith("breakout level") for n in r["notes"]))

    def test_weekly_trend_note_names_w1(self):
        e = next((x for x in self.bos if not x["trend_ok"]), None)
        if e is None:
            self.skipTest("no counter-trend weekly breakout in this synthetic data")
        r = mc.match_alert(self.alert(e), self.ev)
        self.assertTrue(any("NOT aligned with W1" in n for n in r["notes"]))

    def test_restores_settings(self):
        self.assertEqual((bb.STYLE, bb.PRESET), ("legacy", "D1H4"))


class TestRecent(unittest.TestCase):
    def test_lists_only_what_the_live_settings_would_send(self):
        df = walk(100, seed=1, start="2026-10-20")
        end = df.index[-1] + pd.Timedelta(hours=1)

        def ev(hours_ago, stage="BOS", direction="BUY", trend_ok=True, daily_ok=True, invalidated=False, score=3):
            return {"pair": "US100", "stage": stage, "direction": direction, "tf": "D1→H4", "trend_ok": trend_ok,
                    "daily_ok": daily_ok, "invalidated": invalidated, "score": score,
                    "time": end - pd.Timedelta(hours=hours_ago)}
        fake = [ev(2), ev(5, "RETEST", "SELL"), ev(300), ev(3, invalidated=True), ev(4, daily_ok=False),
                ev(6, "BIAS")]
        buf = io.StringIO()
        with mock.patch.object(mc.bb, "PAIRS", {"NQ=F": "US100"}), mock.patch.object(mc.bt, "load", return_value=df), \
                mock.patch.object(mc.bb, "events_all", return_value=fake), \
                mock.patch.object(mc.bb, "ALERT_STAGES", {"BOS", "RETEST"}), \
                mock.patch.object(mc.bb, "REQUIRE_DAILY_ALIGN", True), mock.patch.object(mc.bb, "PROFILE", "all"), \
                mock.patch.dict("os.environ", {"STRICT_MODE": "0"}), contextlib.redirect_stdout(buf):
            mc.recent_report(72)
        out = buf.getvalue()
        self.assertIn("would have sent 2 alert(s)", out)          # the old, invalidated, misaligned and BIAS ones are out
        self.assertIn("NAS100", out)                              # shown with the broker-style name
        self.assertIn("scan job is not running or is failing", out)
        self.assertEqual((mc.bb.STYLE, mc.bb.PRESET), ("legacy", "D1H4"))

    def test_nothing_to_send_says_so(self):
        df = walk(100, seed=1, start="2026-10-20")
        with mock.patch.object(mc.bb, "PAIRS", {"NQ=F": "US100"}), mock.patch.object(mc.bt, "load", return_value=df), \
                mock.patch.object(mc.bb, "events_all", return_value=[]), \
                mock.patch.dict("os.environ", {"STRICT_MODE": "0"}), contextlib.redirect_stdout(io.StringIO()) as buf:
            mc.recent_report(72)
        self.assertIn("would have sent 0 alert(s)", buf.getvalue())
        self.assertIn("Nothing qualified", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
