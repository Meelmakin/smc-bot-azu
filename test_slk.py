"""Run: python -m unittest test_slk -v   (no internet needed)"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import slk_score as ss
from test_bias import walk

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "slk_alerts.csv")


def flat(base, n=600, start="2026-10-01"):
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame({"Open": base, "High": base, "Low": base, "Close": base}, index=idx)


def setbar(df, ts, **kw):
    t = pd.Timestamp(ts, tz="UTC")
    for k, v in kw.items():
        df.loc[t, k.capitalize()] = v


def alert(direction="BUY", bo=np.nan, alert_utc="2026-10-11 02:00", rej="2026-10-10 00:00", hours=24):
    return pd.Series({"id": "t1", "pair": "TEST", "direction": direction, "kind": "external",
                      "alert_utc": pd.Timestamp(alert_utc, tz="UTC"),
                      "rej_open_utc": pd.Timestamp(rej, tz="UTC"), "htf_hours": hours, "bo_level": bo})


class TestCsv(unittest.TestCase):
    def test_file_is_clean(self):
        a = ss.load_alerts(CSV)
        self.assertEqual(len(a), 28)
        self.assertEqual(a["id"].nunique(), 28)
        for c in ("rej_open_utc", "bo_time_utc", "alert_utc"):
            self.assertFalse(a[c].isna().any(), c)
        self.assertTrue(set(a["direction"]) <= {"BUY", "SELL"})
        self.assertTrue((a["alert_utc"] > a["rej_open_utc"]).all())
        self.assertTrue((a["alert_utc"] >= a["bo_time_utc"]).all())
        self.assertTrue(set(a["htf_hours"]) <= {24, 168})
        self.assertEqual(a.loc[a["id"] == "s01", "alert_utc"].iloc[0], pd.Timestamp("2026-10-19 04:00", tz="UTC"))

    def test_every_pair_has_a_ticker(self):
        for p in ss.load_alerts(CSV)["pair"].unique():
            self.assertIsNotNone(ss.ticker_for(p), p)


class TestScoring(unittest.TestCase):
    def test_entry_a_buy(self):
        df = flat(100.0)
        setbar(df, "2026-10-10 05:00", low=98.0)          # rejection low -> stop 98, risk 2 from ref 100
        setbar(df, "2026-10-11 03:00", high=101.0)
        setbar(df, "2026-10-11 04:00", high=102.0)
        setbar(df, "2026-10-11 05:00", high=104.5)
        r = ss.score_alert(alert(), df, cost_pct=0.0, maxbars=100)
        self.assertEqual(r["status"], "ok")
        self.assertAlmostEqual(r["stop"], 98.0)
        self.assertAlmostEqual(r["ref"], 100.0)
        self.assertAlmostEqual(r["risk_pct"], 2.0)
        self.assertAlmostEqual(r["A_0.25"], 0.25)
        self.assertAlmostEqual(r["A_0.5"], 0.5)
        self.assertAlmostEqual(r["A_1"], 1.0)
        self.assertAlmostEqual(r["A_2"], 2.0)
        self.assertAlmostEqual(r["A_3"], 0.0)              # never reached: timeout at the entry price

    def test_entry_a_sell_stopped_out_and_costs(self):
        df = flat(100.0)
        setbar(df, "2026-10-10 05:00", high=102.0)         # SELL stop 102
        setbar(df, "2026-10-11 03:00", high=102.5)         # stopped immediately
        r = ss.score_alert(alert("SELL"), df, cost_pct=0.0, maxbars=100)
        self.assertAlmostEqual(r["A_1"], -1.0)
        r2 = ss.score_alert(alert("SELL"), df, cost_pct=0.001, maxbars=100)
        self.assertLess(r2["A_1"], -1.0)                   # costs make it worse

    def test_no_lookahead_uses_bar_before_alert(self):
        df = flat(100.0)
        setbar(df, "2026-10-10 05:00", low=98.0)
        setbar(df, "2026-10-11 01:00", close=100.5)        # last bar before the alert at 02:00
        setbar(df, "2026-10-11 02:00", close=130.0)        # must NOT be the reference
        r = ss.score_alert(alert(), df, cost_pct=0.0, maxbars=50)
        self.assertAlmostEqual(r["ref"], 100.5)

    def test_entry_b_pullback_fill(self):
        df = flat(102.0)
        setbar(df, "2026-10-10 05:00", low=98.0)           # stop 98, ref 102
        setbar(df, "2026-10-11 05:00", low=100.5, close=101.5)   # pullback through L=101 -> fill at 101
        setbar(df, "2026-10-11 06:00", high=103.0)
        setbar(df, "2026-10-11 07:00", high=104.0)
        r = ss.score_alert(alert(bo=101.0), df, cost_pct=0.0, maxbars=100)
        self.assertTrue(r["B_filled"])
        self.assertAlmostEqual(r["B_0.25"], 0.25)
        self.assertAlmostEqual(r["B_0.5"], 0.5)
        self.assertAlmostEqual(r["B_1"], 1.0)
        self.assertAlmostEqual(r["B_2"], (102.0 - 101.0) / 3.0, places=6)   # timeout, marked to market

    def test_entry_b_no_pullback_means_no_trade(self):
        df = flat(102.0)
        setbar(df, "2026-10-10 05:00", low=98.0)
        r = ss.score_alert(alert(bo=95.0), df, cost_pct=0.0, maxbars=100)     # level never retouched
        self.assertFalse(r["B_filled"])
        self.assertNotIn("B_1", r)

    def test_levels_ignored_for_indices_and_mismatched_prices(self):
        df = flat(102.0)
        setbar(df, "2026-10-10 05:00", low=98.0)
        a = alert(bo=101.0)
        a["pair"] = "NAS100"                               # index: CFD vs Yahoo price basis differs
        r = ss.score_alert(a, df, cost_pct=0.0, maxbars=100)
        self.assertEqual(r["status"], "ok")
        self.assertIn("A_1", r)                            # entry A still scored
        self.assertNotIn("B_filled", r)
        self.assertNotIn("moved_R", r)
        self.assertIn("level_note", r)
        a = alert(bo=1.3)                                  # fx pair but level is nowhere near Yahoo's 102
        a["bo_time_utc"] = pd.Timestamp("2026-10-11 00:00", tz="UTC")
        r = ss.score_alert(a, df, cost_pct=0.0, maxbars=100)
        self.assertNotIn("B_filled", r)
        self.assertIn("away from Yahoo price", r["level_note"])

    def test_missing_data_is_reported_not_crashed(self):
        df = flat(100.0)
        r = ss.score_alert(alert(rej="2030-01-01 00:00", alert_utc="2030-01-02 00:00"), df)
        self.assertNotEqual(r["status"], "ok")
        df = flat(100.0)
        setbar(df, "2026-10-10 05:00", low=98.0)
        r = ss.score_alert(alert(alert_utc="2035-01-01 00:00"), df)
        self.assertEqual(r["status"], "no price data at alert time")

    def test_price_already_beyond_stop(self):
        df = flat(100.0)
        setbar(df, "2026-10-10 05:00", low=105.0)          # BUY stop above the price
        r = ss.score_alert(alert(), df)
        self.assertIn("beyond the stop", r["status"])


class TestStats(unittest.TestCase):
    def test_wilson(self):
        lo, hi = ss.wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=6)
        self.assertAlmostEqual(hi, 0.2775, places=3)
        lo, hi = ss.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)

    def test_summarize(self):
        s = ss.summarize([1.0, -1.0, 2.0, None])
        self.assertEqual(s["n"], 3)
        self.assertEqual(s["wins"], 2)
        self.assertAlmostEqual(s["exp"], 2 / 3)
        self.assertAlmostEqual(s["pf"], 3.0)
        self.assertIsNone(ss.summarize([]))

    def test_asset_class(self):
        self.assertEqual(ss.asset_class("BTCUSDT"), "crypto")
        self.assertEqual(ss.asset_class("ETHUSD"), "crypto")
        self.assertEqual(ss.asset_class("UK100"), "index")
        self.assertEqual(ss.asset_class("USOIL"), "commodity")
        self.assertEqual(ss.asset_class("GBPJPY"), "fx")


class TestEndToEnd(unittest.TestCase):
    def fake_prices(self, pair, cache_dir="x", refresh=False):
        return walk(3600, seed=abs(hash(pair)) % 1000, start="2026-07-01", base=100.0, vol=0.003)

    def run_main(self, prices):
        with tempfile.TemporaryDirectory() as d:
            argv = ["slk_score.py", "--alerts", CSV, "--out", os.path.join(d, "o.csv"), "--reps", "20"]
            buf = io.StringIO()
            with mock.patch.object(sys, "argv", argv), mock.patch.object(ss, "load_prices", prices), \
                    contextlib.redirect_stdout(buf):
                ss.main()
            self.assertTrue(os.path.exists(os.path.join(d, "o.csv")))
        return buf.getvalue()

    def test_full_run(self):
        out = self.run_main(self.fake_prices)
        for s in ("EVERY ALERT", "ENTRY A", "ENTRY B", "CONTROL", "INDEPENDENCE", "BY KIND", "WARNING: n="):
            self.assertIn(s, out)

    def test_pair_without_data_is_skipped(self):
        def prices(pair, cache_dir="x", refresh=False):
            if pair == "BTCUSDT":
                raise RuntimeError("no data")
            return self.fake_prices(pair)
        out = self.run_main(prices)
        self.assertIn("skip BTCUSDT", out)
        self.assertIn("not scored: s04 BTCUSDT", out)
        self.assertIn("ENTRY A", out)


if __name__ == "__main__":
    unittest.main()
