"""Run: python -m unittest test_retest -v   (no internet needed)"""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import bias_bot as bb
import bias_retest as rt
from test_bias import walk

N = 200


def flat(base):
    idx = pd.date_range("2026-10-01", periods=N, freq="h", tz="UTC")
    return pd.DataFrame({"Open": base, "High": base, "Low": base, "Close": base}, index=idx)


def setbar(df, i, o, h, l, c):
    df.iloc[i] = [o, h, l, c]


def event(df, direction="BUY", level=100.0, inval=95.0, at=50):
    return {"pair": "TEST", "direction": direction, "level": level, "invalidation": inval,
            "time": df.index[at], "score": 3, "kind": "A-shape", "stage": "BOS"}


def buy_case():
    df = flat(105.0)
    setbar(df, 60, 103.0, 103.0, 99.8, 100.6)        # pulls back through 100 and closes back above it
    setbar(df, 61, 100.6, 101.6, 100.5, 101.5)       # reaches 1R
    setbar(df, 62, 101.5, 102.5, 101.4, 102.4)       # reaches 2R
    setbar(df, 63, 102.4, 104.0, 102.3, 103.9)       # reaches 3R
    return df


class TestEntry(unittest.TestCase):
    A1 = np.full(N, 1.0)

    def sim(self, df, ev=None, **kw):
        kw.setdefault("cost_pct", 0.0)
        kw.setdefault("maxbars", 100)
        return rt.simulate(ev or event(df), df, self.A1, **kw)

    def test_buy_retest_entry_and_targets(self):
        r = self.sim(buy_case())
        self.assertEqual(r["status"], "entered")
        self.assertEqual(r["j"], 60)
        self.assertAlmostEqual(r["ref"], 100.6)
        self.assertAlmostEqual(r["stop"], 99.7)            # wick 99.8 minus 0.1 ATR
        self.assertAlmostEqual(r["R_1"], 1.0)
        self.assertAlmostEqual(r["R_2"], 2.0)
        self.assertAlmostEqual(r["R_3"], 3.0)
        self.assertAlmostEqual(r["R_5"], (105.0 - 100.6) / 0.9, places=6)   # timeout, marked to market
        self.assertEqual(r["entry_time"], buy_case().index[60] + pd.Timedelta(hours=1))

    def test_sell_side_mirrors(self):
        df = flat(95.0)
        setbar(df, 60, 97.0, 100.2, 97.0, 99.4)          # spikes above 100, closes back below
        setbar(df, 61, 99.4, 99.5, 98.4, 98.5)           # reaches 1R (stop 100.3, risk 0.9)
        r = self.sim(df, event(df, "SELL", level=100.0, inval=105.0))
        self.assertEqual(r["status"], "entered")
        self.assertAlmostEqual(r["stop"], 100.3)
        self.assertAlmostEqual(r["R_1"], 1.0)

    def test_stop_hit_counts_minus_one_and_costs_apply(self):
        df = flat(105.0)
        setbar(df, 60, 103.0, 103.0, 99.8, 100.6)
        setbar(df, 61, 100.6, 100.7, 99.5, 99.6)         # takes out the stop at 99.7
        r = self.sim(df)
        self.assertAlmostEqual(r["R_2"], -1.0)
        r2 = self.sim(df, cost_pct=0.001)
        self.assertAlmostEqual(r2["R_2"], -1.0 - 0.001 * 100.6 / 0.9, places=6)

    def test_bias_invalidated_before_retest(self):
        df = flat(105.0)
        setbar(df, 55, 105.0, 105.0, 93.0, 94.0)         # closes beyond the rejection extreme first
        setbar(df, 60, 103.0, 103.0, 99.8, 100.6)
        r = self.sim(df)
        self.assertEqual(r["status"], "bias invalidated before a retest")

    def test_no_retest_within_the_wait(self):
        df = flat(105.0)
        setbar(df, 120, 103.0, 103.0, 99.8, 100.6)       # too late: wait is 48 hours from bar 50
        self.assertEqual(self.sim(df)["status"], "no retest within the wait")

    def test_touch_without_closing_back_is_not_an_entry(self):
        df = flat(105.0)
        setbar(df, 60, 103.0, 103.0, 99.8, 99.9)         # touched but closed below the level
        self.assertEqual(self.sim(df)["status"], "no retest within the wait")

    def test_tight_stop_is_skipped(self):
        df = flat(105.0)
        setbar(df, 60, 103.0, 103.0, 100.0, 100.1)       # stop 99.9 -> risk 0.2 < 0.3 ATR
        self.assertEqual(self.sim(df)["status"], "stop too tight")

    def test_no_look_ahead(self):
        a, b = buy_case(), buy_case()
        for i in range(61, N):                            # wreck everything after the entry candle
            setbar(b, i, 50.0, 50.0, 50.0, 50.0)
        ra, rb = self.sim(a), self.sim(b)
        for k in ("status", "j", "ref", "stop", "risk_pct"):
            self.assertEqual(ra[k], rb[k], k)
        self.assertNotEqual(ra["R_2"], rb["R_2"])         # only the outcome changes
        c = buy_case()
        setbar(c, 60, 103.0, 103.0, 99.8, 100.6)
        setbar(c, 59, 103.0, 103.0, 99.8, 99.0)           # an earlier touch that closed below: ignored
        self.assertEqual(self.sim(c)["j"], 60)


class TestEndToEnd(unittest.TestCase):
    def tearDown(self):
        bb.use_preset("D1H4")
        bb.use_style("legacy")

    def data(self):
        return {"EURUSD": walk(6000, seed=21, start="2025-01-01", base=1.1, vol=0.0015),
                "XAUUSD": walk(6000, seed=22, start="2025-01-01", base=2000.0, vol=0.002)}

    def test_collect_and_report(self):
        data = self.data()
        for style in ("legacy", "slk"):
            bb.use_style(style)
            t = rt.collect(data, "BOS", cost_pct=0.0002)
            self.assertGreater(len(t), 20)
            self.assertIn("entered", set(t["status"]))
            ent = t[t["status"] == "entered"]
            self.assertTrue((ent["risk_atr"] >= rt.MIN_RISK_ATR - 1e-9).all())
            self.assertTrue((ent["entry_time"] > ent["alert_time"]).all())   # never before the alert
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rt.report(t, data, pd.Timestamp("2025-06-01", tz="UTC"), reps=5)
            out = buf.getvalue()
            for s in ("HEADLINE", "IN-SAMPLE vs OUT-OF-SAMPLE", "BY MINIMUM BIAS SCORE", "ASSET CLASS",
                      "CONTROL", "stop size:"):
                self.assertIn(s, out)

    def test_main_runs_and_restores(self):
        data = self.data()
        with tempfile.TemporaryDirectory() as d:
            argv = ["bias_retest.py", "--pairs", "EURUSD,XAUUSD", "--reps", "5", "--out", os.path.join(d, "o.csv")]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(rt.bt, "load", side_effect=lambda name, *a: data[name]), \
                    contextlib.redirect_stdout(io.StringIO()) as buf:
                rt.main()
            self.assertTrue(os.path.exists(os.path.join(d, "o.csv")))
        self.assertIn("style=slk", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
