"""
Unit tests for memoization cache.
Verifies MD5 hashing, snapping-based cache hits, CSV persistence, and reload.
"""

import os
import shutil
import tempfile
import unittest
from memo.memo_cache_180nm import MemoCache, compute_param_md5


class TestMemoCache(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="cache_test_")
        self.csv_path = os.path.join(self.test_dir, "test_cache.csv")
        self.cache = MemoCache(cache_file=self.csv_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cache_miss_then_store_then_hit(self):
        raw_params = {
            "Wdiff": 3.4,
            "Wnb": 160.0,
            "Wpo": 5.5,
            "Wbp": 30.0,
            "Cc": 1.5,
        }
        # 1. First query should be a miss
        is_hit, snapped, metrics = self.cache.query(raw_params)
        self.assertFalse(is_hit)
        self.assertIsNone(metrics)

        # 2. Store simulation result
        sim_metrics = {
            "gain_db": 52.86,
            "ugbw": 15.1e6,
            "pm": 28.8,
            "f3db": 56.2e3,
            "pdc": 1.9e-4,
            "n_sat": 8,
            "all_sat": True,
            "is_valid": True,
        }
        self.cache.store(snapped, sim_metrics)

        # 3. Second query with exact same params should be a hit
        is_hit2, snapped2, metrics2 = self.cache.query(raw_params)
        self.assertTrue(is_hit2)
        self.assertIsNotNone(metrics2)
        self.assertEqual(metrics2["n_sat"], 8)
        self.assertAlmostEqual(metrics2["gain_db"], 52.86, places=2)

    def test_unsnapped_params_hit_same_cache_entry(self):
        pt1 = {"Wdiff": 3.4, "Wnb": 160.0, "Wpo": 5.5, "Wbp": 30.0, "Cc": 1.5}
        # Slightly off-grid parameters that snap to the exact same values
        pt2 = {"Wdiff": 3.41, "Wnb": 160.2, "Wpo": 5.49, "Wbp": 29.8, "Cc": 1.51}

        is_hit, snapped, _ = self.cache.query(pt1)
        self.assertFalse(is_hit)

        sim_metrics = {
            "gain_db": 52.86,
            "ugbw": 15.1e6,
            "pm": 28.8,
            "f3db": 56.2e3,
            "pdc": 1.9e-4,
            "n_sat": 8,
            "all_sat": True,
            "is_valid": True,
        }
        self.cache.store(snapped, sim_metrics)

        # Query with pt2
        is_hit2, snapped2, metrics2 = self.cache.query(pt2)
        self.assertTrue(is_hit2)
        self.assertEqual(snapped, snapped2)

    def test_persistence_and_reload(self):
        pt = {"Wdiff": 3.4, "Wnb": 160.0, "Wpo": 5.5, "Wbp": 30.0, "Cc": 1.5}
        is_hit, snapped, _ = self.cache.query(pt)
        sim_metrics = {
            "gain_db": 52.86,
            "ugbw": 15.1e6,
            "pm": 28.8,
            "f3db": 56.2e3,
            "pdc": 1.9e-4,
            "n_sat": 8,
            "all_sat": True,
            "is_valid": True,
        }
        self.cache.store(snapped, sim_metrics)

        # Verify CSV exists and is non-empty
        self.assertTrue(os.path.exists(self.csv_path))
        self.assertGreater(os.path.getsize(self.csv_path), 0)

        # Create a new MemoCache instance pointing to the same CSV file
        cache2 = MemoCache(cache_file=self.csv_path)
        is_hit2, snapped2, metrics2 = cache2.query(pt)
        self.assertTrue(is_hit2)
        self.assertEqual(metrics2["n_sat"], 8)

    def test_stats_tracking(self):
        pt = {"Wdiff": 3.4, "Wnb": 160.0, "Wpo": 5.5, "Wbp": 30.0, "Cc": 1.5}
        self.cache.query(pt)  # miss
        self.cache.store(pt, {"gain_db": 50, "ugbw": 1e6, "pm": 60, "f3db": 1e4, "pdc": 1e-4, "n_sat": 8, "all_sat": True, "is_valid": True})
        self.cache.query(pt)  # hit
        self.cache.query(pt)  # hit

        stats = self.cache.get_stats()
        self.assertEqual(stats["total_queries"], 3)
        self.assertEqual(stats["cache_hits"], 2)
        self.assertEqual(stats["new_sims"], 1)
        self.assertAlmostEqual(stats["hit_fraction"], 2.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
