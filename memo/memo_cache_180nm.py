"""
Persistent memoization cache for 180nm OpAmp simulation results.
Keyed by MD5 hash of canonical snapped parameters.
Thread-safe and persisted to memo/memo_cache_180nm.csv.
"""

import os
import csv
import json
import hashlib
import time
import logging
from typing import Dict, Any, Tuple, Optional
from circuits.two_stage_opamp_paper import snap_parameters

logger = logging.getLogger("MemoCache")

DEFAULT_CACHE_CSV = "/home/STUDENT/Nitish_Majorproject_Sem7/memo/memo_cache_180nm.csv"

FIELDNAMES = [
    "md5_hash",
    "Wdiff",
    "Wnb",
    "Wpo",
    "Wbp",
    "Cc",
    "gain_db",
    "ugbw",
    "pm",
    "f3db",
    "pdc",
    "n_sat",
    "all_sat",
    "is_valid",
    "timestamp",
]


def compute_param_md5(snapped_params: Dict[str, float]) -> str:
    """Computes a deterministic MD5 hash for a snapped parameter dictionary."""
    canonical_dict = {
        "Wdiff": round(float(snapped_params["Wdiff"]), 4),
        "Wnb":   round(float(snapped_params["Wnb"]), 4),
        "Wpo":   round(float(snapped_params["Wpo"]), 4),
        "Wbp":   round(float(snapped_params["Wbp"]), 4),
        "Cc":    round(float(snapped_params["Cc"]), 4),
    }
    encoded = json.dumps(canonical_dict, sort_keys=True).encode("utf-8")
    return hashlib.md5(encoded).hexdigest()


class MemoCache:
    """Persistent cache storing simulation evaluations."""

    def __init__(self, cache_file: str = DEFAULT_CACHE_CSV):
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.total_queries = 0
        self.cache_hits = 0
        self.new_sims = 0
        self._last_file_size = 0
        self._load_cache()

    def _load_cache(self):
        """Loads existing entries from the persistent CSV file."""
        if not os.path.exists(self.cache_file):
            return

        try:
            curr_size = os.path.getsize(self.cache_file)
            if curr_size == self._last_file_size:
                return

            with open(self.cache_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    md5 = row.get("md5_hash")
                    if not md5 or md5 in self.cache:
                        continue
                    entry = {
                        "Wdiff": float(row["Wdiff"]),
                        "Wnb": float(row["Wnb"]),
                        "Wpo": float(row["Wpo"]),
                        "Wbp": float(row["Wbp"]),
                        "Cc": float(row["Cc"]),
                        "gain_db": float(row["gain_db"]),
                        "ugbw": float(row["ugbw"]),
                        "pm": float(row["pm"]),
                        "f3db": float(row.get("f3db", 0.0)),
                        "pdc": float(row["pdc"]),
                        "n_sat": int(row["n_sat"]),
                        "all_sat": (row["all_sat"].lower() == "true"),
                        "is_valid": (row["is_valid"].lower() == "true"),
                        "timestamp": row.get("timestamp", ""),
                    }
                    self.cache[md5] = entry
                    count += 1
            self._last_file_size = os.path.getsize(self.cache_file)
            if count > 0:
                logger.info(f"Loaded {count} new cached points from {self.cache_file} (total {len(self.cache)})")
        except Exception as e:
            logger.warning(f"Error loading cache from {self.cache_file}: {e}")

    def query(self, raw_params: Dict[str, float]) -> Tuple[bool, Dict[str, float], Optional[Dict[str, Any]]]:
        """
        Snaps parameters and checks if the point is in cache.
        Returns: (is_hit: bool, snapped_params: dict, cached_metrics: dict or None)
        """
        snapped = snap_parameters(raw_params)
        key = compute_param_md5(snapped)
        self.total_queries += 1

        if key in self.cache:
            self.cache_hits += 1
            logger.info(f"[cache-hit] MD5={key[:8]} params={snapped}")
            return True, snapped, self.cache[key]

        # Check if another process updated the cache CSV
        self._load_cache()
        if key in self.cache:
            self.cache_hits += 1
            logger.info(f"[cache-hit] MD5={key[:8]} params={snapped}")
            return True, snapped, self.cache[key]

        self.new_sims += 1
        logger.info(f"[new-sim]   MD5={key[:8]} params={snapped}")
        return False, snapped, None

    def store(self, snapped_params: Dict[str, float], metrics: Dict[str, Any]):
        """Stores a new simulation evaluation in the cache and appends to CSV with process file lock."""
        import fcntl
        key = compute_param_md5(snapped_params)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        entry = {
            "Wdiff": snapped_params["Wdiff"],
            "Wnb": snapped_params["Wnb"],
            "Wpo": snapped_params["Wpo"],
            "Wbp": snapped_params["Wbp"],
            "Cc": snapped_params["Cc"],
            "gain_db": metrics.get("gain_db", 0.0),
            "ugbw": metrics.get("ugbw", 0.0),
            "pm": metrics.get("pm", 0.0),
            "f3db": metrics.get("f3db", 0.0),
            "pdc": metrics.get("pdc", 0.0),
            "n_sat": metrics.get("n_sat", 0),
            "all_sat": metrics.get("all_sat", False),
            "is_valid": metrics.get("is_valid", False),
            "timestamp": ts,
        }
        self.cache[key] = entry

        # Append to CSV with flock for multiprocess safety
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        try:
            with open(self.cache_file, "a+", newline="", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.seek(0, os.SEEK_END)
                    file_size = f.tell()
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    if file_size == 0:
                        writer.writeheader()
                    row = dict(entry)
                    row["md5_hash"] = key
                    writer.writerow(row)
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            self._last_file_size = os.path.getsize(self.cache_file)
        except Exception as e:
            logger.error(f"Error persisting point to cache CSV: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Returns cache utilization statistics."""
        hit_fraction = (self.cache_hits / self.total_queries) if self.total_queries > 0 else 0.0
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "new_sims": self.new_sims,
            "hit_fraction": hit_fraction,
            "total_cached_entries": len(self.cache),
        }
