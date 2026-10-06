"""
Headless Spectre execution runner.
Provides robust isolated simulation calls with timeout and failure protection.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger("SpectreRunner")

DEFAULT_TIMEOUT = 30  # seconds

# Ensure Cadence environment variables are configured
def ensure_cadence_env():
    spectre_bin_dir = "/home/install/SPECTRE211/bin"
    if spectre_bin_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{spectre_bin_dir}:{os.environ.get('PATH', '')}"
    os.environ.setdefault("CDS_LIC_FILE", "5280@cadence")
    os.environ.setdefault("LM_LICENSE_FILE", "5280@cadence")
    os.environ.setdefault("CDS_Netlisting_Mode", "Analog")
    os.environ.setdefault("CDS_AUTO_64BIT", "ALL")

ensure_cadence_env()


class SpectreRunner:
    """Manages headless Spectre execution in isolated directories."""

    def __init__(self, scratch_base: str = "/home/STUDENT/Nitish_Majorproject_Sem7/scratch/runs"):
        self.scratch_base = scratch_base
        os.makedirs(self.scratch_base, exist_ok=True)

    def run(
        self,
        netlist_content: str,
        fmt: str = "psfascii",
        timeout: int = DEFAULT_TIMEOUT,
    ) -> Tuple[bool, str, str, str]:
        """
        Runs Spectre on the given netlist content in an isolated directory.

        Parameters:
            netlist_content: Full Spectre netlist string.
            fmt: "psfascii" or "nutascii".
            timeout: Maximum execution time in seconds.

        Returns:
            (success: bool, raw_path: str, log_path: str, error_message: str)
            For psfascii, raw_path is a directory containing opInfo.info, acSwp.ac, etc.
            For nutascii, raw_path is a file path to the .raw file.
        """
        temp_dir = tempfile.mkdtemp(prefix="sim_", dir=self.scratch_base)
        netlist_file = os.path.join(temp_dir, "input.scs")
        log_file = os.path.join(temp_dir, "spectre.log")

        with open(netlist_file, "w") as f:
            f.write(netlist_content)

        if fmt == "psfascii":
            raw_path = os.path.join(temp_dir, "raw")
            os.makedirs(raw_path, exist_ok=True)
        elif fmt == "nutascii":
            raw_path = os.path.join(temp_dir, "output.raw")
        else:
            raise ValueError(f"Unsupported format '{fmt}'. Must be 'psfascii' or 'nutascii'.")

        cmd = [
            "spectre",
            "-64",
            netlist_file,
            "-format",
            fmt,
            "-raw",
            raw_path,
            "+log",
            log_file,
        ]

        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                cwd=temp_dir,
            )
            if res.returncode != 0:
                err_msg = f"Spectre exited with return code {res.returncode}."
                if os.path.exists(log_file):
                    with open(log_file, "r", errors="ignore") as lf:
                        err_tail = "\n".join(lf.readlines()[-20:])
                    err_msg += f"\nTail of log:\n{err_tail}"
                logger.warning(err_msg)
                return False, raw_path, log_file, err_msg

            return True, raw_path, log_file, ""

        except subprocess.TimeoutExpired:
            err_msg = f"Spectre simulation timed out after {timeout} seconds."
            logger.warning(err_msg)
            return False, raw_path, log_file, err_msg
        except Exception as e:
            err_msg = f"Spectre execution failed with unexpected exception: {e}"
            logger.error(err_msg)
            return False, raw_path, log_file, err_msg

    def cleanup(self, path: str):
        """Clean up temporary directory."""
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.isfile(path):
                parent = os.path.dirname(path)
                if os.path.exists(parent) and "sim_" in parent:
                    shutil.rmtree(parent, ignore_errors=True)
        except Exception as e:
            logger.debug(f"Failed to cleanup temp path {path}: {e}")
