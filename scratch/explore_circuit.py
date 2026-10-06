import subprocess
import os
import re
import numpy as np

os.environ["PATH"] = "/home/install/SPECTRE211/bin:" + os.environ.get("PATH", "")
os.environ["CDS_LIC_FILE"] = "5280@cadence"
os.environ["LM_LICENSE_FILE"] = "5280@cadence"

def run_spectre(netlist_text, raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    scs_path = os.path.join(raw_dir, "input.scs")
    log_path = os.path.join(raw_dir, "spectre.log")
    with open(scs_path, "w") as f:
        f.write(netlist_text)
    
    cmd = [
        "spectre", scs_path,
        "-format", "psfascii",
        "-raw", raw_dir,
        "+log", log_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    return res.returncode

print("Spectre test script ready")
