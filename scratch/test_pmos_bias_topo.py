import subprocess
import os
import re
import numpy as np

os.environ["PATH"] = "/home/install/SPECTRE211/bin:" + os.environ.get("PATH", "")
os.environ["CDS_LIC_FILE"] = "5280@cadence"
os.environ["LM_LICENSE_FILE"] = "5280@cadence"

# What if M8 is PMOS (Wbp) and M5 is NMOS? But where does M5 gate connect?
# Or what if M8 is NMOS and M_bias_p is PMOS?
# What if the 8 transistors are:
# M1, M2: NMOS diff pair
# M3, M4: PMOS mirror load
# M5: NMOS tail
# M6: PMOS common source
# M7: NMOS load
# M8: bias diode

# Wait, why was M7 in triode earlier?
# Because in:
# M6 (vout net_m4 vdd vdd) pmos1 w=Wpo
# M7 (vout nbias vss vss) nmos1 w=Wnb
# M7 width is Wnb (160u) and M8 width is Wnb (160u).
# So M7 pulls 30uA.
# But M6 width is Wpo (5.5u), while M34 is Wbp (30u).
# So M6 only carries 2.75uA!
# If M6 carries 2.75uA and M7 pulls 30uA, M7 MUST be in triode!

# BUT WHAT IF:
# What if M34 is NOT Wbp?
# What if M34 is Wpo (5.5u), and M6 is Wbp (30u)?
# Then M6 carries 15 * (30/5.5) = 81 uA!
# Still doesn't match 30uA.

# WHAT IF M7 IS NOT CONNECTED TO NBIAS?!
# What if M7 is diode-connected?! No.
# What if M7 is NOT width Wnb?
# What if the width of M7 is Wpo?
# Or what if M5 is Wpo and M7 is Wnb?

# Let's test what happens if M5 is NOT 160u!
