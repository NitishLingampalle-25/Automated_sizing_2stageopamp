import subprocess, os, re, numpy as np

# Test A: PMOS bias branch
# Ibias = 30u from pbias to vss.
# M8 (pbias pbias vdd vdd) pmos1 w=Wbp l=0.18u
# M6 (vout pbias vdd vdd) pmos1 w=Wpo l=0.18u
# M7 (vout net_m4 vss vss) nmos1 w=Wnb l=0.18u
# But what is M5? If M5 is NMOS, where does M5 gate connect?
# Could M5 be self-biased or connected to something?

# Test B: What if M8 is NMOS bias branch, but the prompt called it "PMOS bias mirror branch" because:
# Could M3, M4 be the PMOS bias mirror?
# Let's check how M1..M8 are connected.
