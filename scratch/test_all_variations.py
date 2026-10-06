import subprocess, os, re, numpy as np

os.environ["PATH"] = "/home/install/SPECTRE211/bin:" + os.environ.get("PATH", "")
os.environ["CDS_LIC_FILE"] = "5280@cadence"
os.environ["LM_LICENSE_FILE"] = "5280@cadence"

def run_scs(netlist, name):
    raw_dir = f"scratch/raw_{name}"
    os.makedirs(raw_dir, exist_ok=True)
    scs_file = f"scratch/{name}.scs"
    with open(scs_file, "w") as f:
        f.write(netlist)
    cmd = ["spectre", "-64", scs_file, "-format", "psfascii", "-raw", raw_dir, "+log", f"scratch/{name}.log"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0: return None, "Spectre error"
    
    # Read opInfo.info
    info_file = os.path.join(raw_dir, "opInfo.info")
    sat_dict = {}
    if os.path.exists(info_file):
        with open(info_file) as f:
            content = f.read()
        for m in [f"M{i}" for i in range(1, 9)]:
            idx = content.find(f'"{m}" "bsim3v3"')
            if idx != -1:
                chunk = content[idx:idx+2500]
                lines = chunk.split("\n")
                vals_l = [l.strip() for l in lines[1:] if not l.strip().startswith(")")]
                vds = float(vals_l[4])
                vdsat = float(vals_l[10])
                if vds < 0:
                    is_sat = abs(vds) >= abs(vdsat)
                else:
                    is_sat = vds >= vdsat
                sat_dict[m] = (vds, vdsat, is_sat)
            
    # Read acSwp.ac
    ac_file = os.path.join(raw_dir, "acSwp.ac")
    ac_data = []
    if os.path.exists(ac_file):
        with open(ac_file) as f:
            cur_freq = None
            in_val = False
            for l in f:
                if "VALUE" in l: in_val = True; continue
                if "END" in l: in_val = False
                if in_val:
                    l = l.strip()
                    if l.startswith('"freq"'):
                        cur_freq = float(l.split()[1])
                    elif l.startswith('"vout"'):
                        m = re.search(r'\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\)', l)
                        if m and cur_freq is not None:
                            ac_data.append((cur_freq, complex(float(m.group(1)), float(m.group(2)))))
    
    gain_db, ugbw, pm = 0, 0, 0
    if ac_data:
        freqs = np.array([x[0] for x in ac_data])
        vouts = np.array([x[1] for x in ac_data])
        mags_db = 20 * np.log10(np.maximum(np.abs(vouts), 1e-12))
        gain_db = np.max(mags_db)
        phases_deg = np.unwrap(np.angle(vouts)) * 180 / np.pi
        cross_idx = np.where(np.diff(np.sign(mags_db)))[0]
        if len(cross_idx) > 0:
            i = cross_idx[0]
            f1, f2 = freqs[i], freqs[i+1]
            g1, g2 = mags_db[i], mags_db[i+1]
            ugbw = f1 + (0 - g1) * (f2 - f1) / (g2 - g1)
            p1, p2 = phases_deg[i], phases_deg[i+1]
            phase_ugbw = p1 + (0 - g1) * (p2 - p1) / (g2 - g1)
            pm = (180 + phase_ugbw + 180) % 360 - 180
            if pm < 0: pm += 360

    n_sat = sum(1 for v in sat_dict.values() if v[2])
    return (n_sat, sat_dict, gain_db, ugbw, pm), None

print("Testing PMOS diff pair vs NMOS diff pair...")

# Var A: PMOS input diff pair
# If diff pair is PMOS (M1, M2):
# M1, M2: PMOS (sources to tail, gates to inm, inp)
# M3, M4: NMOS mirror load (sources to vss)
# M5: PMOS tail (source to vdd, gate to pbias)
# M6: NMOS common source (source to vss, gate to net_m4, drain to vout)
# M7: PMOS load (source to vdd, gate to pbias, drain to vout)
# M8: PMOS diode (source to vdd, gate/drain to pbias)
# Ibias = 30u from pbias to vss
# Cc from net_m4 to vout
nl_pmos_diff = """
simulator lang=spectre
include "/home/install/FOUNDRY/analog/180nm/models/spectre/gpdk.scs" section=NN
opt options soft_bin=allmodels save=allpub subcktprobelvl=2
Vdd (vdd 0) vsource dc=0.9
Vss (vss 0) vsource dc=-0.9
Vin_p (inp 0) vsource dc=0 mag=1
Vin_m (inm 0) vsource dc=0 mag=0

// PMOS bias mirror branch: Io = 30u from pbias to vss
Ibias (pbias vss) isource dc=30u
M8 (pbias pbias vdd vdd) pmos1 w=30u l=0.18u

// PMOS tail transistor M5
M5 (tail pbias vdd vdd) pmos1 w=30u l=0.18u

// PMOS diff pair M1, M2
M1 (net_m3 inm tail vdd) pmos1 w=3.4u l=0.18u
M2 (net_m4 inp tail vdd) pmos1 w=3.4u l=0.18u

// NMOS mirror load M3, M4
M3 (net_m3 net_m3 vss vss) nmos1 w=160u l=0.18u
M4 (net_m4 net_m3 vss vss) nmos1 w=160u l=0.18u

// Second stage: NMOS common-source M6, PMOS load M7
M6 (vout net_m4 vss vss) nmos1 w=160u l=0.18u
M7 (vout pbias vdd vdd) pmos1 w=5.5u l=0.18u
Cc (net_m4 vout) capacitor c=1.5p
CL (vout 0) capacitor c=10p

dcOp dc
opInfo info what=oppoint where=rawfile
acSwp ac start=1 stop=100M dec=20
"""

res, _ = run_scs(nl_pmos_diff, "var_pmos_diff")
if res:
    print(f"PMOS diff: Sat={res[0]}/8, Gain={res[2]:.1f} dB, UGBW={res[3]/1e6:.3f} MHz, PM={res[4]:.1f} deg")
    for m, s in res[1].items():
        print(f"  {m}: Vds={s[0]:.4f}, Vdsat={s[1]:.4f}, Sat={s[2]}")
