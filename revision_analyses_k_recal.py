"""R2.5(vi): repeat the RF-Fault recalibration of 27_dis_secici_gurultu.py with GROUP-level calibration/test splits (the 613
near-duplicate groups of 28_az_ornek_uyarlama.py, reconstructed in d_azornek.py) instead of image-level splits. Read-only."""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.stats import beta

sys.stdout.reconfigure(encoding="utf-8")
K = "C:/solarfcd/pilot/"
OUT = Path(__file__).parent
SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
RISK, DEL = 0.05, 0.10
T = pd.read_csv(K + "dis_dogrulama/roboflow/ortusme.csv", encoding="utf-8-sig")
T = T[~T.ortusen].reset_index(drop=True)
y_all = T.label.map(lambda v: SIN.index(v) if v in SIN else -1).to_numpy()
P = np.mean([np.load(f"{K}dis_dogrulama/roboflow/tahminler_{a}.npy") for a in ("ConvNeXt", "Swin", "DINOv2-392")], 0)
keep = y_all >= 0
P, y = P[keep], y_all[keep]
T4 = T[keep].reset_index(drop=True)
H = np.load(OUT / "rf_phash.npy")                      # same order as the 1,108 four-class images in d_azornek.py
kok = T4.path.map(lambda p: re.split(r"[._]rf[._]|_jpg\.rf", Path(p).name)[0]).to_numpy()
A = ((H[:, None, :] != H[None, :, :]).sum(2) <= 6) | (kok[:, None] == kok[None, :])
ng, grp = connected_components(A, directed=False)
print("images", len(y), "groups", ng)


def ust(k, n, d):
    return 1.0 if n == 0 or k == n else beta.ppf(1 - d, k + 1, n - k)


def ltt(P, y, alt):
    conf = P.max(1); o = np.argsort(-conf); err = np.cumsum(P.argmax(1)[o] != y[o]); thr = 1.01
    G = np.arange(alt, 1.0001, 0.05)
    for c in G:
        i = max(int(c * len(y)) - 1, 0)
        if ust(err[i], i + 1, DEL / len(G)) <= RISK:
            thr = conf[o][i]
    sk = 1 - P[np.arange(len(y)), y]
    q = np.quantile(sk, min(1, np.ceil((len(y) + 1) * 0.95) / len(y)), method="higher")
    return thr, q


rng = np.random.default_rng(0)
for alt in (0.50, 0.05):
    for nc in (50, 100, 200, 400):
        R = []
        for _ in range(1000):
            cal = []
            n = 0
            for g_ in rng.permutation(ng):
                if n >= nc:
                    break
                cal.append(g_); n += (grp == g_).sum()
            c = np.isin(grp, cal); t = ~c
            thr, q = ltt(P[c], y[c], alt)
            sel = P[t].max(1) >= thr
            S = (1 - P[t]) <= q
            R.append(dict(cov=sel.mean(), lac=S[np.arange(t.sum()), y[t]].mean(), size=S.sum(1).mean()))
        R = pd.DataFrame(R)
        print(f"grid {alt:.2f}-1, n_cal~{nc}: LTT coverage {R['cov'].mean():.3f}; LAC coverage {R.lac.mean():.3f}, set size {R['size'].mean():.2f}")
