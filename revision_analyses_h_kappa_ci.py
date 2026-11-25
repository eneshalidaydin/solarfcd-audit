"""R2.2: bootstrap CIs for the reported Fleiss' kappa (variable number of ratings per item, all 150 images). Read-only."""
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
A = pd.read_csv("C:/solarfcd/pilot/insan_kontrol/etiket/sonuc.csv", encoding="utf-8-sig")
U = A[["uzman1", "uzman2", "uzman3"]].fillna("bos")
SEC = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction", "karisik", "emin_degil"]
M = np.stack([(U == c).sum(axis=1).to_numpy() for c in SEC], 1)


def fleiss_var(M):
    n = M.sum(1)
    M, n = M[n >= 2], n[n >= 2]
    Pi = (M * (M - 1)).sum(1) / (n * (n - 1))
    p = M.sum(0) / M.sum()
    return (Pi.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum())


rng = np.random.default_rng(0)
for nm, m in [("all", np.ones(len(A), bool)), ("candidates", A.aday.to_numpy()), ("controls", ~A.aday.to_numpy())]:
    X = M[m]
    b = [fleiss_var(X[rng.integers(0, len(X), len(X))]) for _ in range(2000)]
    print(f"{nm:10s} n={len(X)} Fleiss (variable n) {fleiss_var(X):.3f} [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]")
