"""R2.1, R2.6, R2.13: re-run of the 22_secici_konformal.py procedure (E3 = ConvNeXt-224 + Swin-224 + DINOv2-392,
3 seeds, original labels, 1,000 scene-level 50/50 splits, rng seed 0) with extra diagnostics. Read-only inputs."""
import sys
import numpy as np
import pandas as pd
from scipy.stats import beta

sys.stdout.reconfigure(encoding="utf-8")
K = "C:/solarfcd/pilot/"
DEP = "<DEPOSIT_DIR>/"
SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
PC = [f"p_{c}" for c in SIN]
E3 = ["convnext_tiny_224", "swin_tiny_patch4_window7_224_224", "vit_base_patch14_dinov2_392"]
D = [pd.concat([pd.read_csv(f"{K}runs_asama2/grouped{k}_{u}_s{s}_kat_v1_asama2_none/test_predictions.csv")
                for k in range(5)]).set_index("path").sort_index() for u in E3 for s in (42, 43, 44)]
P0 = np.mean([d[PC].to_numpy() for d in D], 0)
lab0 = D[0]["label"]
man = pd.read_csv(K + "asama2/manifest_asama2.csv").set_index("path")
g0 = lab0.index.map(man["sahne_grup"]).to_numpy()
A = pd.read_csv(K + "insan_kontrol/etiket/sonuc.csv", encoding="utf-8-sig")
amb = set(A.path[A.uzlasi == "belirsiz"])
cand = set(A.path[A.aday])


def ust(k, n, d):
    return 1.0 if n == 0 else (1.0 if k == n else beta.ppf(1 - d, k + 1, n - k))


def run(keep, nsplit=1000, seed=0, extra=False, delta=0.1, risk=0.05):
    P, y, g = P0[keep], lab0.to_numpy()[keep], g0[keep]
    y = pd.Series(y).map(SIN.index).to_numpy()
    gu = np.unique(g)
    conf, pred = P.max(1), P.argmax(1)
    ok = pred == y
    rng = np.random.default_rng(seed)
    R = []
    for _ in range(nsplit):
        cal = np.isin(g, rng.choice(gu, len(gu) // 2, replace=False))
        te = ~cal
        o = np.argsort(-conf[cal])
        err = np.cumsum(~ok[cal][o])
        n = np.arange(1, cal.sum() + 1)
        son = None
        for c in np.arange(0.50, 1.0001, 0.05):
            i = int(c * len(n)) - 1
            if ust(err[i], n[i], delta / 11) <= risk:
                son = i
        thr = conf[cal][o][son] if son is not None else 1.01
        acc_ = te & (conf >= thr)
        r = dict(cov=acc_.sum() / te.sum(), acc=ok[acc_].mean() if acc_.any() else np.nan,
                 viol=(1 - ok[acc_].mean()) > risk if acc_.any() else False, any=acc_.any())
        if extra:
            sk = 1 - P[np.arange(len(y)), y]
            q = np.quantile(sk[cal], min(1, np.ceil((cal.sum() + 1) * 0.95) / cal.sum()), method="higher")
            S = (1 - P) <= q
            # Mondrian (class-conditional) LAC
            Sm = np.zeros_like(S)
            for c in range(4):
                mc = cal & (y == c)
                nc = mc.sum()
                qc = np.quantile(sk[mc], min(1, np.ceil((nc + 1) * 0.95) / nc), method="higher")
                Sm[:, c] = (1 - P[:, c]) <= qc
            for c in range(4):
                m = te & (y == c)
                r[f"lac_cov_{c}"] = S[m, c].mean()
                r[f"lac_size_{c}"] = S[m].sum(1).mean()
                r[f"mon_cov_{c}"] = Sm[m, c].mean()
                r[f"acc_rate_{c}"] = acc_[m].mean()
            r["lac_size"] = S[te].sum(1).mean()
            r["mon_size"] = Sm[te].sum(1).mean()
            r["mon_cov"] = Sm[te, y[te]].mean()
            fault = np.isin(y, [0, 2])
            nonf_pred = np.isin(pred, [1, 3])
            r["acc_fault"] = (acc_ & fault).sum()
            r["acc_fault_missed"] = (acc_ & fault & nonf_pred).sum()
            r["acc_total"] = acc_.sum()
            r["acc_pred_nonfault"] = (acc_ & nonf_pred).sum()
            r["acc_pred_nonfault_truefault"] = (acc_ & nonf_pred & fault).sum()
        R.append(r)
    return pd.DataFrame(R)


allk = np.ones(len(lab0), bool)
R = run(allk, extra=True)
print(f"Original labels: coverage {R['cov'].mean():.3f} ± {R['cov'].std():.3f}; zero-coverage splits {(R['cov'] == 0).sum()}; "
      f"median {R['cov'].median():.3f}, IQR {R['cov'].quantile(.25):.3f}-{R['cov'].quantile(.75):.3f}; "
      f"violation {R.viol.mean():.3f}; violation | any accepted {R.viol[R['any']].mean():.3f} (n={R['any'].sum()})")
print("LAC marginal set size", R.lac_size.mean().round(3))
for c in range(4):
    print(f"  {SIN[c]:20s} LAC cov {R[f'lac_cov_{c}'].mean():.3f} size {R[f'lac_size_{c}'].mean():.2f} | "
          f"Mondrian cov {R[f'mon_cov_{c}'].mean():.3f} | LTT accepted share {R[f'acc_rate_{c}'].mean():.3f}")
print(f"Mondrian LAC: marginal cov {R.mon_cov.mean():.3f}, mean set size {R.mon_size.mean():.3f}")
print(f"Accepted fault (EF+SF) test images per split {R.acc_fault.mean():.2f}; of which predicted H/SO {R.acc_fault_missed.mean():.2f} "
      f"(missed-fault rate among accepted fault images {R.acc_fault_missed.sum() / max(R.acc_fault.sum(), 1):.3f}); "
      f"among images auto-labelled non-fault, share truly faulty {R.acc_pred_nonfault_truefault.sum() / R.acc_pred_nonfault.sum():.4f}")
# deposit per-split files: conditional violation, alpha = 2% ambiguous removed
for f in ["original_alpha0.05", "original_ambiguous_removed_alpha0.05", "audited_alpha0.05",
          "original_alpha0.02", "original_ambiguous_removed_alpha0.02", "audited_alpha0.02",
          "original_alpha0.10", "original_ambiguous_removed_alpha0.10", "audited_alpha0.10"]:
    d = pd.read_csv(DEP + f + ".csv")
    a = d.coverage > 0
    print(f"{f:40s} cov {d.coverage.mean():.3f} median {d.coverage.median():.3f} IQR {d.coverage.quantile(.25):.3f}-"
          f"{d.coverage.quantile(.75):.3f}; zero {(~a).sum()}; viol {d.violation.mean():.3f}; viol|accepted "
          f"{d.violation[a].mean() if a.any() else float('nan'):.3f} ({d.violation[a].sum()}/{a.sum()})")
# mechanical effect of removing 46 images (R2.1)
paths = lab0.index
iamb = paths.isin(amb)
icand = paths.isin(cand)
rng = np.random.default_rng(1)
Ra = run(~iamb, nsplit=300)
print(f"\nRemove the 46 expert-ambiguous images: LTT coverage {Ra['cov'].mean():.3f} (300 splits)")
rb, rc = [], []
cidx, nidx = np.flatnonzero(icand), np.flatnonzero(~icand)
for t in range(20):
    k = np.ones(len(paths), bool)
    k[rng.choice(cidx, 35, replace=False)] = False
    k[rng.choice(nidx, 11, replace=False)] = False
    rb.append(run(k, nsplit=300, seed=t)["cov"].mean())
    k2 = np.ones(len(paths), bool)
    k2[rng.choice(len(paths), 46, replace=False)] = False
    rc.append(run(k2, nsplit=300, seed=t)["cov"].mean())
print(f"Remove 35 random candidates + 11 random non-candidates (no expert input), 20 draws: coverage {np.mean(rb):.3f} "
      f"(range {np.min(rb):.3f}-{np.max(rb):.3f})")
print(f"Remove 46 random images, 20 draws: coverage {np.mean(rc):.3f} (range {np.min(rc):.3f}-{np.max(rc):.3f})")
