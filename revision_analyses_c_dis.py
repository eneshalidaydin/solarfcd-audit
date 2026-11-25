"""R2.5, R2.13: external-validation checks from saved predictions (deposit files + internal OOF). Read-only inputs."""
import sys
import numpy as np
import pandas as pd
from scipy.stats import beta, norm

sys.stdout.reconfigure(encoding="utf-8")
K = "C:/solarfcd/pilot/"
DEP = "<DEPOSIT_DIR>/"
SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
PC = [f"p_{c}" for c in SIN]


def wilson(k, n, z=norm.ppf(0.975)):
    p = k / n
    m = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return m - h, m + h


for nm, f in [("RF-Fault", "external_overlap_rf_fault.csv"), ("RF-Soiling", "external_overlap_rf_soiling.csv")]:
    O = pd.read_csv(DEP + f)
    md5, ph, cs = O.md5_match, O.phash_min_distance <= 12, O.dinov2_max_cosine >= 0.85
    print(f"{nm}: n {len(O)}, removed {O.removed_as_overlapping.sum()} (MD5 {md5.sum()}, pHash<=12 {ph.sum()}, cos>=0.85 {cs.sum()}, "
          f"cosine-only {(cs & ~ph & ~md5).sum()}, cos>=0.90 {(O.dinov2_max_cosine >= 0.90).sum()}, cos>=0.95 {(O.dinov2_max_cosine >= 0.95).sum()})")
    for c, g in O.groupby("mapped_label"):
        print(f"   {c:22s} n {len(g):5d} removed {g.removed_as_overlapping.mean():.3f}; cosine-only removed "
              f"{((g.dinov2_max_cosine >= 0.85) & (g.phash_min_distance > 12) & ~g.md5_match).mean():.3f}")
    if nm == "RF-Fault":
        four = O[O.mapped_label.isin(SIN)]
        print("   four-class before screening:", four.mapped_label.value_counts().to_dict(),
              "| after:", four[~four.removed_as_overlapping].mapped_label.value_counts().to_dict())

X = pd.read_csv(DEP + "external_predictions_rf_fault.csv")
O = pd.read_csv(DEP + "external_overlap_rf_fault.csv").set_index("rel_path")
X["cos"] = X.rel_path.map(O.dinov2_max_cosine)
E3cols = ["ConvNeXt", "Swin", "DINOv2-392"]
PE = np.mean([X[[f"{m}__{p}" for p in PC]].to_numpy() for m in E3cols], 0)
X["pred"] = np.array(SIN)[PE.argmax(1)]
X["conf"] = PE.max(1)
F = X[X.mapped_label.isin(SIN)].copy()
F["ok"] = F.pred == F.mapped_label
print(f"\nRF-Fault four-class n {len(F)}: E3 acc {F.ok.mean():.4f}; majority baseline {F.mapped_label.value_counts().max() / len(F):.4f} "
      f"({F.mapped_label.value_counts().idxmax()})")
bands = [0, 0.5, 0.6, 0.7, 0.8, 0.85]
F["band"] = pd.cut(F.cos, bands, right=False)
print(F.groupby("band", observed=True).ok.agg(["mean", "size"]).round(3).to_string())
rec = {c: (F[F.mapped_label == c].pred == c).mean() for c in ["Healthy", "Structural_Fault", "Surface_Obstruction"]}
print("E3 external recalls:", {k: round(v, 3) for k, v in rec.items()}, "balanced acc", round(np.mean(list(rec.values())), 3))
for m in ["ConvNeXt", "Swin", "DINOv2-392"]:
    pm = np.array(SIN)[X.loc[F.index, [f"{m}__{p}" for p in PC]].to_numpy().argmax(1)]
    r = [(pm[F.mapped_label.to_numpy() == c] == c).mean() for c in ["Healthy", "Structural_Fault", "Surface_Obstruction"]]
    k = (pm == F.mapped_label.to_numpy()).sum()
    print(f"  {m:11s} acc {k / len(F):.3f} [Wilson {wilson(k, len(F))[0]:.3f}, {wilson(k, len(F))[1]:.3f}] balanced {np.mean(r):.3f}")
k = F.ok.sum()
print(f"  E3          acc {k / len(F):.3f} [Wilson {wilson(k, len(F))[0]:.3f}, {wilson(k, len(F))[1]:.3f}]")
# in-domain E3 (9-run mean), same three classes
E3 = ["convnext_tiny_224", "swin_tiny_patch4_window7_224_224", "vit_base_patch14_dinov2_392"]
D = [pd.concat([pd.read_csv(f"{K}runs_asama2/grouped{k}_{u}_s{s}_kat_v1_asama2_none/test_predictions.csv")
                for k in range(5)]).set_index("path").sort_index() for u in E3 for s in (42, 43, 44)]
PI = np.mean([d[PC].to_numpy() for d in D], 0)
yi = D[0].label.to_numpy()
pi = np.array(SIN)[PI.argmax(1)]
m3 = yi != "Electrical_Fault"
reci = {c: (pi[yi == c] == c).mean() for c in ["Healthy", "Structural_Fault", "Surface_Obstruction"]}
prior = {c: (yi == c).sum() / m3.sum() for c in reci}
print(f"In-domain E3 on H/SF/SO images (n={m3.sum()}): acc {(pi[m3] == yi[m3]).mean():.4f}; recalls "
      f"{ {k: round(v, 3) for k, v in reci.items()} }; balanced {np.mean(list(reci.values())):.3f}; priors { {k: round(v, 4) for k, v in prior.items()} }")
print(f"RF-Fault E3 recalls re-weighted by SolarFCD H/SF/SO priors: {sum(prior[c] * rec[c] for c in rec):.4f}")
# transferred LTT threshold (calibrated on all 2,095 OOF, dis_secici.txt): 0.9072
thr = 0.9072
acc_ = F.conf >= thr
n, e = acc_.sum(), (~F.ok[acc_]).sum()
print(f"\nTransferred threshold {thr}: accepted {n}/{len(F)} = {n / len(F):.3f}, errors {e}, accuracy {1 - e / n:.4f}; "
      f"one-sided CP upper bound on error 95%: {beta.ppf(0.95, e + 1, n - e):.4f}, 90%: {beta.ppf(0.90, e + 1, n - e):.4f}")
# ECE with bootstrap CIs
man = pd.read_csv(K + "asama2/manifest_asama2.csv").set_index("path")
g = D[0].index.map(man["sahne_grup"]).to_numpy()


def ece(conf, ok, w=None):
    w = np.ones(len(conf)) if w is None else w
    b = np.minimum((conf * 10).astype(int), 9)
    tot = 0
    for i in range(10):
        m = b == i
        if w[m].sum() > 0:
            tot += w[m].sum() / w.sum() * abs(np.average(ok[m], weights=w[m]) - np.average(conf[m], weights=w[m]))
    return tot


rng = np.random.default_rng(0)
ci, oki = PI.max(1), pi == yi
gu, gi = np.unique(g, return_inverse=True)
bi = [ece(ci, oki, np.bincount(rng.choice(len(gu), len(gu)), minlength=len(gu))[gi]) for _ in range(2000)]
print(f"\nECE in-domain {ece(ci, oki):.3f} [cluster bootstrap {np.percentile(bi, 2.5):.3f}, {np.percentile(bi, 97.5):.3f}]; "
      f"max confidence {ci.max():.3f}, 99th percentile {np.percentile(ci, 99):.3f}; bin counts {np.bincount(np.minimum((ci * 10).astype(int), 9), minlength=10).tolist()}")
cf, okf = F.conf.to_numpy(), F.ok.to_numpy()
bf = [ece(cf[i], okf[i]) for i in [rng.integers(0, len(cf), len(cf)) for _ in range(2000)]]
print(f"ECE RF-Fault {ece(cf, okf):.3f} [i.i.d. bootstrap {np.percentile(bf, 2.5):.3f}, {np.percentile(bf, 97.5):.3f}]; bin counts "
      f"{np.bincount(np.minimum((cf * 10).astype(int), 9), minlength=10).tolist()}")
S = pd.read_csv(DEP + "external_predictions_rf_soiling.csv")
PS = np.mean([S[[f"{m}__{p}" for p in PC]].to_numpy() for m in E3cols], 0)
cs_, oks = PS.max(1), np.array(SIN)[PS.argmax(1)] == S.mapped_label.to_numpy()
bs = [ece(cs_[i], oks[i]) for i in [rng.integers(0, len(cs_), len(cs_)) for _ in range(2000)]]
print(f"ECE RF-Soiling {ece(cs_, oks):.3f} [i.i.d. bootstrap {np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]; bin counts "
      f"{np.bincount(np.minimum((cs_ * 10).astype(int), 9), minlength=10).tolist()}")
# fault / no-fault on 1,478
fy = X.mapped_label.isin(["Structural_Fault", "Electrical_Fault", "Fault_unknown_type"])
fp = X.pred.isin(["Structural_Fault", "Electrical_Fault"])
k = (fy == fp).sum()
print(f"\nE3 fault/no-fault acc {k / len(X):.4f} [Wilson {wilson(k, len(X))[0]:.3f}, {wilson(k, len(X))[1]:.3f}] n={len(X)}; "
      f"all-faulty baseline {fy.mean():.4f}")
