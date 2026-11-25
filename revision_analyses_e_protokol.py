"""R2.7, R2.8, R2.9, R2.10, R2.13: protocol, grouping, interval and TTA checks from saved outputs. Read-only inputs."""
import glob
import json
import sys
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.stdout.reconfigure(encoding="utf-8")
K = "C:/solarfcd/pilot/"
DEP = "<DEPOSIT_DIR>/"
SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
PC = [f"p_{c}" for c in SIN]
man = pd.read_csv(K + "asama2/manifest_asama2.csv").set_index("path")


def rd(kok, name):
    return pd.concat([pd.read_csv(f"{K}{kok}/{name.replace('grouped0_', f'grouped{k}_')}/test_predictions.csv")
                      for k in range(5)]).set_index("path").sort_index()


def f1(y, p, w=None):
    w = np.ones(len(y)) if w is None else w
    c = np.zeros((4, 4)); np.add.at(c, (y, p), w)
    den = c.sum(0) + c.sum(1)
    return np.mean(np.where(den > 0, 2 * np.diag(c) / np.maximum(den, 1e-12), 0))


# ---- R2.8: P2 vs P3 by scene-group size (seed 42) ----
ref = rd("runs_asama2", "grouped0_convnext_tiny_224_s42_kat_v1_asama2_none")
paths = ref.index
sg = paths.map(man["sahne_grup"]).to_numpy()
size = pd.Series(sg).map(pd.Series(sg).value_counts()).to_numpy()
y = ref.label.map(SIN.index).to_numpy()
fold_rand = paths.map(pd.read_csv(K + "asama2_rastgele/manifest_asama2.csv").set_index("path")["kat_rastgele"]).to_numpy()
print("manifest fold columns:", [c for c in man.columns if c.startswith("kat")])
fr = fold_rand
partner_other = np.array([(fr[sg == s] != f).any() for s, f in zip(sg, fr)])
print(f"Evaluation images with a scene partner in another fold under P2: {partner_other.mean():.3f}; multi-member images "
      f"{(size > 1).sum()}, singletons {(size == 1).sum()}")
for mdl in ["convnext_tiny_224", "swin_tiny_patch4_window7_224_224", "vit_base_patch14_dinov2_224", "tf_efficientnetv2_s_224",
            "resnetv2_101_224"]:
    p3 = rd("runs_asama2", f"grouped0_{mdl}_s42_kat_v1_asama2_none").loc[paths]
    p2 = rd("runs_rastgele", f"grouped0_{mdl}_s42_kat_rastgele_asama2_rastgele_none").loc[paths]
    a3, a2 = (p3.tahmin == p3.label).to_numpy(), (p2.tahmin == p2.label).to_numpy()
    mm = size > 1
    print(f"  {mdl:34s} P2 vs P3 acc: all {a2.mean():.3f}/{a3.mean():.3f}; multi-member {a2[mm].mean():.3f}/{a3[mm].mean():.3f}; "
          f"singletons {a2[~mm].mean():.3f}/{a3[~mm].mean():.3f}; partner in other P2 fold {a2[partner_other].mean():.3f}/"
          f"{a3[partner_other].mean():.3f}")

# ---- E3 and DINOv2-392 intervals, fold-0 sensitivity, 299-cluster stability ----
E3 = ["convnext_tiny_224", "swin_tiny_patch4_window7_224_224", "vit_base_patch14_dinov2_392"]
runs = {(u, s): rd("runs_asama2", f"grouped0_{u}_s{s}_kat_v1_asama2_none").loc[paths] for u in E3 for s in (42, 43, 44)}
for (u, s) in [("vit_base_patch14_dinov2_224", s) for s in (42, 43, 44)] + [("convnext_tiny_384", s) for s in (42, 43, 44)]:
    runs[(u, s)] = rd("runs_asama2", f"grouped0_{u}_s{s}_kat_v1_asama2_none").loc[paths]
PE = np.mean([runs[(u, s)][PC].to_numpy() for u in E3 for s in (42, 43, 44)], 0)
pe = PE.argmax(1)
fold = paths.map(man["kat_v1"]).to_numpy()
gu = np.unique(sg); gi = np.searchsorted(gu, sg)
rng = np.random.default_rng(0)
W = np.stack([np.bincount(rng.choice(len(gu), len(gu)), minlength=len(gu))[gi] for _ in range(2000)])
bf = np.array([f1(y, pe, w) for w in W]); ba = np.array([np.average(pe == y, weights=w) for w in W])
print(f"\nE3: acc {(pe == y).mean():.4f} [{np.percentile(ba, 2.5):.3f}, {np.percentile(ba, 97.5):.3f}], macro-F1 {f1(y, pe):.4f} "
      f"[{np.percentile(bf, 2.5):.3f}, {np.percentile(bf, 97.5):.3f}]")
rec = [(pe[y == i] == i).mean() for i in range(4)]
print("E3 recalls EF,H,SF,SO", np.round(rec, 3))
for u in ["vit_base_patch14_dinov2_392", "vit_base_patch14_dinov2_224", "convnext_tiny_384"]:
    pr = {s: runs[(u, s)][PC].to_numpy().argmax(1) for s in (42, 43, 44)}
    b = np.mean([[f1(y, p, w) for w in W] for p in pr.values()], 0)
    print(f"{u}: acc {np.mean([(p == y).mean() for p in pr.values()]):.4f} ± {np.std([(p == y).mean() for p in pr.values()], ddof=1):.4f}, "
          f"F1 {np.mean([f1(y, p) for p in pr.values()]):.4f} ± {np.std([f1(y, p) for p in pr.values()], ddof=1):.4f} "
          f"[{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]")
    if u == "vit_base_patch14_dinov2_392":
        b392 = b
    if u == "vit_base_patch14_dinov2_224":
        b224 = b
d = b392 - b224
p_two = 2 * min((d <= 0).mean(), (d >= 0).mean())
print(f"DINOv2-392 minus 224 macro-F1: [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}], bootstrap two-sided p = {p_two:.4f}; "
      f"Holm threshold for the smallest of six p-values = {0.05 / 6:.4f}")
m0 = fold == 0
print(f"E3 fold 0: acc {(pe[m0] == y[m0]).mean():.4f} (n={m0.sum()}); folds 1-4: acc {(pe[~m0] == y[~m0]).mean():.4f}; "
      f"macro-F1 without fold 0 {f1(y[~m0], pe[~m0]):.4f}")
big = pd.Series(sg).value_counts().idxmax()
mb = sg == big
print(f"Largest scene group {big}: {mb.sum()} images, classes {pd.Series(y[mb]).map(dict(enumerate(SIN))).value_counts().to_dict()}, "
      f"folds {np.unique(fold[mb]).tolist()}, E3 acc on it {(pe[mb] == y[mb]).mean():.4f}")
y2, p2_ = y[~mb], pe[~mb]
g2 = sg[~mb]; gu2 = np.unique(g2); gi2 = np.searchsorted(gu2, g2)
W2 = np.stack([np.bincount(rng.choice(len(gu2), len(gu2)), minlength=len(gu2))[gi2] for _ in range(2000)])
bf2 = np.array([f1(y2, p2_, w) for w in W2]); ba2 = np.array([np.average(p2_ == y2, weights=w) for w in W2])
print(f"E3 without the largest group (n={len(y2)}): acc {(p2_ == y2).mean():.4f} [{np.percentile(ba2, 2.5):.3f}, {np.percentile(ba2, 97.5):.3f}], "
      f"F1 {f1(y2, p2_):.4f} [{np.percentile(bf2, 2.5):.3f}, {np.percentile(bf2, 97.5):.3f}]")

# ---- R2.7: scene groups with and without visual-review edges ----
M = pd.read_csv(DEP + "solarfcd_manifest.csv")
idx = {p: i for i, p in enumerate(M.rel_path)}
L = pd.read_csv(DEP + "link_edges.csv")
L = L[L.rel_path_a.isin(idx) & L.rel_path_b.isin(idx)]


def comps(edges):
    a = edges.rel_path_a.map(idx).to_numpy(); b = edges.rel_path_b.map(idx).to_numpy()
    G = coo_matrix((np.ones(len(a)), (a, b)), shape=(len(M), len(M)))
    return connected_components(G, directed=False)[1]


ev = M.in_evaluation_set.to_numpy()
for nm, E in [("all edges", L), ("without scene_visual", L[L.edge_type != "scene_visual"]),
              ("without scene_visual and photo_visual", L[~L.edge_type.isin(["scene_visual", "photo_visual"])])]:
    c = comps(E)[ev]
    vc = pd.Series(c).value_counts()
    print(f"{nm:40s}: groups among the 2,095 = {len(vc)}, singletons {(vc == 1).sum()}, largest {vc.max()}, next {vc.iloc[1:4].tolist()}")
same = pd.crosstab(comps(L)[ev], M.scene_group[ev]).astype(bool).sum(1).max()
print("check: each rebuilt component maps to", same, "manifest scene group(s)")
V = pd.read_csv(DEP + "visual_pair_reviews.csv")
print(f"\nvisual_pair_reviews: {len(V)} decisions; {V.decision.value_counts().to_dict()}; single decision {(V.n_decisions == 1).sum()}; "
      f"certainty {V.certainty.value_counts().to_dict()}")
print("edges from visual review: photo_visual", (L.edge_type == "photo_visual").sum(), "scene_visual", (L.edge_type == "scene_visual").sum())

# ---- R2.10: TTA vs single view, pooled over folds, P3 runs ----
rows = []
for f in glob.glob(K + "runs_asama2/grouped*_kat_v1_asama2_none/results.json"):
    r = json.load(open(f, encoding="utf-8"))
    if "test_tta" not in r:
        continue
    nm = f.split("\\")[-2] if "\\" in f else f.split("/")[-2]
    n = sum(r["test"]["confusion_matrix"][i][j] for i in range(4) for j in range(4))
    rows.append(dict(run=nm.split("_", 1)[1], n=n, acc=r["test"]["accuracy"], acc_tta=r["test_tta"]["accuracy"],
                     f1=r["test"]["macro_f1"], f1_tta=r["test_tta"]["macro_f1"]))
R = pd.DataFrame(rows)
R["model"] = R.run.str.replace(r"_s4[234]_kat.*", "", regex=True)
G = R.groupby("model").apply(lambda d: pd.Series(dict(acc=np.average(d.acc, weights=d.n), acc_tta=np.average(d.acc_tta, weights=d.n),
                                                       runs=len(d))), include_groups=False)
G["diff_pp"] = 100 * (G.acc_tta - G.acc)
print("\nTTA effect (P3, weighted over folds and seeds):"); print(G.round(4).to_string())
