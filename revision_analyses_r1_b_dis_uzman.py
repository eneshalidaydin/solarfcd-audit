"""İç değerlendirme R1 kontrolü - B: uzman denetimi, dis veri (RF-Fault/RF-Soiling), az ornek, cift kararlari.
Yalniz okur: depo CSV'leri (Desktop/.../solarfcd-audit) ve pilot ciktilari. Cikti: bu klasor."""
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
DEP = Path(r"<DEPOSIT_DIR>")
SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    L.append(s)


def wilson(k, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ---------------------------------------------------------------- 1. Uzman denetimi
P("=" * 20, "1. EXPERT AUDIT")
E = pd.read_csv(DEP / "expert_audit.csv", dtype=str)
for c in ("expert1", "expert2", "expert3"):
    E[c] = E[c].fillna("blank")
E["candidate"] = E["candidate"] == "True"
X = E[["expert1", "expert2", "expert3"]]
P("values:", sorted(set(X.to_numpy().ravel())))
for c in X.columns:
    P(c, X[c].value_counts().to_dict())
ctrl = E[~E.candidate]
P("controls by original label:", ctrl.label_original.value_counts().to_dict())
hc = ctrl[ctrl.label_original == "Healthy"]
for c in X.columns:
    P("H controls", c, hc[c].value_counts().to_dict())
soc = ctrl[ctrl.label_original == "Surface_Obstruction"]
for c in X.columns:
    P("SO controls", c, soc[c].value_counts().to_dict())
fs = E[E.label_original.isin(["Electrical_Fault", "Structural_Fault"])]
P("EF/SF originals:", fs.label_original.value_counts().to_dict())
for lab in ["Electrical_Fault", "Structural_Fault"]:
    g = fs[fs.label_original == lab]
    for c in X.columns:
        P(" ", lab, c, g[c].value_counts().to_dict())
    P(" ", lab, "consensus", g.consensus.value_counts().to_dict())
nmix = (X == "mixed").sum().sum()
nmix_fs = (fs[["expert1", "expert2", "expert3"]] == "mixed").sum().sum()
P("mixed answers total", nmix, "| on EF/SF originals", nmix_fs, "| EF/SF images", len(fs))
nns = (X == "not_sure").sum().sum()
P("not_sure total", nns, "by expert", (X == "not_sure").sum().to_dict())

CATS6 = SIN + ["mixed", "not_sure"]


def fleiss_complete(Xs, cats):
    M = np.stack([(Xs == c).sum(1).to_numpy() for c in cats], 1)
    n = M.sum(1)
    assert (n == n[0]).all()
    n = n[0]
    p = M.sum(0) / M.sum()
    Pi = (M * (M - 1)).sum(1) / (n * (n - 1))
    pe = (p ** 2).sum()
    k = (Pi.mean() - pe) / (1 - pe)
    # kategori bazinda kappa (Fleiss 1971)
    N = len(M)
    kj = {c: 1 - (M[:, j] * (n - M[:, j])).sum() / (N * n * (n - 1) * p[j] * (1 - p[j])) if 0 < p[j] < 1 else np.nan
          for j, c in enumerate(cats)}
    return k, kj, N


def kripp_alpha(Xs, cats):
    """Nominal Krippendorff alpha, eksik veri destekli (degerler cats disinda = eksik)."""
    idx = {c: i for i, c in enumerate(cats)}
    K = len(cats)
    o = np.zeros((K, K))
    for row in Xs.itertuples(index=False):
        v = [idx[x] for x in row if x in idx]
        m = len(v)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[v[a], v[b]] += 1 / (m - 1)
    nc = o.sum(1)
    n = nc.sum()
    Do = (o.sum() - np.trace(o))
    De = (n * n - (nc ** 2).sum()) / (n - 1)
    return 1 - Do / De, int(n)


def gwet_ac1(Xs, cats):
    idx = {c: i for i, c in enumerate(cats)}
    K = len(cats)
    R = []
    for row in Xs.itertuples(index=False):
        r = np.zeros(K)
        for x in row:
            if x in idx:
                r[idx[x]] += 1
        R.append(r)
    R = np.array(R)
    ri = R.sum(1)
    use = ri >= 2
    pa = np.mean([(R[i] * (R[i] - 1)).sum() / (ri[i] * (ri[i] - 1)) for i in np.where(use)[0]])
    pik = np.mean(R[ri > 0] / ri[ri > 0, None], 0)
    pe = (pik * (1 - pik)).sum() / (K - 1)
    return (pa - pe) / (1 - pe), int(use.sum())


def cohen(a, b, cats):
    m = a.isin(cats) & b.isin(cats)
    a, b = a[m], b[m]
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe), po, int(m.sum())


nb = (X != "blank").all(1)
k6, kj6, N6 = fleiss_complete(X[nb], CATS6)
P(f"Fleiss kappa, 6 categories, images without blank answers (n={N6}): {k6:.3f}")
P("  category-specific kappa (6 cat):", {c: round(v, 3) for c, v in kj6.items()})
comp = X.isin(SIN).all(1)
k4, kj4, N4 = fleiss_complete(X[comp], SIN)
P(f"Fleiss kappa, 4 classes, images where all three gave a class (n={N4}): {k4:.3f}")
P("  category-specific kappa (4 cls):", {c: round(v, 3) for c, v in kj4.items()})
a6, n6 = kripp_alpha(X, CATS6)
a4, n4 = kripp_alpha(X, SIN)
P(f"Krippendorff alpha nominal: 6 categories {a6:.3f} (pairable values {n6}); mixed/not_sure/blank as missing {a4:.3f} (pairable values {n4})")
g6, u6 = gwet_ac1(X, CATS6)
g4, u4 = gwet_ac1(X, SIN)
P(f"Gwet AC1: 6 categories {g6:.3f} (items {u6}); mixed/not_sure as missing {g4:.3f} (items {u4})")
for i, j in [("expert1", "expert2"), ("expert1", "expert3"), ("expert2", "expert3")]:
    k, po, n = cohen(X[i], X[j], CATS6)
    k_, po_, n_ = cohen(X[i], X[j], SIN)
    P(f"  {i}-{j}: 6 cat kappa {k:.3f}, raw agreement {po:.3f} (n={n}) | 4 cls kappa {k_:.3f}, raw {po_:.3f} (n={n_})")
hs = X.isin(["Healthy", "Surface_Obstruction"]).all(1)
kh, _, Nh = fleiss_complete(X[hs], ["Healthy", "Surface_Obstruction"])
P(f"Fleiss kappa H vs SO, images where all three answered H or SO (n={Nh}): {kh:.3f}")
for nm, sub in [("candidates", E[E.candidate]), ("controls", ctrl)]:
    Xs = sub[["expert1", "expert2", "expert3"]]
    a_, _ = kripp_alpha(Xs, SIN)
    g_, _ = gwet_ac1(Xs, SIN)
    P(f"  {nm}: Krippendorff alpha (4 cls, others missing) {a_:.3f}, Gwet AC1 {g_:.3f}")
P("unanimous share:", round((X.nunique(axis=1) == 1).mean(), 3))
P("status by candidate:", E.groupby("candidate").status.value_counts().to_dict())

# ---------------------------------------------------------------- 2. RF-Fault etiket esleme
P("=" * 20, "2. RF-FAULT LABEL MAPPING")
O = pd.read_csv(DEP / "external_overlap_rf_fault.csv", dtype={"source_label_or_box_classes": str})
O["removed"] = O.removed_as_overlapping.astype(str) == "True"
P("RF-Fault rows", len(O), "removed", int(O.removed.sum()), f"({O.removed.mean():.3%})", "retained", int((~O.removed).sum()))
R = O[~O.removed].copy()
R["cls"] = R.source_label_or_box_classes.fillna("").map(lambda s: {int(x) for x in s.split(",") if x != ""})
P("retained mapped labels:", R.mapped_label.value_counts().to_dict())
U = R[R.mapped_label == "Fault_unknown_type"]
soil = {0, 2, 3}
cat = U.cls.map(lambda c: "Defective+snow" if 6 in c and not (c & soil) else (
    "Defective+soiling+snow" if 6 in c and (c & soil) else ("Defective+soiling" if c & soil else (
        "Defective+NonDefective" if 4 in c else ("Defective only" if c == {1} else "other")))))
P("unknown-type (n=%d) composition:" % len(U), cat.value_counts().to_dict())
U_soil = U[U.cls.map(lambda c: bool(c & (soil | {6})))]
P("unknown with soiling or snow box:", len(U_soil), "| of which with snow:", int(U_soil.cls.map(lambda c: 6 in c).sum()))
P("box-class combinations among all retained (top 20):")
P(R.source_label_or_box_classes.fillna("").value_counts().head(20).to_dict())
P("retained images with >1 box class:", int((R.cls.map(len) > 1).sum()), "of", len(R))

R["name"] = R.rel_path.map(lambda p: Path(p).name)
R["stem"] = R.name.map(lambda n: re.split(r"[._]rf[._]|_jpg\.rf", n)[0])
imd = R[R.name.str.startswith("Imgdirty_")]
P("retained RF-Fault with Imgdirty_* names:", len(imd), "| carrying Defective (class 1):", int(imd.cls.map(lambda c: 1 in c).sum()),
  "| mapped labels:", imd.mapped_label.value_counts().to_dict())
P("  Imgdirty box combos:", imd.source_label_or_box_classes.value_counts().head(8).to_dict())
imc = R[R.name.str.startswith("Imgclean_")]
P("retained RF-Fault with Imgclean_* names:", len(imc))
M = pd.read_csv(DEP / "solarfcd_manifest.csv")
M["stem"] = M.rel_path.map(lambda p: Path(p).stem)
sstems = set(M.stem)
same = R[R.stem.isin(sstems)]
P("retained RF-Fault whose stem equals a SolarFCD file stem:", len(same), "| unique stems:", same.stem.nunique())
R["nn_stem"] = R.nearest_solarfcd.map(lambda p: Path(str(p)).stem)
nn_same = R[R.stem == R.nn_stem]
P("retained RF-Fault whose nearest SolarFCD neighbour has the same stem:", len(nn_same))
P(nn_same[["stem", "phash_min_distance", "dinov2_max_cosine", "nearest_solarfcd", "mapped_label"]].to_string())
P("same-stem retained: pHash range", same.phash_min_distance.min(), same.phash_min_distance.max(),
  "cos range", same.dinov2_max_cosine.min(), same.dinov2_max_cosine.max())
P("retained RF-Fault unique stems:", R.stem.nunique(), "for", len(R), "images")

# Tahminler ve ariza/ariza-yok yeniden hesap
Q = pd.read_csv(DEP / "external_predictions_rf_fault.csv")
Q = Q.merge(R[["rel_path", "cls", "stem"]], on="rel_path", how="left")
assert Q.cls.notna().all(), "eslesmeyen satir"
P("prediction rows", len(Q))


def probs(df, m):
    return df[[f"{m}__p_{c}" for c in SIN]].to_numpy()


MODS = {"ConvNeXt": probs(Q, "ConvNeXt"), "Swin": probs(Q, "Swin"), "DINOv2-392": probs(Q, "DINOv2-392")}
MODS["E3"] = np.mean([MODS["ConvNeXt"], MODS["Swin"], MODS["DINOv2-392"]], 0)
lab = Q.mapped_label.to_numpy()
unk = lab == "Fault_unknown_type"
unk_soil = unk & Q.cls.map(lambda c: bool(c & (soil | {6}))).to_numpy()
y_fault_ms = np.isin(lab, ["Electrical_Fault", "Structural_Fault"]) | unk
y_fault_alt = np.isin(lab, ["Electrical_Fault", "Structural_Fault"]) | (unk & ~unk_soil)
known = np.isin(lab, SIN)
for nm, Pm in MODS.items():
    pr = Pm.argmax(1)
    pf = np.isin(pr, [0, 2])
    a1 = (pf == y_fault_ms).mean()
    b1 = max(y_fault_ms.mean(), 1 - y_fault_ms.mean())
    a2 = (pf == y_fault_alt).mean()
    b2 = max(y_fault_alt.mean(), 1 - y_fault_alt.mean())
    ex = ~unk
    a3 = (pf[ex] == y_fault_ms[ex]).mean()
    b3 = max(y_fault_ms[ex].mean(), 1 - y_fault_ms[ex].mean())
    yk = np.array([SIN.index(v) for v in lab[known]])
    acc4 = (pr[known] == yk).mean()
    # 4 sinif, Defective+kir/kar -> SO
    lab2 = np.where(unk_soil, "Surface_Obstruction", lab)
    k2 = np.isin(lab2, SIN)
    yk2 = np.array([SIN.index(v) for v in lab2[k2]])
    acc4b = (pr[k2] == yk2).mean()
    sorec = (pr[k2][yk2 == 3] == 3).mean()
    P(f"{nm:10s} fault/no-fault: manuscript mapping {a1:.3%} (baseline {b1:.3%}, n={len(lab)}) | Defective+soil/snow as no-fault "
      f"{a2:.3%} (baseline {b2:.3%}) | unknown excluded {a3:.3%} (baseline {b3:.3%}, n={int(ex.sum())}) | 4-class {acc4:.3%} "
      f"(n={int(known.sum())}) | 4-class with Defective+soil/snow as SO {acc4b:.3%} (n={int(k2.sum())}, SO recall {sorec:.3f})")
pE3 = MODS["E3"].argmax(1)
P("E3 predictions on Defective+soiling/snow images (n=%d):" % unk_soil.sum(),
  pd.Series(np.array(SIN)[pE3[unk_soil]]).value_counts().to_dict())
P("E3 predictions on Defective-only/other unknown (n=%d):" % (unk & ~unk_soil).sum(),
  pd.Series(np.array(SIN)[pE3[unk & ~unk_soil]]).value_counts().to_dict())

# Kume bootstrap (kok ad) vs Wilson
rng = np.random.default_rng(20261124)


def stem_boot(correct, stems, B=2000):
    df = pd.DataFrame(dict(c=correct.astype(float), s=stems))
    gs = df.groupby("s").c.agg(["sum", "count"])
    sm, ct = gs["sum"].to_numpy(), gs["count"].to_numpy()
    idx = rng.integers(0, len(gs), size=(B, len(gs)))
    v = sm[idx].sum(1) / ct[idx].sum(1)
    return np.percentile(v, [2.5, 97.5]), len(gs)


st = Q.stem.to_numpy()
for nm in ["ConvNeXt", "Swin", "DINOv2-392", "E3"]:
    pr = MODS[nm].argmax(1)
    yk = np.array([SIN.index(v) if v in SIN else -1 for v in lab])
    c = (pr == yk)[known]
    ci, ng = stem_boot(c, st[known])
    w = wilson(int(c.sum()), len(c))
    P(f"{nm:10s} 4-class acc {c.mean():.3%}: Wilson {w[0]:.3%}-{w[1]:.3%} | stem-cluster bootstrap {ci[0]:.3%}-{ci[1]:.3%} ({ng} stems)")
    pf = np.isin(pr, [0, 2])
    c2 = pf == y_fault_ms
    ci2, ng2 = stem_boot(c2, st)
    w2 = wilson(int(c2.sum()), len(c2))
    P(f"{'':10s} fault/no-fault {c2.mean():.3%}: Wilson {w2[0]:.3%}-{w2[1]:.3%} | stem bootstrap {ci2[0]:.3%}-{ci2[1]:.3%} ({ng2} stems)")
P("four-class known images:", int(known.sum()), "| unique stems among them:", Q.stem[known].nunique())

# ---------------------------------------------------------------- 3. RF-Soiling
P("=" * 20, "3. RF-SOILING")
S = pd.read_csv(DEP / "external_overlap_rf_soiling.csv", dtype={"source_label_or_box_classes": str})
S["removed"] = S.removed_as_overlapping.astype(str) == "True"
P("RF-Soiling rows", len(S), "removed", int(S.removed.sum()), f"({S.removed.mean():.3%})", "retained", int((~S.removed).sum()))
Sr = S[~S.removed].copy()
Sr["stem"] = Sr.rel_path.map(lambda p: re.split(r"[._]rf[._]|_jpg\.rf", Path(p).name)[0])
P("retained RF-Soiling unique stems:", Sr.stem.nunique(), "| mapped:", Sr.mapped_label.value_counts().to_dict())
fst = set(R.stem)
shared = sorted(set(Sr.stem) & fst)
P("RF-Soiling retained stems also among retained RF-Fault stems:", len(shared), shared[:10])
allf = set(O.rel_path.map(lambda p: re.split(r"[._]rf[._]|_jpg\.rf", Path(p).name)[0]))
P("RF-Soiling retained stems among ALL RF-Fault stems (incl. removed):", len(set(Sr.stem) & allf))
P("RF-Fault retained images with ROI_* stems:", int(R.stem.str.startswith("ROI_").sum()), "| their mapped labels:",
  R[R.stem.str.startswith("ROI_")].mapped_label.value_counts().to_dict())
QS = pd.read_csv(DEP / "external_predictions_rf_soiling.csv")
QS["stem"] = QS.rel_path.map(lambda p: re.split(r"[._]rf[._]|_jpg\.rf", Path(p).name)[0])
for nm in ["ConvNeXt", "Swin", "DINOv2-392"]:
    pr = probs(QS, nm).argmax(1)
    c = pr == 3
    ci, ng = stem_boot(c, QS.stem.to_numpy())
    w = wilson(int(c.sum()), len(c))
    P(f"{nm:10s} RF-Soiling SO recall {c.mean():.3f}: Wilson {w[0]:.3f}-{w[1]:.3f} | stem bootstrap {ci[0]:.3f}-{ci[1]:.3f} ({ng} stems)")
pE = np.mean([probs(QS, m) for m in ["ConvNeXt", "Swin", "DINOv2-392"]], 0).argmax(1)
c = pE == 3
ci, ng = stem_boot(c, QS.stem.to_numpy())
P(f"E3         RF-Soiling SO recall {c.mean():.3f} | stem bootstrap {ci[0]:.3f}-{ci[1]:.3f}; E3 predictions:",
  pd.Series(np.array(SIN)[pE]).value_counts().to_dict())

# ---------------------------------------------------------------- 4. Az ornek: oturum yapisi
P("=" * 20, "4. FEW-SHOT SESSION STRUCTURE (1,108 four-class RF-Fault images)")
K4 = R[R.mapped_label.isin(SIN)].copy()
P("four-class images", len(K4), K4.mapped_label.value_counts().to_dict())
d03 = K4[K4.stem.str.startswith("IMG_20241003_")]
tm = d03.stem.str.extract(r"IMG_20241003_(\d{6})")[0].dropna()
P("IMG_20241003_* images:", len(d03), "| unique stems:", d03.stem.nunique(), "| time range:", tm.min(), "-", tm.max(),
  "| labels:", d03.mapped_label.value_counts().to_dict())
dts = K4.stem.str.extract(r"IMG_(\d{8})_")[0].dropna()
P("IMG_YYYYMMDD_* by date:", dts.value_counts().to_dict())
cam = K4[K4.stem.str.match(r"^IMG_\d{4}")]
camn = cam.stem.str.extract(r"^IMG_(\d{4})")[0]
P("IMG_#### images:", len(cam), "| distinct camera numbers:", camn.nunique(), "| labels:", cam.mapped_label.value_counts().to_dict())
fam = K4.stem.map(lambda s: re.sub(r"[\d_\-\s\(\)]+$", "", re.split(r"\d", s)[0]) or s[:6])
P("top name families (four-class):", fam.value_counts().head(12).to_dict())
pdm = K4[K4.stem.str.contains("Physical_Dam", case=False)]
P("'Physical_Dam*' family:", len(pdm), pdm.mapped_label.value_counts().to_dict())
P("SF images total:", int((K4.mapped_label == "Structural_Fault").sum()))
F = pd.read_csv(DEP / "few_shot_adaptation.csv")
P(F.to_string())
P(F.groupby(["target_fraction", "initialisation"]).acc.agg(["mean", "std", "min", "max", "count"]).round(4).to_string())
# eslestirilmis fark (ayni tekrar, iki baslangic)
pv = F.pivot_table(index=["target_fraction", "repeat"], columns="initialisation", values="acc").dropna()
pv["diff"] = pv["solarfcd"] - pv["imagenet"]
P(pv.round(4).to_string())

# ---------------------------------------------------------------- 5. Cift kararlari
P("=" * 20, "5. VISUAL PAIR REVIEWS")
V = pd.read_csv(DEP / "visual_pair_reviews.csv")
P("rows", len(V), "| decisions:", V.decision.value_counts().to_dict())
P("review_round prefixes:", V.review_round.astype(str).map(lambda s: s.split("_")[0]).value_counts().to_dict())
P("n_decisions sum:", int(V.n_decisions.sum()))

# ---------------------------------------------------------------- 6. LTT asgari n
P("=" * 20, "6. LTT MINIMUM ACCEPTED CALIBRATION IMAGES WITH ZERO ERRORS")
from scipy.stats import beta
for d in (0.1, 0.05, 0.01):
    n = 1
    while beta.ppf(1 - d / 11, 1, n) > 0.05:
        n += 1
    P(f"delta={d}: zero errors needs n_k >= {n}; with the 50% minimum coverage level, calibration set >= {2 * n}; "
      f"one error needs n_k >= " + str(next(m for m in range(2, 2000) if beta.ppf(1 - d / 11, 2, m - 1) <= 0.05)))

(OUT / "r1_b_sonuc.txt").write_text("\n".join(L), encoding="utf-8")
