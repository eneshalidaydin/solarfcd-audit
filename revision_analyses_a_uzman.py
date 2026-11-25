"""R2.1, R2.2, R2.3: expert-audit checks from saved outputs only (read-only inputs under C:/solarfcd/pilot)."""
import sys
from itertools import combinations
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
K = "C:/solarfcd/pilot/"
SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
PC = [f"p_{c}" for c in SIN]
E3 = ["convnext_tiny_224", "swin_tiny_patch4_window7_224_224", "vit_base_patch14_dinov2_392"]


def oof(kok, son, uyeler=E3):
    D = [pd.concat([pd.read_csv(f"{K}{kok}/grouped{k}_{u}_s{s}_{son}/test_predictions.csv") for k in range(5)])
         .set_index("path").sort_index() for u in uyeler for s in (42, 43, 44)]
    return pd.DataFrame(np.mean([d[PC].to_numpy() for d in D], 0), index=D[0].index, columns=PC), D[0]["label"]


P, lab = oof("runs_asama2", "kat_v1_asama2_none")
PT, _ = oof("runs_temiz", "kat_v1_asama2_temiz_none")
pred = pd.Series(np.array(SIN)[P.to_numpy().argmax(1)], index=P.index)
predT = pd.Series(np.array(SIN)[PT.to_numpy().argmax(1)], index=PT.index)
A = pd.read_csv(K + "insan_kontrol/etiket/sonuc.csv", encoding="utf-8-sig")
A["e3"] = A.path.map(pred)
A["e3T"] = A.path.map(predT)
cand, ctrl = A[A.aday], A[~A.aday]
amb = A.uzlasi == "belirsiz"
print("E3 overall acc (orig labels):", (pred == lab).mean().round(4), "n", len(pred))
print("Candidates: n", len(cand), "| E3 == original label:", (cand.e3 == cand.label).sum())
c77 = cand[cand.uzlasi != "belirsiz"]
print("Non-ambiguous candidates", len(c77), "| consensus == E3:", (c77.uzlasi == c77.e3).sum(),
      "| consensus == original:", (c77.uzlasi == c77.label).sum())
print("Ambiguous (all 150):", amb.sum(), "| of which candidates", (amb & A.aday).sum(),
      "| E3 errors among ambiguous:", (A[amb].e3 != A[amb].label).sum())
ch = ctrl[ctrl.durum == "degisti"]
print("Controls changed:", len(ch), ch.label.value_counts().to_dict(), "->", ch.uzlasi.value_counts().to_dict(),
      "| E3 prediction on them:", ch.e3.value_counts().to_dict())
print("Controls ambiguous:", (ctrl.durum == "belirsiz").sum(), "| kept", (ctrl.durum == "ayni").sum())
c27 = ctrl[ctrl.uzlasi != "belirsiz"]
print("Controls: E3 == original", (ctrl.e3 == ctrl.label).sum(), "/", len(ctrl),
      "| E3 == consensus", (c27.e3 == c27.uzlasi).sum(), "/", len(c27),
      "| retrained E3 == consensus", (c27.e3T == c27.uzlasi).sum(), "/", len(c27))
for u in ("uzman1", "uzman2", "uzman3"):
    v = ctrl[ctrl[u].isin(SIN)]
    print(f"  controls, E3 vs {u}: {(v.e3 == v[u]).sum()}/{len(v)} = {(v.e3 == v[u]).mean():.3f} (4-class answers only); "
          f"over all 38: {(ctrl.e3 == ctrl[u]).mean():.3f}; {u} vs original {(ctrl[u] == ctrl.label).mean():.3f}")

# ---------- stratified (inverse sampling probability) estimator, R2.1 ----------
ev = pd.DataFrame({"label": lab, "e3": pred})
candset = set(cand.path)
ev["cand"] = ev.index.isin(candset)
N_nc = ev[~ev.cand].label.value_counts()          # non-candidate population by class
n_ct = ctrl.label.value_counts()                    # controls by class (proportional allocation)
print("Non-candidate population by class:", N_nc.to_dict(), "| controls by class:", n_ct.to_dict())


def strat(ctrl_df, ref="uzlasi", pred_col="e3"):
    # candidates fully enumerated; controls weighted by N_class / n_class (post-stratified)
    w = ctrl_df.label.map(N_nc / n_ct)
    na_c = cand[cand[ref] != "belirsiz"]
    ok_c = (na_c[pred_col] == na_c[ref]).sum()
    m = ctrl_df[ref] != "belirsiz"
    num = ok_c + (w[m] * (ctrl_df[m][pred_col] == ctrl_df[m][ref])).sum()
    den = len(na_c) + w[m].sum()
    return num / den, den


est, den = strat(ctrl)
rng = np.random.default_rng(0)
bs = []
for _ in range(2000):
    s = pd.concat([g.sample(len(g), replace=True, random_state=int(rng.integers(1e9))) for _, g in ctrl.groupby("label")])
    bs.append(strat(s)[0])
print(f"Stratified E3 accuracy vs expert consensus (non-ambiguous): {est:.3f} [{np.percentile(bs, 2.5):.3f}, "
      f"{np.percentile(bs, 97.5):.3f}] (estimated non-ambiguous population {den:.0f})")
estT, _ = strat(ctrl, pred_col="e3T")
print(f"Same for retrained E3 (audited labels): {estT:.3f}")
# same estimator with ORIGINAL label as reference: noise check against the known full-data value
ctrl2 = ctrl.assign(ref_o=ctrl.label)
cand_o = (cand.e3 == cand.label).sum()
w = ctrl.label.map(N_nc / n_ct)
est_o = (cand_o + (w * (ctrl.e3 == ctrl.label)).sum()) / (len(cand) + w.sum())
print(f"Check: same estimator with original labels = {est_o:.3f} vs true full-data E3 accuracy {(pred == lab).mean():.3f}")

# ---------- agreement statistics, R2.2 ----------
U = A[["uzman1", "uzman2", "uzman3"]].fillna("bos").replace({"bos": np.nan})
SEC = SIN + ["karisik", "emin_degil"]


def counts(Ux, cats):
    return np.stack([(Ux == c).sum(1).to_numpy() for c in cats], 1)


def fleiss_script(M):  # as in 23_etiket_birlestir.py (n from first row)
    n = M.sum(1)[0]
    p = M.sum(0) / M.sum()
    Pi = (M * (M - 1)).sum(1) / (n * (n - 1))
    return (Pi.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum())


def fleiss_var(M):  # variable number of ratings per item (items with >= 2 ratings)
    n = M.sum(1)
    M = M[n >= 2]; n = n[n >= 2]
    Pi = (M * (M - 1)).sum(1) / (n * (n - 1))
    p = M.sum(0) / M.sum()
    return (Pi.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum())


def kripp(M):  # nominal Krippendorff alpha from item x category counts (missing allowed)
    n = M.sum(1)
    M = M[n >= 2]; n = n[n >= 2]
    o = np.zeros((M.shape[1], M.shape[1]))
    for r, m in zip(M, n):
        o += (np.outer(r, r) - np.diag(r)) / (m - 1)
    nc = o.sum(1); N = nc.sum()
    Do = (o.sum() - np.trace(o)) / N
    De = (N * N - (nc ** 2).sum()) / (N * (N - 1))
    return 1 - Do / De


def ac1(M):
    n = M.sum(1)
    M = M[n >= 2]; n = n[n >= 2]
    pa = ((M * (M - 1)).sum(1) / (n * (n - 1))).mean()
    pi = (M / n[:, None]).mean(0)
    pe = (pi * (1 - pi)).sum() / (M.shape[1] - 1)
    return (pa - pe) / (1 - pe)


M6 = counts(U, SEC)
full = M6.sum(1) == 3
print(f"Fleiss (script, 6 options, all 150): {fleiss_script(M6):.3f}; complete items only ({full.sum()}): "
      f"{fleiss_script(M6[full]):.3f}; variable-n version: {fleiss_var(M6):.3f}")
for nm, msk in [("all", np.ones(len(A), bool)), ("candidates", A.aday.to_numpy()), ("controls", ~A.aday.to_numpy())]:
    Mx = M6[msk & full]
    bsk = [fleiss_script(Mx[rng.integers(0, len(Mx), len(Mx))]) for _ in range(2000)]
    print(f"  Fleiss {nm:10s} complete n={len(Mx)}: {fleiss_script(Mx):.3f} [{np.percentile(bsk, 2.5):.3f}, "
          f"{np.percentile(bsk, 97.5):.3f}]")
U4 = U.where(U.isin(SIN))           # mixed / not sure / blank -> missing
M4 = counts(U4, SIN)
all4 = M4.sum(1) == 3
print(f"Items where all three chose one of the four classes: {all4.sum()}; Fleiss on them {fleiss_script(M4[all4]):.3f}")
for nm, msk in [("all", np.ones(len(A), bool)), ("candidates", A.aday.to_numpy()), ("controls", ~A.aday.to_numpy())]:
    Mk6, Mk4 = M6[msk], M4[msk]
    bk = [kripp(Mk4[i]) for i in [rng.integers(0, len(Mk4), len(Mk4)) for _ in range(1000)]]
    bg = [ac1(Mk6[i]) for i in [rng.integers(0, len(Mk6), len(Mk6)) for _ in range(1000)]]
    print(f"  {nm:10s} Krippendorff alpha (6 options, missing allowed) {kripp(Mk6):.3f}; (4 classes, other answers missing) "
          f"{kripp(Mk4):.3f} [{np.percentile(bk, 2.5):.3f}, {np.percentile(bk, 97.5):.3f}]; Gwet AC1 (6 options) {ac1(Mk6):.3f} "
          f"[{np.percentile(bg, 2.5):.3f}, {np.percentile(bg, 97.5):.3f}]; unanimous {(Mk6.max(1) == 3).mean():.3f}")
for a, b in combinations(["uzman1", "uzman2", "uzman3"], 2):
    both = U[a].notna() & U[b].notna()
    print(f"  pairwise raw agreement {a}-{b}: {(U[a][both] == U[b][both]).mean():.3f} (n={both.sum()})")
# category-specific agreement (H and SO), 4-class answers
for c in ("Healthy", "Surface_Obstruction"):
    k = SIN.index(c)
    n = M4.sum(1)
    ps = (M4[:, k] * (M4[:, k] - 1)).sum() / (M4[:, k] * (n - 1)).clip(0).sum()
    print(f"  specific agreement for {c}: {ps:.3f}")
# votes behind SO consensus
so = A[A.uzlasi == "Surface_Obstruction"]
print("SO consensus labels:", len(so), {u: int((so[u] == "Surface_Obstruction").sum()) for u in ("uzman1", "uzman2", "uzman3")})
# leave-one-rater-out
for drop in ("uzman1", "uzman2", "uzman3"):
    a_, b_ = [u for u in ("uzman1", "uzman2", "uzman3") if u != drop]
    for nm, d in [("controls", ctrl), ("candidates", cand)]:
        agree = (d[a_] == d[b_]) & d[a_].isin(SIN)
        newl = d[a_].where(agree, "belirsiz")
        print(f"  drop {drop}, {nm}: changed {(agree & (newl != d.label)).sum()}, kept {(agree & (newl == d.label)).sum()}, "
              f"ambiguous {(~agree).sum()}")
# changed-or-ambiguous among controls
k = (ctrl.durum != "ayni").sum()
print(f"Controls changed or ambiguous: {k}/{len(ctrl)}")

# ---------- Dawid-Skene (4 classes; mixed/not sure/blank missing) ----------
R = np.stack([U4[u].map(lambda v: SIN.index(v) if v in SIN else -1).to_numpy() for u in ("uzman1", "uzman2", "uzman3")], 1)
T = np.zeros((len(R), 4))
for j in range(3):
    for i, r in enumerate(R[:, j]):
        if r >= 0:
            T[i, r] += 1
T = (T + 0.01) / (T + 0.01).sum(1, keepdims=True)
for _ in range(200):
    pri = T.mean(0)
    C = np.zeros((3, 4, 4))
    for j in range(3):
        for i in range(len(R)):
            if R[i, j] >= 0:
                C[j, :, R[i, j]] += T[i]
    C = (C + 0.1) / (C + 0.1).sum(2, keepdims=True)
    L = np.log(pri)[None].repeat(len(R), 0)
    for j in range(3):
        m = R[:, j] >= 0
        L[m] += np.log(C[j][:, R[m, j]].T)
    T = np.exp(L - L.max(1, keepdims=True)); T /= T.sum(1, keepdims=True)
H, S = SIN.index("Healthy"), SIN.index("Surface_Obstruction")
print("Dawid-Skene class priors (EF,H,SF,SO):", pri.round(3))
for j in range(3):
    print(f"  expert{j+1}: P(says SO | latent H) = {C[j, H, S]:.2f}, P(says H | latent H) = {C[j, H, H]:.2f}, "
          f"P(says H | latent SO) = {C[j, S, H]:.2f}, P(says SO | latent SO) = {C[j, S, S]:.2f}")
ds = np.array(SIN)[T.argmax(1)]
print("DS label vs original on controls:", (ds[~A.aday.to_numpy()] == ctrl.label.to_numpy()).sum(), "/ 38")
