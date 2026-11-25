"""İç değerlendirme R1 kontrolü - A: alan ici (E3 OOF), secici tahmin sinif bazinda, delta duyarliligi, LAC -> eylem, katlar,
en buyuk sahne grubu, cozunurluk, P1 ayristirma. Yalniz okur (pilot ciktilari). Cikti: bu klasor."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

OUT = Path(__file__).parent
PIL = Path(r"C:\solarfcd\pilot")
SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
AB = ["EF", "H", "SF", "SO"]
PC = [f"p_{c}" for c in SIN]
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


def f1m(y, p, k=4):
    c = np.zeros((k, k))
    np.add.at(c, (y, p), 1)
    den = c.sum(0) + c.sum(1)
    return np.mean(np.where(den > 0, 2 * np.diag(c) / np.maximum(den, 1e-12), 0))


E3 = ("convnext_tiny_224", "swin_tiny_patch4_window7_224_224", "vit_base_patch14_dinov2_392")


def load(kok, son, uyeler=E3, seeds=(42, 43, 44)):
    D = [pd.concat([pd.read_csv(PIL / kok / f"grouped{k}_{u}_s{s}_{son}" / "test_predictions.csv") for k in range(5)])
         .set_index("path").sort_index() for u in uyeler for s in seeds]
    for d in D[1:]:
        assert (d.index == D[0].index).all()
    return D[0], np.mean([d[PC].to_numpy() for d in D], 0)


M = pd.read_csv(PIL / "asama2" / "manifest_asama2.csv").set_index("path")
FAM = {"Imgclean": "K-clean", "Imgdirty": "K-dirty", "Bird": "PD-bird", "Clean": "PD-clean", "Dust": "PD-dust",
       "Electrical": "PD-electrical", "Physical": "PD-physical", "Snow": "PD-snow"}

# ------------------------------------------------------------------ 1. E3 orijinal
P("=" * 20, "1. E3 OOF (original labels)")
d0, P0 = load("runs_asama2", "kat_v1_asama2_none")
y = d0.label.map(SIN.index).to_numpy()
pr = P0.argmax(1)
ok = pr == y
fam = d0.index.map(M["aile"]).map(lambda a: FAM.get(a, a)).to_numpy()
fold = d0.index.map(M["kat_v1"]).to_numpy()
sg = d0.index.map(M["sahne_grup"]).to_numpy()
P(f"n={len(y)} acc {ok.mean():.4f} macroF1 {f1m(y, pr):.4f}")
cm = np.zeros((4, 4), int)
np.add.at(cm, (y, pr), 1)
P("confusion (rows true EF,H,SF,SO):", cm.tolist())
hso = ((y == 1) & (pr == 3)) | ((y == 3) & (pr == 1))
P("H<->SO errors", int(hso.sum()), "of", int((~ok).sum()))
yf, pf = np.isin(y, [0, 2]), np.isin(pr, [0, 2])
tp, fp, fn = int((yf & pf).sum()), int((~yf & pf).sum()), int((yf & ~pf).sum())
rec, prec, fpr = tp / yf.sum(), tp / pf.sum(), fp / (~yf).sum()
P(f"fault (EF+SF) vs no-fault (H+SO): acc {(yf == pf).mean():.4f} baseline {max(yf.mean(), 1 - yf.mean()):.4f} "
  f"TP {tp} FP {fp} FN {fn} recall {rec:.3f} precision {prec:.3f} FPR {fp}/{int((~yf).sum())}={fpr:.4f}")
for pi in (0.01, 0.02, 0.05):
    P(f"  precision at fault prevalence {pi:.0%}: {rec * pi / (rec * pi + fpr * (1 - pi)):.3f}")
for a, b in [(2, 3), (2, 1), (0, 3), (0, 1), (2, 0), (0, 2)]:
    P(f"  true {AB[a]} -> pred {AB[b]}: {int(((y == a) & (pr == b)).sum())} of {int((y == a).sum())}")
# eylem odakli
ya, pa = y != 1, pr != 1
P(f"action needed (SO/EF/SF) vs no action (H): acc {(ya == pa).mean():.4f} baseline {max(ya.mean(), 1 - ya.mean()):.4f} "
  f"recall {(ya & pa).sum() / ya.sum():.3f} precision {(ya & pa).sum() / pa.sum():.3f} missed {int((ya & ~pa).sum())} "
  f"false alarms {int((~ya & pa).sum())}")
g3 = lambda v: np.where(np.isin(v, [0, 2]), 2, np.where(v == 3, 1, 0))  # 0 none, 1 cleaning, 2 repair
y3, p3 = g3(y), g3(pr)
P(f"3 action groups (H / SO / EF+SF): acc {(y3 == p3).mean():.4f} macroF1 {f1m(y3, p3, 3):.4f} recalls " +
  " ".join(f"{n} {(p3[y3 == i] == i).mean():.3f}" for i, n in enumerate(["none", "clean", "repair"])))
# kaynak ailesi
P("by source family: n, acc, predicted-class counts")
for f in sorted(set(fam)):
    m = fam == f
    P(f"  {f:14s} n={m.sum():4d} acc {ok[m].mean():.3f} pred", dict(zip(AB, np.bincount(pr[m], minlength=4).tolist())))
ns = fam != "PD-snow"
so, so2 = y == 3, (y == 3) & ns
P(f"SO recall all {(pr[so] == 3).mean():.3f} (n={so.sum()}), without snow {(pr[so2] == 3).mean():.3f} (n={so2.sum()})")
P(f"SO precision all {((pr == 3) & so).sum() / (pr == 3).sum():.3f}, without snow images "
  f"{((pr == 3) & so2).sum() / ((pr == 3) & ns).sum():.3f}")
P(f"without snow: n={ns.sum()} acc {ok[ns].mean():.4f} macroF1 {f1m(y[ns], pr[ns]):.4f}")
# katlar
P("per fold: n, acc, macroF1, SF/EF n")
for k in range(5):
    m = fold == k
    P(f"  fold {k}: n={m.sum()} acc {ok[m].mean():.4f} macroF1 {f1m(y[m], pr[m]):.4f} SF {int((y[m] == 2).sum())} "
      f"EF {int((y[m] == 0).sum())}")
ss = pd.Series(sg).value_counts()
big = ss.index[0]
mb = sg == big
P(f"largest scene group id {big}: size {ss.iloc[0]} folds {sorted(set(fold[mb]))} share of fold "
  f"{mb.sum() / (fold == fold[mb][0]).sum():.3f} (fold size {(fold == fold[mb][0]).sum()})")
P("  families:", pd.Series(fam[mb]).value_counts().to_dict(), "labels:", pd.Series(np.array(AB)[y[mb]]).value_counts().to_dict())
P(f"  E3 acc inside {ok[mb].mean():.4f}; pooled without it n={(~mb).sum()} acc {ok[~mb].mean():.4f} "
  f"macroF1 {f1m(y[~mb], pr[~mb]):.4f}")
P("  next largest groups:", ss.iloc[1:6].to_dict())

# ------------------------------------------------------------------ 2. secici tahmin
P("=" * 20, "2. SELECTIVE PREDICTION (LTT, replication of 22_secici_konformal.py)")


def ust(k, n, d):
    return 1.0 if n == 0 else (1.0 if k == n else beta.ppf(1 - d, k + 1, n - k))


def kosul(ad):
    if ad == "orij":
        d, Pm = d0, P0
        lab = d.label
    elif ad == "orijbel":
        d, Pm = d0, P0
        e = pd.read_csv(PIL / "asama2_sonuclar" / "orijinal_belirsizsiz_etiketler.csv").set_index("path")["etiket"]
        lab = d.label.index.map(lambda p: e.get(p, d.label[p])).to_series(index=d.index)
    else:
        d, Pm = load("runs_temiz", "kat_v1_asama2_temiz_none")
        e = pd.read_csv(PIL / "asama2_sonuclar" / "temiz_etiketler.csv").set_index("path")["etiket"]
        lab = d.label.index.map(lambda p: e.get(p, d.label[p])).to_series(index=d.index)
    tut = lab.isin(SIN).to_numpy()
    yy = lab[tut].map(SIN.index).to_numpy()
    g = lab.index[tut].map(M["sahne_grup"]).to_numpy()
    return Pm[tut], yy, g


def ltt(Pm, yy, g, delta=0.1, risk=0.05, nrep=1000, ayrinti=True):
    gu = np.unique(g)
    conf, pred = Pm.max(1), Pm.argmax(1)
    dg = pred == yy
    fa, pfa = np.isin(yy, [0, 2]), np.isin(pred, [0, 2])
    rng = np.random.default_rng(0)
    R = []
    for _ in range(nrep):
        cal = np.isin(g, rng.choice(gu, len(gu) // 2, replace=False))
        te = ~cal
        o = np.argsort(-conf[cal])
        hata = np.cumsum(~dg[cal][o])
        n = np.arange(1, cal.sum() + 1)
        son = None
        for c in np.arange(0.50, 1.0001, 0.05):
            i = int(c * len(n)) - 1
            if ust(hata[i], n[i], delta / 11) <= risk:
                son = i
        esik = conf[cal][o][son] if son is not None else 1.01
        sec = te & (conf >= esik)
        r = dict(kapsam=sec.sum() / te.sum(), acc=dg[sec].mean() if sec.any() else np.nan,
                 ihlal=(1 - dg[sec].mean()) > risk if sec.any() else False, esik=esik)
        if ayrinti:
            for k, a in enumerate(AB):
                m = te & (yy == k)
                r[f"kabul_{a}"] = (sec & m).sum() / m.sum()
                r[f"hata_{a}"] = (sec & m & ~dg).sum()
                r[f"nk_{a}"] = (sec & m).sum()
            r["kabul_ariza"] = (sec & fa).sum() / (te & fa).sum()
            r["ariza_kacan"] = (sec & fa & ~pfa).sum()  # kabul edilip ariza-yok denen gercek arizalar
            r["ariza_kabul_n"] = (sec & fa).sum()
            r["yanlis_alarm"] = (sec & ~fa & pfa).sum()
            r["kabul_n"] = sec.sum()
            r["te_n"] = te.sum()
            # LAC -> eylem
            sk = 1 - Pm[np.arange(len(yy)), yy]
            q = np.quantile(sk[cal], min(1, np.ceil((cal.sum() + 1) * 0.95) / cal.sum()), method="higher")
            K = (1 - Pm) <= q
            Kt, yt = K[te], yy[te]
            faultset = Kt[:, 0] | Kt[:, 2]
            only_h = Kt[:, 1] & ~Kt[:, 0] & ~Kt[:, 2] & ~Kt[:, 3]
            r["lac_noaction"] = only_h.mean()
            r["lac_clean"] = (~faultset & ~only_h & (Kt.sum(1) > 0)).mean()
            r["lac_priority"] = faultset.mean()
            r["lac_empty"] = (Kt.sum(1) == 0).mean()
            ft = np.isin(yt, [0, 2])
            r["lac_fault_flagged"] = faultset[ft].mean()
            r["lac_nonfault_flagged"] = faultset[~ft].mean()
            r["lac_cov"] = Kt[np.arange(len(yt)), yt].mean()
        R.append(r)
    return pd.DataFrame(R)


ref = {"orij": "orij_r0.05.csv", "orijbel": "orijbel_r0.05.csv", "temiz": "temiz_r0.05.csv"}
for ad in ("orij", "orijbel", "temiz"):
    Pm, yy, g = kosul(ad)
    R = ltt(Pm, yy, g)
    R.to_csv(OUT / f"ltt_sinif_{ad}.csv", index=False)
    rf = pd.read_csv(PIL / "asama2_sonuclar" / "secici_ek" / ref[ad])
    same = np.allclose(rf["kapsam"].to_numpy(), R["kapsam"].to_numpy()) if "kapsam" in rf else "?"
    P(f"[{ad}] n={len(yy)} coverage {R.kapsam.mean():.4f} +- {R.kapsam.std():.4f} acc {R.acc.mean():.4f} violation "
      f"{R.ihlal.mean():.3f} zero-coverage splits {(R.kapsam == 0).mean():.3f} | matches saved per-split coverage: {same}")
    P("   class counts in data:", dict(zip(AB, np.bincount(yy, minlength=4).tolist())))
    P("   mean acceptance rate by true class (all splits): " + " ".join(f"{a} {R[f'kabul_{a}'].mean():.3f}" for a in AB) +
      f" | faults (EF+SF) {R.kabul_ariza.mean():.3f}")
    nz = R[R.kapsam > 0]
    P("   same, splits with coverage>0 (n=%d): " % len(nz) + " ".join(f"{a} {nz[f'kabul_{a}'].mean():.3f}" for a in AB) +
      f" | faults {nz.kabul_ariza.mean():.3f}")
    P("   errors among accepted by true class, total over splits / accepted: " +
      " ".join(f"{a} {int(R[f'hata_{a}'].sum())}/{int(R[f'nk_{a}'].sum())}" for a in AB))
    P(f"   accepted true faults predicted as no-fault: {int(R.ariza_kacan.sum())} of {int(R.ariza_kabul_n.sum())} accepted faults "
      f"(sum over 1000 splits); per split mean {R.ariza_kacan.mean():.3f}; non-faults accepted as faults per split "
      f"{R.yanlis_alarm.mean():.3f}; accepted per split {R.kabul_n.mean():.1f} of {R.te_n.mean():.1f}")
    P(f"   LAC (eps 0.05) -> action: no action {{H}} {R.lac_noaction.mean():.3f}, cleaning check (H/SO sets) {R.lac_clean.mean():.3f}, "
      f"priority inspection (set has EF or SF) {R.lac_priority.mean():.3f}, empty {R.lac_empty.mean():.4f}; true faults flagged "
      f"{R.lac_fault_flagged.mean():.3f}; non-faults flagged {R.lac_nonfault_flagged.mean():.3f}; coverage {R.lac_cov.mean():.4f}")
    for dl in (0.05, 0.01):
        Rd = ltt(Pm, yy, g, delta=dl, ayrinti=False)
        P(f"   delta={dl}: coverage {Rd.kapsam.mean():.4f} +- {Rd.kapsam.std():.4f} acc {Rd.acc.mean():.4f} violation "
          f"{Rd.ihlal.mean():.3f} zero-coverage {(Rd.kapsam == 0).mean():.3f}")
# tek esik denetimi (hakemin yaklasimi)
conf = P0.max(1)
for cov in (0.571,):
    t = np.quantile(conf, 1 - cov)
    s = conf >= t
    P(f"single threshold on full OOF at coverage {cov}: t={t:.3f} accepted faults {int((s & yf).sum())}/{int(yf.sum())}, "
      f"H {(s & (y == 1)).sum() / (y == 1).sum():.3f}, SO {(s & (y == 3)).sum() / (y == 3).sum():.3f}")
Pm, yy, g = kosul("orijbel")
c2 = Pm.max(1)
t = np.quantile(c2, 1 - 0.816)
s = c2 >= t
f2 = np.isin(yy, [0, 2])
P(f"single threshold, ambiguous removed, coverage 0.816: t={t:.3f} accepted faults {int((s & f2).sum())}/{int(f2.sum())}, "
  f"accepted faults predicted no-fault {int((s & f2 & ~np.isin(Pm.argmax(1), [0, 2])).sum())}")

# ------------------------------------------------------------------ 3. cozunurluk
P("=" * 20, "3. NATIVE RESOLUTION (evaluation set)")
ev = M[(M.kat_v1 >= 0)]
P("evaluation images:", len(ev))
for lab_, g_ in [("all", ev)] + [(c, ev[ev.label == c]) for c in SIN]:
    w, h = g_.width.to_numpy(), g_.height.to_numpy()
    mn = np.minimum(w, h)
    P(f"  {lab_:20s} n={len(g_):4d} median WxH {int(np.median(w))}x{int(np.median(h))} | median long side "
      f"{int(np.median(np.maximum(w, h)))} | min width {w.min()} | short side <224: {(mn < 224).mean():.3f}, <392: "
      f"{(mn < 392).mean():.3f} | short side p10/p50/p90 {np.percentile(mn, [10, 50, 90]).astype(int).tolist()}")
for f in sorted(set(ev.aile)):
    g_ = ev[ev.aile == f]
    P(f"  family {FAM.get(f, f):14s} n={len(g_):4d} median WxH {int(g_.width.median())}x{int(g_.height.median())}")

# ------------------------------------------------------------------ 4. P1 ayristirma
P("=" * 20, "4. P1 DECOMPOSITION")
P1 = {"convnext_tiny": "official_convnext_tiny_224_s42", "swin_tiny": "official_swin_tiny_patch4_window7_224_224_s42",
      "dinov2": "official_vit_base_patch14_dinov2_224_s42", "efficientnetv2_s": "official_tf_efficientnetv2_s_224_s42",
      "resnetv2_101": "official_resnetv2_101_224_s42"}
P3 = {"convnext_tiny": "convnext_tiny_224", "swin_tiny": "swin_tiny_patch4_window7_224_224",
      "dinov2": "vit_base_patch14_dinov2_224", "efficientnetv2_s": "tf_efficientnetv2_s_224", "resnetv2_101": "resnetv2_101_224"}
Mall = M.copy()
fg = Mall.groupby("foto_grup").split.agg(lambda s: set(s))
test = Mall[Mall.split == "test"]
rgb_test = test[~test.termal]
nf_tr = rgb_test.foto_grup.map(lambda g: "train" not in fg[g])
nf_trva = rgb_test.foto_grup.map(lambda g: not ({"train", "valid"} & fg[g]))
P(f"official test: {len(test)} files, RGB {len(rgb_test)}; RGB test files with no same-photo copy in train: {int(nf_tr.sum())} "
  f"{rgb_test[nf_tr].label.map(dict(zip(SIN, AB))).value_counts().to_dict()}; none in train or valid: {int(nf_trva.sum())} "
  f"{rgb_test[nf_trva].label.map(dict(zip(SIN, AB))).value_counts().to_dict()}")
kg = Mall.groupby("kati_grup").split.agg(lambda s: set(s))
nk = rgb_test.kati_grup.map(lambda g: not ({"train", "valid"} & kg[g]))
P(f"RGB test files with no strict (MD5/dihedral pHash<=4) copy in train or valid: {int(nk.sum())}")
lf_paths = set(rgb_test[nf_trva].index)
# P3 degerlendirme kumesinde: foto grubu yalniz test dosyalarindan olusan temsilciler
evset = Mall[Mall.kat_v1 >= 0]
ev_test_only = evset[evset.foto_grup.map(lambda g: fg[g] == {"test"})]
ev_from_test = evset[evset.split == "test"]
P(f"evaluation-set representatives that are official test files: {len(ev_from_test)}; whose photo group holds only test files: "
  f"{len(ev_test_only)} {ev_test_only.label.map(dict(zip(SIN, AB))).value_counts().to_dict()}")
for k, run in P1.items():
    t1 = pd.read_csv(PIL / "runs" / run / "test_predictions.csv").set_index("path")
    c1 = t1.label == t1.tahmin
    term = np.asarray(t1.index.map(Mall["termal"]).astype(bool))  # not: bolum 4 icin r1_a4_p1.py esas alinir
    a_all, a_rgb = c1.mean(), c1[~term].mean()
    a_lf = c1[c1.index.isin(lf_paths)].mean()
    a_evt = c1[c1.index.isin(ev_test_only.index)].mean()
    a_evf = c1[c1.index.isin(ev_from_test.index)].mean()
    d3 = pd.concat([pd.read_csv(PIL / "runs_asama2" / f"grouped{f}_{P3[k]}_s42_kat_v1_asama2_none" / "test_predictions.csv")
                    for f in range(5)]).set_index("path")
    c3 = d3.label == d3.tahmin
    b_all = c3.mean()
    b_evt = c3[c3.index.isin(ev_test_only.index)].mean()
    b_evf = c3[c3.index.isin(ev_from_test.index)].mean()
    P(f"  {k:17s} P1(no TTA): all {a_all:.4f} | RGB-only {a_rgb:.4f} (n={int((~term).sum())}) | RGB no same-photo copy in "
      f"train/valid {a_lf:.4f} (n={int(c1.index.isin(lf_paths).sum())}) | eval-set reps from test-only photo groups {a_evt:.4f} "
      f"(n={int(c1.index.isin(ev_test_only.index).sum())}) | eval-set reps that are test files {a_evf:.4f} || P3 s42 OOF: all "
      f"{b_all:.4f} | same test-only reps {b_evt:.4f} | same test-file reps {b_evf:.4f}")

(OUT / "r1_a_sonuc.txt").write_text("\n".join(L), encoding="utf-8")
