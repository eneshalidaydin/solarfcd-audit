"""Adım 31 - Renewable Energy revizyonu için ek analizler ve 12 şekil (kullanıcı isteği 22 Kasım; yeni eğitim yok,
yalnız mevcut çıktılardan + çıkarım gecikmesi ölçümü). Çıktılar: makale/fig01..fig12 (*.png, 300 dpi),
makale/ek_sayilar.json (metinde kullanılacak sayılar)."""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageOps

OUT = Path("makale")
SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
K = ["EF", "H", "SF", "SO"]
PC = [f"p_{c}" for c in SIN]
E3 = ["convnext_tiny_224", "swin_tiny_patch4_window7_224_224", "vit_base_patch14_dinov2_392"]
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, C4 = "#3b6ea5", "#d9822b", "#2e8b57", "#c0392b"
SAY = {}
rng = np.random.default_rng(0)


def kaydet(ad):
    plt.savefig(OUT / ad, dpi=300, bbox_inches="tight")
    plt.close()


def oof(kok, uyeler, son, tohumlar=(42, 43, 44)):
    D = [pd.concat([pd.read_csv(f"{kok}/grouped{k}_{u}_s{s}_{son}/test_predictions.csv") for k in range(5)])
         .set_index("path").sort_index() for u in uyeler for s in tohumlar]
    return np.mean([d[PC].to_numpy() for d in D], 0), D[0]


M = pd.read_csv("asama2/manifest_asama2.csv")
R = M[M.temsilci & (M.kat_v1 >= 0)].set_index("path")
AILE = {"Imgclean": "K-clean", "Imgdirty": "K-dirty", "Clean": "PD-clean", "Dust": "PD-dust", "Bird": "PD-bird",
        "Snow": "PD-snow", "Electrical": "PD-electrical", "Physical": "PD-physical"}
Po, Do = oof("runs_asama2", E3, "kat_v1_asama2_none")
yo = Do.label.map(SIN.index).to_numpy()
po = Po.argmax(1)
g = R.loc[Do.index, "sahne_grup"].to_numpy()
gu, gi = np.unique(g, return_inverse=True)
BOOT = [np.bincount(rng.choice(len(gu), len(gu)), minlength=len(gu))[gi] for _ in range(1000)]

# ---------- sayılar: arıza var/yok, aile doğruluğu, iş yükü, ECE ----------
fy, fp = np.isin(yo, [0, 2]), np.isin(po, [0, 2])
SAY["ariza"] = dict(acc=float((fy == fp).mean()), recall=float((fy & fp).sum() / fy.sum()),
                    precision=float((fy & fp).sum() / fp.sum()), baseline=float(1 - fy.mean()), n_fault=int(fy.sum()))
SAY["E3_orij"] = dict(acc=float((po == yo).mean()), recall={k: float((po[yo == i] == i).mean()) for i, k in enumerate(K)})
aile = R.loc[Do.index, "aile"].map(AILE).to_numpy()
fam = {}
for a in sorted(set(aile)):
    s = aile == a
    d = po[s] == yo[s]
    b = [np.average(d, weights=w[s]) if w[s].sum() > 0 else np.nan for w in BOOT]
    fam[a] = dict(n=int(s.sum()), acc=float(d.mean()), lo=float(np.nanpercentile(b, 2.5)), hi=float(np.nanpercentile(b, 97.5)))
SAY["aile_acc"] = fam


def ece(P, y, nb=10):
    c, d = P.max(1), P.argmax(1) == y
    b = np.minimum((c * nb).astype(int), nb - 1)
    e, rows = 0.0, []
    for i in range(nb):
        s = b == i
        if s.any():
            e += s.mean() * abs(d[s].mean() - c[s].mean())
            rows.append((c[s].mean(), d[s].mean(), int(s.sum())))
    return e, rows


Pt, Dt = oof("runs_temiz", E3, "kat_v1_asama2_temiz_none")
yt = Dt.label.map(SIN.index).to_numpy()
MT = pd.read_csv("asama2_temiz/manifest_asama2.csv").set_index("path")
amb = MT.loc[Do.index, "kat_v1"].to_numpy() < 0
EGRI = {}
for ad, P_, y_ in [("Original labels (n = 2,095)", Po, yo), ("Original labels, ambiguous removed (n = 2,049)", Po[~amb], yo[~amb]),
                   ("Expert-audited labels (n = 2,049)", Pt, yt)]:
    o = np.argsort(-P_.max(1))
    d = (P_.argmax(1) == y_)[o]
    cov = np.arange(1, len(d) + 1) / len(d)
    acc = np.cumsum(d) / np.arange(1, len(d) + 1)
    EGRI[ad] = (cov, acc)
    SAY.setdefault("is_yuku", {})[ad] = {f"{t:.2f}": dict(kapsam=float(cov[acc >= t].max()) if (acc >= t).any() else 0.0,
                                                          elle_1000=int(round(1000 * (1 - (cov[acc >= t].max() if (acc >= t).any() else 0)))))
                                         for t in (0.95, 0.97, 0.99)}

# ---------- Şekil 1: çalışma tasarımı ----------
fig, ax = plt.subplots(figsize=(7.2, 5.6))
ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)


def kutu(x, y, w, h, t, c="#eef3f9", ec=C1, fs=7.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6", fc=c, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)


def ok(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.9))


kutu(2, 84, 40, 13, "SolarFCD release\n3,742 files (train 3,030 / val 337 / test 375)\n4 classes: EF, H, SF, SO")
kutu(2, 58, 40, 20, "Duplicate and scene audit\nMD5 · dihedral pHash · RootSIFT + MAGSAC\nDINOv2 · EXIF · visual confirmation\n2,562 unique contents (31.5% redundant)\n407 thermal + 93 conflicting files excluded")
kutu(2, 36, 40, 16, "Evaluation set: 2,095 RGB images\nscene-grouped 5-fold CV\nH 1,178 · SO 812 · EF 71 · SF 34\nresidual leakage audit: 5/160 (3.1%)")
kutu(2, 14, 40, 16, "Protocols P1 (official split)\nP2 (random folds) · P3 (scene-grouped)\n6 backbones × 3 seeds, 30 epochs\npooled OOF + cluster bootstrap")
ok(22, 84, 22, 78.6); ok(22, 58, 22, 52.6); ok(22, 36, 22, 30.6)
R5 = [(82, "Expert label audit\n150 images (112 flagged + 38 random)\n3 blinded PV experts, Fleiss κ"),
      (64, "Risk-controlled selective prediction\nLearn-then-Test (α = 5%, δ = 0.1)\nLAC conformal sets (95%)"),
      (46, "Overlap-screened external test\nKaggle: 98.5% overlap → rejected\nRF-Fault 1,478 · RF-Soiling 137"),
      (28, "Few-shot target adaptation\n5 / 10 / 20% labelled target images\nSolarFCD vs ImageNet initialisation"),
      (10, "Explainability\nGrad-CAM on correct and\nH↔SO error cases")]
for y, t in R5:
    kutu(56, y, 41, 14, t, c="#f6efe6", ec=C2)
    ok(42.6, 44, 55.4, y + 7)
kaydet("fig01_study_design.png")

# ---------- Şekil 2: örnekler ve sızıntı türleri ----------
K_ = pd.read_csv("asama2/kenarlar.csv")
MR = M.set_index("rel_path")
sec = []
for c in SIN:
    r = R[(R.label == c)].sample(1, random_state=3).iloc[0]
    sec.append((r.rel_path, f"(a) {K[SIN.index(c)]} example"))
d = M[~M.termal]
dup = d[d.md5.duplicated(keep=False)]
par = None
for md, gr in dup.groupby("md5"):
    if {"test", "train"} <= set(gr.split) and gr.dosya.str[:4].nunique() > 1:
        par = (gr[gr.split == "test"].iloc[0].rel_path, gr[gr.split == "train"].iloc[0].rel_path); break
sec += [(par[0], "(b) Official TEST file"), (par[1], "(b) Byte-identical TRAIN file\n(different name)")]
fg = K_[(K_.tur == "foto_geometrik")]
fg = fg[[MR.loc[a, "split"] != MR.loc[b, "split"] and MR.loc[a, "md5"] != MR.loc[b, "md5"] and not MR.loc[a, "termal"]
         for a, b in zip(fg.rel_i, fg.rel_j)]].iloc[0]
sec += [(fg.rel_i, f"(c) Near-duplicate ({MR.loc[fg.rel_i, 'split'].upper()})"),
        (fg.rel_j, f"(c) Near-duplicate ({MR.loc[fg.rel_j, 'split'].upper()},\ncollage tile)")]
sg = K_[K_.tur.str.startswith("sahne")]
sg = sg[[{MR.loc[a, "label"], MR.loc[b, "label"]} == {"Healthy", "Surface_Obstruction"} for a, b in zip(sg.rel_i, sg.rel_j)]].iloc[0]
sec += [(sg.rel_i, f"(d) Same scene, labelled {K[SIN.index(MR.loc[sg.rel_i, 'label'])]}"),
        (sg.rel_j, f"(d) Same scene, labelled {K[SIN.index(MR.loc[sg.rel_j, 'label'])]}")]
CG = pd.read_csv("asama2/celiskili_gruplar.csv")
tg = CG[CG.termal].groupby("foto_grup").filter(lambda x: x.label.nunique() > 1)
tg = tg[tg.foto_grup == tg.foto_grup.iloc[0]]
for _, r in tg.drop_duplicates("label").head(2).iterrows():
    sec.append((r.rel_path, f"(e) Identical thermal image,\nlabelled {K[SIN.index(r.label)]}"))
fig, axs = plt.subplots(3, 4, figsize=(7.2, 6.3))
for a, (rp, t) in zip(axs.flat, sec):
    im = ImageOps.exif_transpose(Image.open(MR.loc[rp, "path"])).convert("RGB")
    im.thumbnail((400, 400))
    a.imshow(im); a.set_title(t, fontsize=7); a.axis("off")
for a in axs.flat[len(sec):]:
    a.axis("off")
plt.tight_layout(); kaydet("fig02_samples_and_leakage.png")
SAY["fig2_dosyalar"] = sec

# ---------- Şekil 3: veri bileşimi ----------
fig = plt.figure(figsize=(7.2, 5.4))
a1 = fig.add_subplot(2, 2, 1)
yay = M.label.value_counts().reindex(SIN)
tek = M[~M.termal].drop_duplicates("md5").label.value_counts().reindex(SIN)
son = R.label.value_counts().reindex(SIN)
x = np.arange(4)
for j, (v, l, c) in enumerate([(yay, "Released files", "#9e9e9e"), (tek, "Unique RGB contents", C2), (son, "Evaluation set", C1)]):
    a1.bar(x + (j - 1) * 0.27, v.values, 0.27, label=l, color=c)
a1.set_xticks(x, K); a1.set_ylabel("Images"); a1.legend(frameon=False, fontsize=7); a1.set_title("(a) Class counts", loc="left", fontsize=9)
a2 = fig.add_subplot(2, 2, 2)
ct = pd.crosstab(R.aile.map(AILE), R.label).reindex(columns=SIN, fill_value=0)
a2.imshow(np.log1p(ct.to_numpy()), cmap="Blues", aspect="auto")
for i in range(ct.shape[0]):
    for j in range(4):
        if ct.iat[i, j]:
            a2.text(j, i, ct.iat[i, j], ha="center", va="center", fontsize=7)
a2.set_xticks(range(4), K); a2.set_yticks(range(ct.shape[0]), ct.index, fontsize=7)
a2.set_title("(b) Source family × class (n = 2,095)", loc="left", fontsize=9)
a3 = fig.add_subplot(2, 1, 2)
gs = R.groupby("sahne_grup").size()
b = np.unique(np.logspace(0, np.log10(gs.max() + 1), 25).astype(int))
a3.hist(gs, bins=b, color=C1); a3.set_xscale("log"); a3.set_yscale("log")
a3.set_xlabel("Scene-group size (images)"); a3.set_ylabel("Number of scene groups")
a3.set_title(f"(c) Scene groups: {len(gs):,}; median size {int(gs.median())}; largest {gs.max()}", loc="left", fontsize=9)
plt.tight_layout(); kaydet("fig03_data_composition.png")
SAY["sahne_grup"] = dict(n=int(len(gs)), medyan=float(gs.median()), en_buyuk=int(gs.max()), tekil=int((gs == 1).sum()))
SAY["tekil_rgb_icerik"] = tek.to_dict()

# ---------- Şekil 4: değerlendirme ortamına göre doğruluk ----------
def wilson(k, n, z=1.96):
    p = k / n; m = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return m - h, m + h


def kume_acc(kok, son):
    D = pd.concat([pd.read_csv(f"{kok}/grouped{k}_vit_base_patch14_dinov2_224_s42_{son}/test_predictions.csv") for k in range(5)]).set_index("path").sort_index()
    d = (D.tahmin == D.label).to_numpy()
    gg = np.unique(R.loc[D.index, "sahne_grup"].to_numpy(), return_inverse=True)[1]
    b = [np.average(d, weights=np.bincount(rng.choice(gg.max() + 1, gg.max() + 1), minlength=gg.max() + 1)[gg]) for _ in range(1000)]
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


v = [(0.976, *wilson(366, 375))]  # P1: Wilson (iyimser; test içi kopyalar)
v.append(kume_acc("runs_rastgele", "kat_rastgele_asama2_rastgele_none"))
v.append(kume_acc("runs_asama2", "kat_v1_asama2_none"))
v.append((0.7067, *wilson(round(0.7067 * 1108), 1108)))
v.append((0.2847, *wilson(39, 137)))
lab = ["P1 official split\n(duplicates kept)", "P2 random 5-fold\n(scene leakage)", "P3 scene-grouped\n5-fold", "RF-Fault\n(external)", "RF-Soiling\n(external, SO only)"]
fig, ax = plt.subplots(figsize=(7.2, 3.3))
m_ = np.array([a for a, _, _ in v]); lo = m_ - np.array([b for _, b, _ in v]); hi = np.array([c for _, _, c in v]) - m_
ax.bar(lab, m_, yerr=[lo, hi], capsize=3, color=["#9e9e9e", "#bdbdbd", C1, C2, C4])
for i, t in enumerate(m_):
    ax.text(i, t + hi[i] + 0.02, f"{t:.3f}", ha="center")
ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy (DINOv2 ViT-B/14, 224 px, seed 42)")
kaydet("fig04_evaluation_settings.png")
SAY["fig4"] = [list(map(float, t)) for t in v]

# ---------- Şekil 5: karışıklık matrisleri + aile doğruluğu ----------
fig = plt.figure(figsize=(7.2, 5.6))
for j, (P_, y_, t) in enumerate([(Po, yo, "(a) E3, original labels (n = 2,095)"), (Pt, yt, "(b) E3, expert-audited labels (n = 2,049)")]):
    a = fig.add_subplot(2, 2, j + 1)
    cm = np.zeros((4, 4), int); np.add.at(cm, (y_, P_.argmax(1)), 1)
    cn = cm / cm.sum(1, keepdims=True)
    a.imshow(cn, cmap="Blues", vmin=0, vmax=1)
    for r_ in range(4):
        for c_ in range(4):
            a.text(c_, r_, f"{cm[r_, c_]}\n{cn[r_, c_]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if cn[r_, c_] > 0.6 else "black")
    a.set_xticks(range(4), K); a.set_yticks(range(4), K); a.set_xlabel("Predicted"); a.set_ylabel("Label"); a.set_title(t, loc="left", fontsize=8.5)
    SAY[f"cm_{j}"] = cm.tolist()
a = fig.add_subplot(2, 1, 2)
fs_ = sorted(fam, key=lambda k: fam[k]["acc"])
a.barh(fs_, [fam[k]["acc"] for k in fs_], xerr=[[fam[k]["acc"] - fam[k]["lo"] for k in fs_], [fam[k]["hi"] - fam[k]["acc"] for k in fs_]],
       color=[C4 if fam[k]["acc"] < 0.85 else C1 for k in fs_], capsize=2)
for i, k in enumerate(fs_):
    a.text(1.005, i, f"n = {fam[k]['n']}", va="center", fontsize=7)
a.set_xlim(0.5, 1.08); a.set_xlabel("Accuracy of E3 by source family (95% cluster-bootstrap CI)"); a.set_title("(c)", loc="left", fontsize=8.5)
plt.tight_layout(); kaydet("fig05_confusion_and_sources.png")

# ---------- Şekil 6: uzman denetimi ----------
A = pd.read_csv("insan_kontrol/etiket/sonuc.csv", encoding="utf-8-sig")
SEC = SIN + ["karisik", "emin_degil"]
SL = K + ["mixed", "not sure"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
alt = np.zeros(3)
renk = [C4, "#7fb069", "#8e6c8a", C2, "#bdbdbd", "#5d5d5d"]
for s_, l_, c_ in zip(SEC, SL, renk):
    v_ = np.array([(A[f"uzman{u}"] == s_).sum() for u in (1, 2, 3)])
    axs[0].bar(["Expert 1", "Expert 2", "Expert 3"], v_, bottom=alt, label=l_, color=c_); alt += v_
axs[0].set_ylabel("Images (of 150)"); axs[0].legend(frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
axs[0].set_title("(a) Label distribution per expert", loc="left", fontsize=8.5)
hed = SIN + ["belirsiz"]
ct = pd.crosstab(A.label, A.uzlasi).reindex(index=SIN, columns=hed, fill_value=0)
axs[1].imshow(ct.to_numpy(), cmap="Oranges")
for i in range(4):
    for j in range(5):
        axs[1].text(j, i, ct.iat[i, j], ha="center", va="center", fontsize=8)
axs[1].set_xticks(range(5), K + ["ambig."]); axs[1].set_yticks(range(4), K)
axs[1].set_xlabel("Expert consensus (≥2 of 3)"); axs[1].set_ylabel("Original label")
axs[1].set_title("(b) Original label → consensus (n = 150)", loc="left", fontsize=8.5)
plt.tight_layout(); kaydet("fig06_expert_audit.png")
SAY["uzman_dagilim"] = {u: A[u].value_counts().to_dict() for u in ("uzman1", "uzman2", "uzman3")}
SAY["gecis"] = ct.to_dict()

# ---------- Şekil 7: seçici tahmin ----------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1.35, 1]))
RS = {"Original labels (n = 2,095)": "orij", "Original labels, ambiguous removed (n = 2,049)": "orijbel", "Expert-audited labels (n = 2,049)": "temiz"}
for (ad, (cov, acc)), c in zip(EGRI.items(), [C1, C2, C3]):
    axs[0].plot(cov[20:], acc[20:], color=c, label=ad, lw=1.2)
    r_ = pd.read_csv(f"asama2_sonuclar/secici_ek/{RS[ad]}_r0.05.csv")
    axs[0].plot(r_.kapsam.mean(), r_.acc.mean(), "o", color=c, ms=5, mec="black", mew=0.5)
axs[0].axhline(0.95, ls="--", color="grey", lw=0.8)
axs[0].set_ylim(0.88, 1.002); axs[0].set_xlabel("Coverage"); axs[0].set_ylabel("Accuracy on accepted images")
axs[0].legend(frameon=False, fontsize=6.3, loc="lower left"); axs[0].set_title("(a) Accuracy–coverage (dots: LTT, α = 5%)", loc="left", fontsize=8.5)
data = [pd.read_csv(f"asama2_sonuclar/secici_ek/{v_}_r0.05.csv").kapsam for v_ in RS.values()]
axs[1].boxplot(data, widths=0.5, showfliers=True, flierprops=dict(ms=2))
axs[1].set_xticks([1, 2, 3], ["Original", "Ambiguous\nremoved", "Audited"]); axs[1].set_ylabel("Certified coverage over 1,000 splits")
axs[1].set_title("(b) Stability of LTT coverage", loc="left", fontsize=8.5)
plt.tight_layout(); kaydet("fig07_selective_prediction.png")

# ---------- Şekil 8: güvenilirlik diyagramı ----------
def dis(veri):
    T = pd.read_csv(f"dis_dogrulama/{veri}/ortusme.csv", encoding="utf-8-sig")
    T = T[~T.ortusen].reset_index(drop=True)
    y = T.label.map(lambda s: SIN.index(s) if s in SIN else -1).to_numpy()
    P = np.mean([np.load(f"dis_dogrulama/{veri}/tahminler_{a}.npy") for a in ("ConvNeXt", "Swin", "DINOv2-392")], 0)
    return P[y >= 0], y[y >= 0]


Pf, yf = dis("roboflow")
Ps, ys = dis("melbourne")
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
for a, (P_, y_, t) in zip(axs, [(Po, yo, "SolarFCD OOF"), (Pf, yf, "RF-Fault"), (Ps, ys, "RF-Soiling")]):
    e, rows = ece(P_, y_)
    a.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    cc, aa, nn = zip(*rows)
    a.plot(cc, aa, "o-", color=C1, ms=3)
    a.set_title(f"{t}\nECE = {e:.3f}, accuracy = {(P_.argmax(1) == y_).mean():.3f}", fontsize=8)
    a.set_xlabel("Mean confidence")
    SAY.setdefault("ece", {})[t] = float(e)
axs[0].set_ylabel("Observed accuracy")
plt.tight_layout(); kaydet("fig08_reliability.png")

# ---------- Şekil 9: örtüşme taraması ----------
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
for a, (f, t) in zip(axs, [("dis_dogrulama/ortusme.csv", "Kaggle PV Panel Defect"), ("dis_dogrulama/roboflow/ortusme.csv", "RF-Fault"),
                           ("dis_dogrulama/melbourne/ortusme.csv", "RF-Soiling")]):
    X = pd.read_csv(f, encoding="utf-8-sig")
    a.scatter(X.ph_min[~X.ortusen], X.cos_max[~X.ortusen], s=3, color=C1, alpha=0.5, label="retained")
    a.scatter(X.ph_min[X.ortusen & ~X.md5_eslesme], X.cos_max[X.ortusen & ~X.md5_eslesme], s=3, color=C2, alpha=0.5, label="removed")
    a.scatter(X.ph_min[X.md5_eslesme], X.cos_max[X.md5_eslesme], s=3, color=C4, alpha=0.5, label="byte-identical")
    a.axvline(12, ls="--", color="grey", lw=0.7); a.axhline(0.85, ls="--", color="grey", lw=0.7)
    a.set_title(f"{t}\nremoved {int(X.ortusen.sum())} / {len(X)}", fontsize=8); a.set_xlabel("Min pHash distance to SolarFCD")
axs[0].set_ylabel("Max DINOv2 cosine to SolarFCD"); axs[0].legend(frameon=False, fontsize=6.5, markerscale=3)
plt.tight_layout(); kaydet("fig09_overlap_screening.png")

# ---------- Şekil 10: dış sınıf duyarlılığı + sızıntılı/gruplu ----------
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
rin = [(po[yo == i] == i).mean() for i in (1, 2, 3)]
rex = [(Pf.argmax(1)[yf == i] == i).mean() for i in (1, 2, 3)]
x = np.arange(3)
axs[0].bar(x - 0.2, rin, 0.4, color=C1, label="SolarFCD (OOF)"); axs[0].bar(x + 0.2, rex, 0.4, color=C2, label="RF-Fault")
axs[0].set_xticks(x, ["H", "SF", "SO"]); axs[0].set_ylabel("Recall (E3)"); axs[0].set_ylim(0, 1.05); axs[0].legend(frameon=False, fontsize=7)
axs[0].set_title("(a) Per-class recall", loc="left", fontsize=8.5)
SAY["recall_ic_dis"] = dict(ic=list(map(float, rin)), dis=list(map(float, rex)))
ic = {"ConvNeXt": (0.9103, 0.9002), "Swin": (0.9084, 0.8998), "DINOv2-224": (0.9103, 0.8974)}
dx = {"ConvNeXt": (0.6372, 0.6074), "Swin": (0.5821, 0.5912), "DINOv2-224": (0.6778, 0.7067)}
x = np.arange(3); w = 0.2
for j, (lab_, vals, c) in enumerate([("Internal CV, P2 (leaky)", [ic[k][0] for k in ic], "#bdbdbd"), ("Internal CV, P3 (grouped)", [ic[k][1] for k in ic], C1),
                                      ("RF-Fault, P2 models", [dx[k][0] for k in dx], "#f0b27a"), ("RF-Fault, P3 models", [dx[k][1] for k in dx], C2)]):
    axs[1].bar(x + (j - 1.5) * w, vals, w, label=lab_, color=c)
axs[1].set_xticks(x, list(ic)); axs[1].set_ylim(0, 1.05); axs[1].set_ylabel("Accuracy (seed 42)")
axs[1].legend(frameon=False, fontsize=6.3, loc="lower center", ncol=2); axs[1].set_title("(b) Leakage vs domain shift", loc="left", fontsize=8.5)
plt.tight_layout(); kaydet("fig10_external_validation.png")

# ---------- Şekil 11-12: mevcut ----------
Rf = pd.read_csv("asama2_sonuclar/az_ornek.csv")
fig, ax = plt.subplots(figsize=(4.6, 3.2))
for bas, ad, c in [("solarfcd", "Initialised from SolarFCD model", C1), ("imagenet", "Initialised from ImageNet only", C2)]:
    gg_ = Rf[Rf.baslangic == bas].groupby("oran").agg(n=("n_egitim", "mean"), m=("acc", "mean"), s=("acc", "std")).fillna(0)
    ax.errorbar(gg_.n, gg_.m, yerr=gg_.s, marker="o", capsize=3, label=ad, color=c)
ax.set_xlabel("Labelled target-domain training images"); ax.set_ylabel("Accuracy, RF-Fault test (n = 552)")
ax.set_ylim(0.55, 1.0); ax.legend(frameon=False, loc="lower right", fontsize=7)
kaydet("fig11_few_shot.png")
shutil.copy("asama2_sonuclar/gradcam.png", OUT / "fig12_gradcam.png")

# ---------- hesaplama maliyeti ----------
import timm
import torch
MD = {"ResNetV2-101": ("resnetv2_101", 224, "resnetv2_101_224"), "EfficientNetV2-S": ("tf_efficientnetv2_s", 224, "tf_efficientnetv2_s_224"),
      "ConvNeXt-T": ("convnext_tiny", 224, "convnext_tiny_224"), "ConvNeXt V2-T": ("convnextv2_tiny", 224, "convnextv2_tiny_224"),
      "Swin-T": ("swin_tiny_patch4_window7_224", 224, "swin_tiny_patch4_window7_224_224"),
      "DINOv2 ViT-B/14 (224)": ("vit_base_patch14_dinov2", 224, "vit_base_patch14_dinov2_224"),
      "DINOv2 ViT-B/14 (392)": ("vit_base_patch14_dinov2", 392, "vit_base_patch14_dinov2_392"),
      "ConvNeXt-T (384)": ("convnext_tiny", 384, "convnext_tiny_384")}
mal = {}
for ad, (mdl, S_, kisa) in MD.items():
    m = timm.create_model(mdl, pretrained=False, num_classes=4, **({"img_size": S_} if "dinov2" in mdl else {})).cuda().eval()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    lg = list(Path("runs_asama2").glob(f"grouped0_{kisa}_s42_*/log.csv"))
    t = float(pd.read_csv(lg[0]).sure_sn.sum()) if lg else float("nan")
    x = torch.randn(1, 3, S_, S_, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(20):
            m(x)
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record()
        for _ in range(100):
            m(x); m(x.flip(3)); m(x.flip(2))
        ev1.record(); torch.cuda.synchronize()
    mal[ad] = dict(param_M=round(n, 1), egitim_30ep_s=round(t), gecikme_ms_tta=round(ev0.elapsed_time(ev1) / 100, 1))
    del m; torch.cuda.empty_cache()
tot = sum(pd.read_csv(f).sure_sn.sum() for k in ("runs", "runs_asama2", "runs_rastgele", "runs_temiz", "runs_lr") for f in Path(k).glob("*/log.csv"))
SAY["maliyet"] = mal
SAY["toplam_gpu_saat"] = round(tot / 3600, 1)
json.dump(SAY, open(OUT / "ek_sayilar.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: SAY[k] for k in ("ariza", "E3_orij", "aile_acc", "is_yuku", "ece", "sahne_grup", "recall_ic_dis", "maliyet", "toplam_gpu_saat", "fig4")}, indent=1, default=str))
