"""Adım 31 - Renewable Energy gönderimi için ek analizler ve 12 şekil (kullanıcı isteği 22 Kasım; yeni eğitim yok,
yalnız mevcut çıktılardan + çıkarım gecikmesi ölçümü). Çıktılar: makale/fig01..fig12 (*.png, 300 dpi),
makale/ek_sayilar.json (metinde kullanılacak sayılar).
Gönderim öncesi iç değerlendirme O1 (24 Kasım): Şekil 1'e O&M şeridi; Şekil 4'te P1 yalnız RGB (TTA), RF-Soiling ayrı panelde SO
duyarlılığı, dış GA kök-ad kümesi bootstrap; Şekil 7b histogram; fig08'e kutu sayıları; fig10'a %95 GA; fig12 =
29b_gradcam_v2.py çıktısı; yeni makale/figS3_training_curves.png. Yeni bootstrap'lar ayrı rng kullanır (eski sayılar aynı)."""
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


def pdf_kaydet(ad, fig=None):
    """Dergiye ayrı yüklenecek vektörel sürüm (SEKIL_PDF=1 iken), makale/yuksek/<ad>.pdf; PNG çıktısı değişmez."""
    import os
    if not os.environ.get("SEKIL_PDF"):
        return
    (OUT / "yuksek").mkdir(exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42}):
        (fig or plt.gcf()).savefig(OUT / "yuksek" / (Path(ad).stem + ".pdf"), bbox_inches="tight", dpi=600)


def kaydet(ad):
    plt.savefig(OUT / ad, dpi=300, bbox_inches="tight")
    pdf_kaydet(ad)
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
# 25 Kasım: alttaki O&M şeridi yeni akış şemasına (37d_akis_semasi.py, makale/fig_pipeline.png) taşındı. Şekil basılı
# boyutta çizilir (13 cm genişlik, koordinatlar cm), tüm yazılar 7 pt; kutu boyları metin ölçüsünden hesaplanır.
GEN1, FS1, PAD1 = 13.0, 7.0, 0.07  # basılı genişlik (cm), yazı boyu (pt), kutu köşe dolgusu (cm)
W1 = GEN1 - 0.508  # bbox_inches="tight" her kenara 0,1 inç ekler
fig = plt.figure(figsize=(W1 / 2.54, 3.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
rnd1 = fig.canvas.get_renderer()


def olc1(t):  # metnin cm cinsinden genişliği ve yüksekliği
    tt = ax.text(0, 0, t, fontsize=FS1, ha="center", va="center")
    bb = tt.get_window_extent(rnd1)
    tt.remove()
    return bb.width / fig.dpi * 2.54, bb.height / fig.dpi * 2.54


def kutu(x, y, w, h, t, c="#eef3f9", ec=C1):  # (x, y, w, h): görünen kutu, köşe dolgusu içeride kalır
    ax.add_patch(FancyBboxPatch((x + PAD1, y + PAD1), w - 2 * PAD1, h - 2 * PAD1, boxstyle=f"round,pad={PAD1}",
                                fc=c, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=FS1)


def ok(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.9))


SOL1 = ["SolarFCD release\n3,742 files (train 3,030 / val 337 / test 375)\n4 classes: EF, H, SF, SO",
        "Duplicate and scene audit\nMD5 · dihedral pHash · RootSIFT + MAGSAC++\nDINOv2 · EXIF · visual confirmation\n"
        "2,562 unique contents (31.5% redundant)\n407 thermal + 93 conflicting files excluded",
        "Evaluation set: 2,095 RGB images\nscene-grouped 5-fold CV\nH 1,178 · SO 812 · EF 71 · SF 34\n"
        "residual leakage audit: 5/160 (3.1%)",
        "Protocols P1 (official split)\nP2 (random folds) · P3 (scene-grouped)\nP3: 6 backbones × 3 seeds, 30 epochs\n"
        "P1, P2: 5 backbones, seed 42"]
SAG1 = ["Expert label audit\n150 images (112 flagged + 38 random)\n3 blinded PV experts, Fleiss κ",
        "Risk-controlled selective prediction\nLearn-then-Test (α = 5%, δ = 0.1)\nLAC conformal sets (95%)",
        "Overlap-screened external test\nKaggle: 98.5% overlap → rejected\nRF-Fault 1,478 · RF-Soiling 137",
        "Few-shot target adaptation\n5 / 10 / 20% labelled target images\nSolarFCD vs ImageNet initialisation",
        "Explainability\nGrad-CAM on correct and\nH↔SO error cases"]
MX1, MY1, OKB1, KEN1 = 0.17, 0.12, 0.42, 0.04  # iç yatay/dikey boşluk, sol sütun ok boyu, dış kenar (cm)
oL, oR = [olc1(t) for t in SOL1], [olc1(t) for t in SAG1]
wL, wR = max(w for w, _ in oL) + 2 * MX1, max(w for w, _ in oR) + 2 * MX1
hL = [h + 2 * MY1 for _, h in oL]
hR = [max(h for _, h in oR) + 2 * MY1] * len(SAG1)
H1 = max(sum(hL) + OKB1 * (len(hL) - 1), sum(hR) + 0.2 * (len(hR) - 1)) + 2 * KEN1
assert H1 + 0.508 <= 9.0, f"Şekil 1 yüksekliği {H1 + 0.508:.2f} cm > 9 cm"
fig.set_size_inches(W1 / 2.54, H1 / 2.54)
ax.set_xlim(0, W1); ax.set_ylim(0, H1)
aL = (H1 - 2 * KEN1 - sum(hL)) / (len(hL) - 1)  # sol sütun ok boyu
aR = (H1 - 2 * KEN1 - sum(hR)) / (len(hR) - 1)  # sağ sütunda kutular arası boşluk
yy1, kL = H1 - KEN1, []
for t, h in zip(SOL1, hL):
    kutu(KEN1, yy1 - h, wL, h, t)
    kL.append((yy1 - h, h))
    yy1 -= h + aL
for (yu, _), (ya, hal) in zip(kL[:-1], kL[1:]):  # üst kutunun altından alttakinin üstüne
    ok(KEN1 + wL / 2, yu, KEN1 + wL / 2, ya + hal)
xR, yy1 = W1 - KEN1 - wR, H1 - KEN1
yk = kL[2][0] + kL[2][1] / 2  # yelpaze oklarının çıkışı: değerlendirme seti kutusunun sağ kenarı
for t, h in zip(SAG1, hR):
    kutu(xR, yy1 - h, wR, h, t, c="#f6efe6", ec=C2)
    ok(KEN1 + wL + 0.02, yk, xR - 0.03, yy1 - h / 2)
    yy1 -= h + aR
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
    a.imshow(im); a.set_title(t, fontsize=8.5); a.axis("off")
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
a1.set_xticks(x, K); a1.set_ylabel("Images"); a1.set_ylim(0, 1900)
a1.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.11), handlelength=1.2, columnspacing=1.2)
a1.set_title("(a) Class counts", loc="left", fontsize=9)
a2 = fig.add_subplot(2, 2, 2)
ct = pd.crosstab(R.aile.map(AILE), R.label).reindex(columns=SIN, fill_value=0)
lv = np.ma.masked_where(ct.to_numpy() == 0, np.log1p(ct.to_numpy()))
cmap_b = plt.get_cmap("Blues").copy(); cmap_b.set_bad("white")
a2.imshow(lv, cmap=cmap_b, aspect="auto", vmin=-2.5, vmax=np.log1p(ct.to_numpy().max()))
for i in range(ct.shape[0]):
    for j in range(4):
        if ct.iat[i, j]:
            a2.text(j, i, ct.iat[i, j], ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
a2.set_xticks(range(4), K); a2.set_yticks(range(ct.shape[0]), ct.index, fontsize=8.5)
a2.set_title("(b) Source family × class (n = 2,095)", loc="left", fontsize=9)
a3 = fig.add_subplot(2, 1, 2)
gs = R.groupby("sahne_grup").size()
b = np.unique(np.logspace(0, np.log10(gs.max() + 1), 25).astype(int))
a3.hist(gs, bins=b, color=C1); a3.set_xscale("log"); a3.set_yscale("log")
from matplotlib.ticker import FuncFormatter  # log eksen etiketleri 10^n yerine düz sayı (1, 10, 100, 1000)
a3.xaxis.set_major_formatter(FuncFormatter(lambda v_, _: f"{v_:g}")); a3.yaxis.set_major_formatter(FuncFormatter(lambda v_, _: f"{v_:g}"))
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


def kok_ad(p):
    """Roboflow kaynak dosya kökü: dosya adının '_jpg.rf.' ya da '.rf.' öncesi (aynı kaynaktan türetilmiş kopyalar)."""
    ad_ = Path(p).name
    for ayr in ("_jpg.rf.", ".rf."):
        if ayr in ad_:
            return ad_.split(ayr)[0]
    return Path(p).stem


RNG_KOK = np.random.default_rng(20261124)  # hakem_revizyon/analiz/r1_e_fig4.py ile aynı tohum ve çekiliş sırası


def kok_agirlik(kokler, B=2000):
    """Kök-ad küme bootstrap ağırlıkları (B x n): her yeniden örneklemde kökler yerine koyarak çekilir."""
    ku, ki = np.unique(kokler, return_inverse=True)
    sec_ = RNG_KOK.integers(0, len(ku), size=(B, len(ku)))
    W_ = np.zeros((B, len(ku)))
    np.add.at(W_, (np.arange(B)[:, None], sec_), 1)
    return W_[:, ki]


def ga(W_, d_, m_=None):
    """Ağırlıklı oran sum(w*d*m)/sum(w*m) için yüzdelik %95 bootstrap aralığı."""
    m_ = np.ones(len(d_)) if m_ is None else np.asarray(m_, float)
    b_ = (W_ @ (d_ * m_)) / (W_ @ m_)
    return float(np.nanpercentile(b_, 2.5)), float(np.nanpercentile(b_, 97.5))


def dis_tablo(veri):
    T = pd.read_csv(f"dis_dogrulama/{veri}/ortusme.csv", encoding="utf-8-sig")
    T = T[~T.ortusen].reset_index(drop=True)
    y = T.label.map(lambda s: SIN.index(s) if s in SIN else -1).to_numpy()
    return T[y >= 0].reset_index(drop=True), y[y >= 0], y >= 0


def dis_p(veri, ad_):  # 25_dis_dogrulama.py olasılıkları (5 kat modeli, TTA), eşlenmiş etiketli örtüşmeyen görüntüler
    return np.load(f"dis_dogrulama/{veri}/tahminler_{ad_}.npy")[dis_tablo(veri)[2]]


# P1 yalnız RGB, TTA'lı: görüntü düzeyinde TTA tahmini kaydedilmemiş. 11 termal test görüntüsünün tümü SF ve TTA karışıklık
# matrisinde SF satırının tümü doğru -> termallerin tümü TTA ile doğru; RGB doğru = iz(CM) - termal sayısı (355/364).
TR = pd.read_csv("runs/official_vit_base_patch14_dinov2_224_s42/test_predictions.csv", encoding="utf-8-sig")
term = M.set_index("path").loc[TR.path, "termal"].astype(bool).to_numpy()
CM = np.array(json.load(open("runs/official_vit_base_patch14_dinov2_224_s42/results.json", encoding="utf-8"))["test_tta"]["confusion_matrix"])
assert set(TR.label[term]) == {"Structural_Fault"} and CM[2, 2] == CM[2].sum() == (TR.label == "Structural_Fault").sum()
k1, n1 = int(np.trace(CM) - term.sum()), int((~term).sum())
v = [(k1 / n1, *wilson(k1, n1))]  # P1: yalnız RGB, TTA; Wilson (iyimser; test içi kopyalar)
v.append(kume_acc("runs_rastgele", "kat_rastgele_asama2_rastgele_none"))
v.append(kume_acc("runs_asama2", "kat_v1_asama2_none"))
TF, yF, _ = dis_tablo("roboflow")
TS, yS, _ = dis_tablo("melbourne")
kF = TF.path.map(kok_ad).to_numpy()
W_rf = kok_agirlik(kF)  # 1. çekiliş: RF-Fault (Şekil 4; Şekil 8'de E3 ve P3 modelleri)
W_rf2 = kok_agirlik(kF)  # 2. çekiliş: RF-Fault, P2 modelleri (Şekil 8b)
W_so = kok_agirlik(TS.path.map(kok_ad).to_numpy())  # 3. çekiliş: RF-Soiling
dF = (dis_p("roboflow", "DINOv2-224_sızıntısız_s42").argmax(1) == yF).astype(float)
v.append((dF.mean(), *ga(W_rf, dF)))
assert (yS == 3).all()  # RF-Soiling tek sınıflı (SO): doğruluk = SO duyarlılığı
dS = (dis_p("melbourne", "DINOv2-224_sızıntısız_s42").argmax(1) == 3).astype(float)
v.append((dS.mean(), *ga(W_so, dS)))
from matplotlib.ticker import PercentFormatter
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw=dict(width_ratios=[4.3, 1], wspace=0.06))
lab = ["P1\nofficial split,\nRGB only\n(duplicates kept)", "P2\nrandom folds\n(scene leakage)", "P3\nscene-grouped\nfolds",
       "RF-Fault\n(external,\nmapped labels)"]
m_ = np.array([a for a, _, _ in v]); lo = m_ - np.array([b for _, b, _ in v]); hi = np.array([c for _, _, c in v]) - m_
xs = np.arange(len(lab))
ax.bar(xs, m_[:4], 0.62, yerr=[lo[:4], hi[:4]], capsize=3, color=["#9e9e9e", "#bdbdbd", C1, C2])
ax2.bar([0], m_[4:], 0.62, yerr=[lo[4:], hi[4:]], capsize=3, color=C4)
for a_, i0, xx in [(ax, 0, xs), (ax2, 4, [0])]:
    for i, xi in enumerate(xx):
        a_.text(xi, m_[i0 + i] + hi[i0 + i] + 0.02, f"{m_[i0 + i] * 100:.1f}%", ha="center", fontsize=9)
    a_.set_ylim(0, 1.1); a_.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_xticks(xs, lab, fontsize=8.5)
ax.set_ylabel("Accuracy (DINOv2, 224 px, seed 42)", fontsize=9)
ax2.set_xticks([0], ["RF-Soiling\n(external,\nsoiling only)"], fontsize=8.5); ax2.set_xlim(-0.6, 0.6)
# ayrı ölçüt (SO duyarlılığı): kendi ekseni sağda, sol panelden ayrık
ax2.yaxis.tick_right(); ax2.yaxis.set_label_position("right")
ax2.spines["right"].set_visible(True); ax2.spines["left"].set_visible(False)
ax2.set_ylabel("SO recall", fontsize=9)
kaydet("fig04_evaluation_settings.png")
SAY["fig4"] = [list(map(float, t)) for t in v]
SAY["fig4_ayrinti"] = dict(sira=["P1 RGB TTA (Wilson)", "P2 (sahne kümesi, 1000)", "P3 (sahne kümesi, 1000)",
                                 "RF-Fault acc (kök kümesi, 2000)", "RF-Soiling SO recall (kök kümesi, 2000)"],
                           P1_k=k1, P1_n=n1, P1_tta_degil_k=int((TR.tahmin == TR.label)[~term].sum()),
                           RF_Fault_n=int(len(dF)), RF_Fault_kok=int(len(set(kF))), RF_Soiling_n=int(len(dS)),
                           RF_Soiling_kok=int(TS.path.map(kok_ad).nunique()))

# ---------- Şekil 5: karışıklık matrisleri + aile doğruluğu ----------
fig = plt.figure(figsize=(7.2, 5.6))
for j, (P_, y_, t) in enumerate([(Po, yo, "(a) E3, original labels (n = 2,095)"), (Pt, yt, "(b) E3, expert-audited labels (n = 2,049)")]):
    a = fig.add_subplot(2, 2, j + 1)
    cm = np.zeros((4, 4), int); np.add.at(cm, (y_, P_.argmax(1)), 1)
    cn = cm / cm.sum(1, keepdims=True)
    a.imshow(cn, cmap="Blues", vmin=0, vmax=1)
    for r_ in range(4):
        for c_ in range(4):
            a.text(c_, r_, f"{cm[r_, c_]}\n{cn[r_, c_]:.2f}", ha="center", va="center", fontsize=9, color="white" if cn[r_, c_] > 0.6 else "black")
    a.set_xticks(range(4), K, fontsize=10); a.set_yticks(range(4), K, fontsize=10)
    a.set_xlabel("Predicted", fontsize=10); a.set_ylabel("Label", fontsize=10); a.set_title(t, loc="left", fontsize=9.5)
    SAY[f"cm_{j}"] = cm.tolist()
a = fig.add_subplot(2, 1, 2)
fs_ = sorted(fam, key=lambda k: fam[k]["acc"])
a.barh(fs_, [fam[k]["acc"] for k in fs_], xerr=[[fam[k]["acc"] - fam[k]["lo"] for k in fs_], [fam[k]["hi"] - fam[k]["acc"] for k in fs_]],
       color=[C4 if fam[k]["acc"] < 0.85 else C1 for k in fs_], capsize=3, error_kw=dict(lw=1.2, capthick=1.2))
for i, k in enumerate(fs_):
    if fam[k]["hi"] - fam[k]["lo"] < 0.005:  # sıfır genişlikli GA (ör. kar: tümü doğru) görünür olsun
        a.plot([fam[k]["acc"]], [i], marker="|", color="black", ms=9, mew=1.5)
    a.text(1.035, i, f"n = {fam[k]['n']}", va="center", fontsize=8)
a.set_xlim(0.5, 1.13); a.set_xticks(np.arange(0.5, 1.01, 0.1)); a.set_xlabel("Accuracy of E3 by source family (95% cluster-bootstrap CI)"); a.set_title("(c)", loc="left", fontsize=8.5)
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
fig = plt.figure(figsize=(7.2, 3.5))
gs7 = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], hspace=0.16, wspace=0.27, left=0.095, right=0.985, bottom=0.14, top=0.91)
axs = [fig.add_subplot(gs7[:, 0])]
RS = {"Original labels (n = 2,095)": "orij", "Original labels, ambiguous removed (n = 2,049)": "orijbel", "Expert-audited labels (n = 2,049)": "temiz"}
ETK7 = {"Original labels (n = 2,095)": "Original (n = 2,095)", "Original labels, ambiguous removed (n = 2,049)": "Ambiguous removed (n = 2,049)",
        "Expert-audited labels (n = 2,049)": "Expert-audited (n = 2,049)"}  # yalnız lejant metni (kısa); anahtarlar aynı
for (ad, (cov, acc)), c in zip(EGRI.items(), [C1, C2, C3]):
    axs[0].plot(cov[20:], acc[20:], color=c, label=ETK7[ad], lw=1.2)
    r_ = pd.read_csv(f"asama2_sonuclar/secici_ek/{RS[ad]}_r0.05.csv")
    axs[0].plot(r_.kapsam.mean(), r_.acc.mean(), "o", color=c, ms=5, mec="black", mew=0.5)
axs[0].axhline(0.95, ls="--", color="grey", lw=0.8)
axs[0].set_ylim(0.88, 1.002); axs[0].set_xlabel("Coverage"); axs[0].set_ylabel("Accuracy on accepted images")
axs[0].legend(frameon=False, fontsize=8.5, loc="lower left"); axs[0].set_title("(a) Accuracy–coverage (dots: LTT, α = 5%)", loc="left", fontsize=8.5)
# (b) her etiket koşulu için 1.000 bölmenin sertifikalı kapsam histogramı; ilk kutu [0, 0,025) sıfır kapsamı gösterir
kb = np.linspace(0, 1, 41)
SAY["fig7b"] = {}
for j, ((ad, kis), c, kisa) in enumerate(zip(RS.items(), [C1, C2, C3], ["Original", "Ambiguous removed", "Audited"])):
    a = fig.add_subplot(gs7[j, 1], sharex=axs[1] if j else None, sharey=axs[1] if j else None)
    axs.append(a)
    kp = pd.read_csv(f"asama2_sonuclar/secici_ek/{kis}_r0.05.csv").kapsam.to_numpy()
    sifir = int((kp == 0).sum())
    assert not ((kp > 0) & (kp < kb[1])).any()  # ilk kutuda yalnız sıfırlar
    a.hist(kp, bins=kb, color=c)
    a.text(0.06, 0.93, f"{kisa}: zero in {sifir:,} of {len(kp):,}", transform=a.transAxes, ha="left", va="top", fontsize=8.5)
    if j < 2:
        a.tick_params(labelbottom=False)
    SAY["fig7b"][kis] = dict(n=int(len(kp)), sifir=sifir, medyan=float(np.median(kp)), ortalama=float(kp.mean()))
axs[1].set_xlim(-0.02, 1.02); axs[1].set_xticks(np.arange(0, 1.01, 0.2)); axs[1].set_ylim(0, axs[1].get_ylim()[1] * 1.3)
axs[2].set_ylabel("Splits"); axs[3].set_xlabel("Certified coverage")
axs[1].set_title("(b) LTT coverage over 1,000 splits", loc="left", fontsize=8.5)
kaydet("fig07_selective_prediction.png")  # yerleşim gs7 kenar boşluklarıyla (tight_layout bu ızgarayla uyumsuz)

# ---------- Şekil 8: güvenilirlik diyagramı ----------
def dis(veri):
    T = pd.read_csv(f"dis_dogrulama/{veri}/ortusme.csv", encoding="utf-8-sig")
    T = T[~T.ortusen].reset_index(drop=True)
    y = T.label.map(lambda s: SIN.index(s) if s in SIN else -1).to_numpy()
    P = np.mean([np.load(f"dis_dogrulama/{veri}/tahminler_{a}.npy") for a in ("ConvNeXt", "Swin", "DINOv2-392")], 0)
    return P[y >= 0], y[y >= 0]


Pf, yf = dis("roboflow")
Ps, ys = dis("melbourne")
fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.75), sharex="col", gridspec_kw=dict(height_ratios=[2.6, 1.15]))
SAY["s1_kutu_sayilari"] = {}
for j, (P_, y_, t) in enumerate([(Po, yo, "SolarFCD OOF"), (Pf, yf, "RF-Fault"), (Ps, ys, "RF-Soiling")]):
    a, h_ = axs[0, j], axs[1, j]
    e, rows = ece(P_, y_)
    a.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    cc, aa, nn = zip(*rows)
    a.plot(cc, aa, "o-", color=C1, ms=3)
    a.set_title(f"{t}\nECE = {e:.3f}, accuracy = {(P_.argmax(1) == y_).mean():.3f}", fontsize=8)
    if j:
        a.sharey(axs[0, 0]); a.tick_params(labelleft=False)
    a.tick_params(labelbottom=False)
    # alt: her güven kutusundaki görüntü sayısı (ece() ile aynı 10 eşit genişlikli kutu)
    sayi = np.bincount(np.minimum((P_.max(1) * 10).astype(int), 9), minlength=10)
    assert sayi.sum() == len(y_) and list(sayi[sayi > 0]) == list(nn)
    h_.bar(np.arange(10) / 10 + 0.05, sayi, 0.09, color="#c8c8c8")
    for i in np.flatnonzero(sayi):  # sayı: uzun çubukta çubuğun içinde (komşu etiketler çakışmasın), kısa çubukta üstünde
        ic_ = 0.42 * sayi.max() <= sayi[i] < 1000  # 4 haneli etiket çubuğa sığmaz: üstte
        h_.text(i / 10 + 0.05, sayi[i] - sayi.max() * 0.05 if ic_ else sayi[i] + sayi.max() * 0.04, f"{sayi[i]:,}",
                ha="center", va="top" if ic_ else "bottom", fontsize=6.5)
    h_.set_ylim(0, sayi.max() * 1.2); h_.set_xlim(-0.03, 1.03)
    h_.set_xlabel("Confidence")
    SAY["s1_kutu_sayilari"][t] = sayi.tolist()
    SAY.setdefault("ece", {})[t] = float(e)
axs[0, 0].set_ylabel("Observed accuracy"); axs[1, 0].set_ylabel("Images")
plt.tight_layout(h_pad=0.4); kaydet("fig08_reliability.png")

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
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4))
# %95 GA: iç alan sahne kümesi bootstrap (2.000, rng 0; hakem_revizyon/analiz/i_p2p3_ci.py ile aynı ağırlıklar),
# RF-Fault kök-ad kümesi bootstrap (2.000; Şekil 4'teki W_rf / W_rf2)
r8 = np.random.default_rng(0)
_, gi8 = np.unique(R.loc[Do.index, "sahne_grup"].to_numpy(), return_inverse=True)
W_ic = np.stack([np.bincount(r8.choice(gi8.max() + 1, gi8.max() + 1), minlength=gi8.max() + 1)[gi8] for _ in range(2000)]).astype(float)
assert (yf == yF).all()
dI, dE = (po == yo).astype(float), (Pf.argmax(1) == yf).astype(float)
rin = [(dI[yo == i].mean(), *ga(W_ic, dI, yo == i)) for i in (1, 2, 3)]
rex = [(dE[yf == i].mean(), *ga(W_rf, dE, yf == i)) for i in (1, 2, 3)]


def hata(L_):  # [(değer, alt, üst)] -> değerler, [alt hata, üst hata]
    a_ = np.array(L_)
    return a_[:, 0], [a_[:, 0] - a_[:, 1], a_[:, 2] - a_[:, 0]]


x = np.arange(3)
EK = dict(capsize=2.5, error_kw=dict(lw=0.9, capthick=0.9))
axs[0].bar(x - 0.2, hata(rin)[0], 0.4, yerr=hata(rin)[1], color=C1, label="In-domain (SolarFCD OOF)", **EK)
axs[0].bar(x + 0.2, hata(rex)[0], 0.4, yerr=hata(rex)[1], color=C2, label="RF-Fault", **EK)
axs[0].set_xticks(x, ["H", "SF", "SO"]); axs[0].set_ylabel("Recall (E3)"); axs[0].set_ylim(0, 1.05); axs[0].legend(frameon=False, fontsize=7)
axs[0].set_title("(a) Per-class recall", loc="left", fontsize=8.5)
SAY["recall_ic_dis"] = dict(ic=[float(t[0]) for t in rin], dis=[float(t[0]) for t in rex])
SAY["fig8_ga"] = dict(recall_ic={k: list(map(float, t)) for k, t in zip(["H", "SF", "SO"], rin)},
                      recall_rf_fault={k: list(map(float, t)) for k, t in zip(["H", "SF", "SO"], rex)})
M8 = {"ConvNeXt": "convnext_tiny_224", "Swin": "swin_tiny_patch4_window7_224_224", "DINOv2-224": "vit_base_patch14_dinov2_224"}


def ic_acc(kok, kisa, son):  # iç CV (OOF, tohum 42, TTA'sız test_predictions.csv)
    D_ = pd.concat([pd.read_csv(f"{kok}/grouped{k}_{kisa}_s42_{son}/test_predictions.csv") for k in range(5)]).set_index("path").sort_index()
    assert (D_.index == Do.index).all()
    d_ = (D_.tahmin == D_.label).to_numpy().astype(float)
    return (d_.mean(), *ga(W_ic, d_))


def dx_acc(ad_, tur, W_):  # RF-Fault (5 kat modeli, TTA; 25_dis_dogrulama.py)
    d_ = (dis_p("roboflow", f"{ad_}_{tur}_s42").argmax(1) == yF).astype(float)
    return (d_.mean(), *ga(W_, d_))


ic = {k: (ic_acc("runs_rastgele", u, "kat_rastgele_asama2_rastgele_none"), ic_acc("runs_asama2", u, "kat_v1_asama2_none")) for k, u in M8.items()}
dx = {k: (dx_acc(k, "sızıntılı", W_rf2), dx_acc(k, "sızıntısız", W_rf)) for k in M8}
ESKI8 = {"ConvNeXt": (0.9103, 0.9002, 0.6372, 0.6074), "Swin": (0.9084, 0.8998, 0.5821, 0.5912), "DINOv2-224": (0.9103, 0.8974, 0.6778, 0.7067)}
for k in M8:  # önceki sabit değerlerle (25_dis_dogrulama.py özeti) aynı mı
    assert np.allclose([ic[k][0][0], ic[k][1][0], dx[k][0][0], dx[k][1][0]], ESKI8[k], atol=5e-5), k
SAY["fig8_ga"].update(ic_P2={k: list(map(float, ic[k][0])) for k in M8}, ic_P3={k: list(map(float, ic[k][1])) for k in M8},
                      rf_fault_P2={k: list(map(float, dx[k][0])) for k in M8}, rf_fault_P3={k: list(map(float, dx[k][1])) for k in M8})
x = np.arange(3); w = 0.2
for j, (lab_, vals, c) in enumerate([("Internal CV, P2 (leaky)", [ic[k][0] for k in ic], "#bdbdbd"), ("Internal CV, P3 (grouped)", [ic[k][1] for k in ic], C1),
                                      ("RF-Fault, P2 models", [dx[k][0] for k in dx], "#f0b27a"), ("RF-Fault, P3 models", [dx[k][1] for k in dx], C2)]):
    axs[1].bar(x + (j - 1.5) * w, hata(vals)[0], w, yerr=hata(vals)[1], label=lab_, color=c, **dict(EK, capsize=2))
axs[1].set_xticks(x, list(ic)); axs[1].set_ylim(0, 1.05); axs[1].set_ylabel("Accuracy (seed 42)")
for a_ in axs:
    a_.set_ylim(0, 1.4); a_.set_yticks(np.arange(0, 1.01, 0.2))
axs[1].legend(frameon=False, fontsize=8.5, loc="upper center", ncol=1, handlelength=1.2, columnspacing=0.8)
axs[1].set_title("(b) Internal versus external accuracy", loc="left", fontsize=8.5)
axs[0].legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2, handlelength=1.2, columnspacing=1.0)
axs[0].set_title("(a) Per-class recall", loc="left", fontsize=8.5)
plt.tight_layout(); kaydet("fig10_external_validation.png")

# ---------- Şekil S3: eğitim eğrileri (P3, 224 px; omurga başına 5 kat x 3 tohum = 15 koşunun ortalaması) ----------
OMG = {"ResNetV2-101": ("resnetv2_101_224", "#5d5d5d"), "EfficientNetV2-S": ("tf_efficientnetv2_s_224", "#8e6c8a"),
       "ConvNeXt-T": ("convnext_tiny_224", C1), "ConvNeXt V2-T": ("convnextv2_tiny_224", C3),
       "Swin-T": ("swin_tiny_patch4_window7_224_224", C2), "DINOv2 ViT-B/14": ("vit_base_patch14_dinov2_224", C4)}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3))
ins = [axs[0].inset_axes([0.42, 0.07, 0.55, 0.47]), axs[1].inset_axes([0.42, 0.4, 0.55, 0.47])]  # son 10 dönem yakınlaştırma
SAY["s3_egitim"] = {}
ep = np.arange(1, 31)
for ad, (kisa, c) in OMG.items():
    L_ = [pd.read_csv(f"runs_asama2/grouped{k}_{kisa}_s{s}_kat_v1_asama2_none/log.csv") for k in range(5) for s in (42, 43, 44)]
    assert len(L_) == 15 and all(len(l_) == 30 and (l_.epoch.to_numpy() == ep).all() for l_ in L_)
    acc_, kay_ = np.mean([l_.train_acc.to_numpy() for l_ in L_], 0), np.mean([l_.train_loss.to_numpy() for l_ in L_], 0)
    axs[0].plot(ep, acc_, color=c, lw=1.2, label=ad + (" (not shown in insets)" if ad == "ResNetV2-101" else ""))
    axs[1].plot(ep, kay_, color=c, lw=1.2)
    if ad != "ResNetV2-101":  # yakınlaştırmada ölçek dışı (30. dönem: doğruluk %84,1, kayıp 0,638)
        ins[0].plot(ep[20:], acc_[20:], color=c, lw=1.1, marker="o", ms=1.8)
        ins[1].plot(ep[20:], kay_[20:], color=c, lw=1.1, marker="o", ms=1.8)
    SAY["s3_egitim"][ad] = dict(kosu=len(L_), acc_ep25=float(acc_[24]), acc_ep30=float(acc_[29]),
                                kosu_acc_ep30_min=float(min(l_.train_acc.iloc[-1] for l_ in L_)),
                                kosu_acc_ep30_max=float(max(l_.train_acc.iloc[-1] for l_ in L_)),
                                kayip_ep25=float(kay_[24]), kayip_ep30=float(kay_[29]),
                                kayip_degisim_ep25_30=float(kay_[24] - kay_[29]),
                                kayip_aralik_ep26_30=float(kay_[25:].max() - kay_[25:].min()))
from matplotlib.ticker import FuncFormatter, NullFormatter
axs[0].set_ylabel("Training accuracy"); axs[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
axs[1].set_ylabel("Training loss (log scale)"); axs[1].set_yscale("log"); axs[1].set_yticks([0.3, 0.5, 1, 2, 5])
axs[1].yaxis.set_major_formatter(FuncFormatter(lambda v_, _: f"{v_:g}")); axs[1].yaxis.set_minor_formatter(NullFormatter())
for a_, t in zip(axs, ["(a) Training accuracy", "(b) Training loss"]):
    a_.set_xlabel("Epoch"); a_.set_xlim(0.5, 30.5); a_.set_xticks([1, 5, 10, 15, 20, 25, 30]); a_.set_title(t, loc="left", fontsize=8.5)
ins[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
for i_ in ins:
    i_.set_xlim(20.5, 30.5); i_.set_xticks([21, 25, 30]); i_.tick_params(labelsize=7, length=2.5, pad=1.5)
    i_.set_title("Epochs 21–30", fontsize=7, pad=2)
    i_.spines["top"].set_visible(True); i_.spines["right"].set_visible(True)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=7.5, handlelength=1.5,
           columnspacing=1.5)
plt.tight_layout(rect=(0, 0.11, 1, 1)); kaydet("figS3_training_curves.png")

# ---------- Şekil 9 (fig11): oturum/alt koleksiyon gruplu az örnekli uyarlama (28b, gönderim öncesi iç değerlendirme) ----------
Rf = pd.read_csv("asama2_sonuclar/az_ornek_gruplu.csv")
fig, ax = plt.subplots(figsize=(4.6, 3.3))
rj = np.random.default_rng(0)
for yon, ad, c, mk in [("ince_ayar_solarfcd", "Fine-tuned from SolarFCD model", C1, "o"),
                       ("ince_ayar_imagenet", "Fine-tuned from ImageNet only", C2, "s")]:
    d = Rf[Rf.yontem == yon]
    ax.scatter(d.n_egitim + rj.uniform(-4, 4, len(d)), d.acc, s=9, color=c, alpha=0.45, marker=mk, lw=0)
    g = d.groupby("butce").agg(n=("n_egitim", "mean"), m=("acc", "mean"))
    ax.plot(g.n, g.m, marker=mk, color=c, label=ad, ms=4)
for yon, ad, c, ls in [("dogrusal_solarfcd", "Logistic regression on SolarFCD features", C3, "--"),
                       ("1nn_solarfcd", "1-NN on SolarFCD features", "#7f7f7f", ":")]:
    g = Rf[Rf.yontem == yon].groupby("butce").agg(n=("n_egitim", "mean"), m=("acc", "mean"))
    ax.plot(g.n, g.m, ls=ls, color=c, label=ad, lw=1.2)
ax.axhline(Rf.groupby("tekrar").test_cogunluk.first().mean(), color="#9e9e9e", lw=0.8, ls="-.", label="Majority class (mean)")
ax.set_xlabel("Labelled target images"); ax.set_ylabel("Accuracy on held-out RF-Fault groups")
ax.set_ylim(0.15, 1.22); ax.set_yticks(np.arange(0.2, 1.01, 0.1)); ax.legend(frameon=False, loc="upper left", ncol=1, fontsize=8)
kaydet("fig11_few_shot.png")
shutil.copy("asama2_sonuclar/gradcam_v2.png", OUT / "fig12_gradcam.png")  # 29b_gradcam_v2.py: 300 dpi, filigran adı bulanık

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
ESKI = json.load(open(OUT / "ek_sayilar.json", encoding="utf-8")) if (OUT / "ek_sayilar.json").exists() else {}
if "maliyet" in ESKI:  # gecikme yeniden ölçülmez; makaledeki değerler korunur
    MD = {}
    mal = ESKI["maliyet"]
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
print(json.dumps({k: SAY[k] for k in ("fig4_ayrinti", "fig7b", "s1_kutu_sayilari", "fig8_ga", "s3_egitim")}, indent=1, default=str))
