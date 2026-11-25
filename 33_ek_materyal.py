"""Adım 33 - Makale ek materyali mevcut sonuç dosyalarından üretilir -> makale/supplementary.md
Sürüm 24 Kasım (gönderim öncesi iç değerlendirme, plan O3; önceki sürüm 33_ek_materyal_yedek_20261124.py):
  başlık (UD1), kısaltmalar, Ek Yöntemler, Tablo S1-S14, Şekil S1-S3 ve ek kaynakça.
  S1-S7 önceki sürümle aynı; yalnız S1'e E3 gecikme notu, S3'e α = %2 ihlal notu eklendi. S2 (az örnek) yazarın süren
  deneyi bitene kadar başlığıyla birlikte aynen kalır. Şekil S1 açıklaması ECE GA'ları ve kutu sayısı histogramlarıyla
  güncellendi; Şekil S3 (eğitim eğrileri) başka adımda çiziliyor (makale/figS3_training_curves.png), açıklaması plan O1'den.
  Sayıların kaynakları (hepsi salt okunur):
    hakem_revizyon/analiz/*_out.txt, *_sonuc.txt  doğrulayıcı çözümlemeleri (r1_a: iç alan, r1_a4: P1 ayrıştırma,
        r1_b: uzman/dış eşleme, r1_c, r1_d: TTA, r1_f: kutu birleşimleri, r1_h: iş yükü, a_uzman, h_kappa_ci, b_secici,
        c_dis, e_protokol, g_egitim_egrisi, k_recal)
    asama2/ozet.json, kat_sinif_v1.csv, buyuk_gruplar.csv, manifest_asama2.csv  son (r4.1) manifest ve parametreleri
    asama2_sonuclar/ (secici_ek, dis_secici.txt, cozunurluk.txt, az_ornek.csv -> S2, az_ornek_gruplu.csv -> S14 madde 6),
    insan_kontrol/sonuc/,
    dis_dogrulama/ozet.txt, makale/ek_sayilar.json
    08_asama2_manifest.py  eşleştirici sabitleri (kaynak koddan okunur)
    NOTLAR.md §4 ve §9  denetim turları (elle, satırlar aşağıda yorumda)
  Ek kaynakça: geifman2017, vovk2022 (makale/kaynakca.py HAND); hayes2007 (Crossref) ve lipton2018 (PMLR + DataCite)
  makale/ek_kaynak_meta.json'dan (makale/ek_kaynak_meta.py ile indirildi). Metindeki atıflar [S1]... geçiş sırasıyla.
Doğruluk ve kapsam yüzde; eksi işareti U+2212; model adları ana metinle aynı; 'nan' yerine '–'."""
import ast
import json
import math
import re
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
from sklearn.metrics import cohen_kappa_score

M = "−"


def md(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])


def yuzde(m, sd=None, d=1):
    if pd.isna(m):
        return "–"
    return f"{m * 100:.{d}f}" + ("" if sd is None or pd.isna(sd) else f" ± {sd * 100:.{d}f}") + "%"


def eksi(s):
    return str(s).replace("-", M)


# ------------------------------------------------------------------ yardımcılar (yeni bölümler)
AN = "hakem_revizyon/analiz/"


def oku(p):
    return open(p, encoding="utf-8-sig").read()


def bul(rx, t, fl=re.M):
    m = re.search(rx, t, fl)
    assert m, f"bulunamadı: {rx}"
    return m


def f(x):
    return float(x)


def pr1(x):                     # oran -> '91.5' (bir ondalık, yarım yukarı; çift yuvarlamaya karşı sayımla: yk)
    return str((Decimal(repr(f(x))) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def y1(x):                      # oran -> '91.5%'
    return pr1(x) + "%"


def yk(x, n):                   # 4 ondalıklı yazılmış oranı sayıya çevirip kesin yüzde (ör. 0.9725 = 354/364 -> 97.3%)
    k = round(f(x) * int(n))
    assert abs(k / int(n) - f(x)) < 6e-5, (x, n)
    return str((Decimal(k) * 100 / Decimal(int(n))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)) + "%"


def d3(x):                      # '0.123', eksi işaretli
    return eksi(f"{f(x):.3f}")


def d2(x):                      # yarım yukarı yuvarlama (0.425 -> 0.43)
    return eksi(str(Decimal(repr(f(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)))


def b3(v, lo, hi):              # '0.869 [0.827, 0.902]'
    return f"{d3(v)} [{d3(lo)}, {d3(hi)}]"


def pb(v, lo, hi, n=None):      # '91.5% [90.0, 92.8]' (oranlardan; n verilirse nokta tahmini sayımdan)
    return f"{y1(v) if n is None else yk(v, n)} [{pr1(lo)}, {pr1(hi)}]"


def n_(x):
    return f"{int(x):,}"


def wilson(k, n, z=1.959963984540054):
    p = k / n
    dd = 1 + z * z / n
    c = (p + z * z / (2 * n)) / dd
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / dd
    return c - h, c + h


def tablo(sutun, satir):
    for r in satir:
        assert len(r) == len(sutun), (sutun, r)
    return md(pd.DataFrame(satir, columns=sutun))


SIRA = []


def atif(k):
    if k not in SIRA:
        SIRA.append(k)
    return f"[S{SIRA.index(k) + 1}]"


# ------------------------------------------------------------------ kaynak dosyalar
A = oku(AN + "r1_a_sonuc.txt")
A4 = oku(AN + "r1_a4_sonuc.txt")
B = oku(AN + "r1_b_sonuc.txt")
C = oku(AN + "c_dis_out.txt")
C1 = oku(AN + "r1_c_sonuc.txt")
D = oku(AN + "r1_d_sonuc.txt")
E = oku(AN + "e_protokol_out.txt")
F = oku(AN + "r1_f_sonuc.txt")
G = oku(AN + "g_egitim_egrisi_out.txt")
H = oku(AN + "r1_h_sonuc.txt")
K = oku(AN + "h_kappa_ci_out.txt")
KR = oku(AN + "k_recal_out.txt")
U = oku(AN + "a_uzman_out.txt")
BS = oku(AN + "b_secici_out.txt")
IK = oku("insan_kontrol/sonuc/ozet.txt")
DS = oku("asama2_sonuclar/dis_secici.txt")
KG = oku("dis_dogrulama/ozet.txt")
M8 = oku("08_asama2_manifest.py")
OZ = json.load(open("asama2/ozet.json", encoding="utf-8"))
PR, KS = OZ["parametreler"], OZ["kenar_sayisi"]
EKS = json.load(open("makale/ek_sayilar.json", encoding="utf-8"))
MAN = pd.read_csv("asama2/manifest_asama2.csv")

s = ["---", 'title: "Supplementary material for: Leakage, label ambiguity and domain shift in image-based photovoltaic '
     'fault classification and implications for operation and maintenance"', "---", "", "Enes Halit Aydın", ""]

# ------------------------------------------------------------------ kısaltmalar (ana metin + ek materyal)
KIS = [("AC1", "Gwet's first-order agreement coefficient"), ("AdamW", "adaptive moment estimation (Adam) with decoupled weight decay"),
       ("AI", "artificial intelligence"), ("CI", "confidence interval"), ("CNN", "convolutional neural network"),
       ("CRediT", "Contributor Roles Taxonomy"), ("CV", "cross-validation"),
       ("DINOv2", "self-distillation with no labels, version 2 (self-supervised vision transformer)"),
       ("E3", "ensemble of ConvNeXt-T (224 px), Swin-T (224 px) and DINOv2 (392 px), each over three seeds"),
       ("ECE", "expected calibration error"), ("EF", "Electrical Fault class"), ("EL", "electroluminescence"),
       ("EXIF", "exchangeable image file format"),
       ("F1", "harmonic mean of precision and recall (macro-F1, its unweighted mean over the classes)"),
       ("FCMAE", "fully convolutional masked autoencoder"), ("GB", "gigabyte"), ("GPU", "graphics processing unit"),
       ("Grad-CAM", "gradient-weighted class activation mapping"), ("H", "Healthy class"),
       ("IEC", "International Electrotechnical Commission"), ("IoU", "intersection over union (mIoU, mean IoU)"),
       ("IQR", "interquartile range"), ("IR", "infrared"), ("I–V", "current–voltage"),
       ("K", "Kaggle clean/dirty archive, with K-clean (Imgclean) and K-dirty (Imgdirty) images"),
       ("LAC", "least-ambiguous set-valued classifier"), ("LTT", "Learn-then-Test"), ("MAE", "mean absolute error"),
       ("MAGSAC++", "marginalizing sample consensus, a robust estimator of geometric models"),
       ("MD5", "message-digest algorithm 5"), ("NCC", "normalised cross-correlation"), ("O&M", "operation and maintenance"),
       ("OOF", "out-of-fold"),
       ("P1, P2, P3", "evaluation protocols with the official split, random five-fold CV and scene-grouped five-fold CV"),
       ("PD", "PV Panel Defect collection, with PD-bird, PD-clean, PD-dust, PD-electrical, PD-physical and PD-snow images"),
       ("pHash", "perceptual hash"), ("PID", "potential-induced degradation"), ("PV", "photovoltaic"), ("px", "pixel"),
       ("RF-Fault, RF-Soiling", "external datasets from Roboflow Universe"), ("RGB", "red–green–blue"),
       ("RootSIFT", "scale-invariant feature transform descriptors with square-root (Hellinger) normalisation"),
       ("SD", "standard deviation"), ("SF", "Structural Fault class"), ("SGR", "selection with guaranteed risk"),
       ("SO", "Surface Obstruction class"), ("timm", "PyTorch Image Models library"), ("TTA", "test-time augmentation"),
       ("ViT-B/14", "Vision Transformer, base size, with patches of 14 × 14 px")]
KIS.sort(key=lambda x: re.sub(r"[^a-z0-9]", "", x[0].lower()))
s += ["# Abbreviations {.unnumbered}", "", "Abbreviations used in the main text and in this supplement.", "",
      tablo(["Abbreviation", "Meaning"], KIS), ""]

# ------------------------------------------------------------------ Ek Yöntemler
# TTA etkisi: r1_d (koşu bazında P2/P3 havuzlanmış), e_protokol (P3, kat ve tohum ağırlıklı ortalama)
tta = re.findall(r"^runs_\w+\s+\S+\s+s\d+: no TTA acc ([\d.]+) F1 ([\d.]+) \| TTA acc ([\d.]+) F1 ([\d.]+)", D, re.M)
assert len(tta) == 23, len(tta)
tta_acc = max(abs(f(b) - f(a)) for a, _, b, _ in tta) * 100
tta_f1 = max(abs(f(b) - f(a)) for _, a, _, b in tta)
tt = re.findall(r"^\w+\s+[\d.]+\s+[\d.]+\s+15\.0\s+(-?[\d.]+)$", E.split("TTA effect")[1], re.M)
assert len(tt) == 8
tt = [f(x) for x in tt]
s += ["# Supplementary Methods", "",
      "## Selective prediction, conformal sets and calibration error", "",
      r"Following Learn-then-Test, eleven candidate coverage levels $c_k \in \{0.50, 0.55, \ldots, 1.00\}$ define thresholds $\lambda_k$. "
      r"For each level, $e_k$ denotes the number of errors among the $n_k$ accepted calibration images, and the null hypothesis that the "
      r"risk exceeds α is rejected when the one-sided Clopper–Pearson upper bound at the Bonferroni-corrected level does not exceed α,", "",
      r"$$U_k=\mathrm{Beta}^{-1}\!\left(1-\delta/11;\; e_k+1,\; n_k-e_k\right)\le\alpha. \qquad (S1)$$", "",
      "The widest coverage whose hypothesis is rejected is selected. The least-ambiguous set-valued classifier (LAC) returns the conformal sets", "",
      r"$$C(x)=\{\,y:\;1-\hat p_y(x)\le\hat q\,\}, \qquad (S2)$$", "",
      r"where $\hat q$ is the $\lceil (n+1)(1-\varepsilon)\rceil/n$ empirical quantile of the calibration scores $1-\hat p_{y_i}(x_i)$ and ε = 0.05. "
      r"Calibration was summarised by the expected calibration error over ten equal-width confidence bins $B_b$,", "",
      r"$$\mathrm{ECE}=\sum_{b=1}^{10}\frac{|B_b|}{n}\,\bigl|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)\bigr|. \qquad (S3)$$", "",
      "## Test-time augmentation", "",
      "Test-time augmentation (TTA) averaged the softmax probabilities for the original image and its horizontally and vertically "
      "flipped versions. It was applied to the official-split evaluation (P1), to all predictions on external data and to the "
      "few-shot experiments. The pooled out-of-fold (OOF) metrics under P2 and P3, and all analyses built on them, used single-view "
      f"predictions. Pooling the per-fold TTA results instead would have changed the pooled accuracy of individual P2 and P3 runs "
      f"by at most {tta_acc:.1f} percentage points and their macro-F1 by at most {tta_f1:.3f}, and the P3 accuracy averaged over "
      f"the three seeds by {eksi(f'{min(tt):+.1f}')} to {max(tt):+.1f} points.", "",
      "## Ensembles and training protocols", "",
      "The candidate ensemble that selected the images for the expert audit averaged the class probabilities of ConvNeXt-T, Swin-T "
      "and DINOv2, each at 224 px and over seeds 42, 43 and 44. The ensemble E3 used in all other analyses averaged ConvNeXt-T "
      "(224 px), Swin-T (224 px) and DINOv2 (392 px) over the same three seeds. In-domain, each OOF prediction of E3 therefore "
      "averaged nine models, namely the fold model of each backbone and seed that had not seen the image. External predictions "
      "averaged the five fold models of each backbone over the three seeds with TTA, and the external E3 averaged these three "
      "backbone means, so it combined 45 models. Under P1, each model was trained for at most 30 epochs with early stopping after "
      "eight epochs without improvement, and the checkpoint with the highest macro-F1 on the official validation split was kept. "
      "P2 and P3 trained every model for a fixed 30 epochs without a validation split and used the last epoch.", ""]

# LTT: bölme başına dosyalar (asama2_sonuclar/secici_ek) ve b_secici, dis_secici, r1_b (etiket bütçesi)
KOS = {}
for et in ("orij", "orijbel", "temiz"):
    for r in ("0.02", "0.05", "0.10"):
        x = pd.read_csv(f"asama2_sonuclar/secici_ek/{et}_r{r}.csv")
        assert not (x.ihlal & (x.kapsam <= 0)).any()
        KOS[et, r] = (int((x.kapsam > 0).sum()), int(x.ihlal.sum()), len(x))
bo = bul(r"^Original labels: coverage ([\d.]+) ± ([\d.]+); zero-coverage splits (\d+); median ([\d.]+), IQR ([\d.]+)-([\d.]+); "
         r"violation ([\d.]+); violation \| any accepted ([\d.]+) \(n=(\d+)\)", BS)
assert KOS["orij", "0.05"][0] == int(bo[9]) and 1000 - KOS["orij", "0.05"][0] == int(bo[3])
cb_n, cb_v, _ = KOS["orij", "0.05"]
ob_n, ob_v, _ = KOS["orijbel", "0.02"]
te_n, te_v, _ = KOS["temiz", "0.02"]
assert (ob_n, ob_v) == (17, 17) and (te_n, te_v) == (168, 15) and KOS["orij", "0.02"][0] == 0
ic = bul(r"İç kalibrasyon \(2\.095\): eşik ([\d.]+), LAC q [\d.]+; iç uygulama: \{'kapsam': np\.float64\(([\d.]+)\), "
         r"'acc': np\.float64\(([\d.]+)\)", DS)
bt = {d_: bul(rf"^delta={d_}: zero errors needs n_k >= (\d+); with the 50% minimum coverage level, calibration set >= (\d+); "
              rf"one error needs n_k >= (\d+)", B) for d_ in ("0.1", "0.05", "0.01")}
s += ["## Risk-controlled selective prediction", "",
      r"For each candidate coverage level $c_k$, the threshold $\lambda_k$ was set to the maximum E3 probability of the "
      r"$\lfloor c_k n\rfloor$-th most confident of the $n$ calibration images. The thresholds therefore depend on the calibration "
      "data, and the finite-sample guarantee rests on conditioning on these order statistics, as in selection with guaranteed risk "
      f"(SGR) {atif('geifman2017')}. Because the calibration and test scores are OOF probabilities of fold models that were each "
      "trained partly on images of the other half of a split, the guarantee holds only approximately. The Bonferroni grid was "
      "adopted after two alternatives had been tried on the same data and splits, so the reported violation rates are not an "
      "independent check of the final procedure.", "",
      f"A split in which no image was accepted counts as a non-violation. With the original labels at α = 5%, {int(bo[3])} of the "
      f"1,000 splits accepted no image, {cb_v} of the {cb_n} splits that accepted any image violated the bound ({cb_v / cb_n:.3f}), "
      f"and the certified coverage had a median of {y1(bo[4])} with an interquartile range (IQR) of {pr1(bo[5])}–"
      f"{y1(bo[6])}. At α = 2%, all {ob_n} splits that accepted any image violated the bound with the ambiguous images removed, "
      f"which gives the unconditional rate of {ob_v / 1000:.3f} in Table S3, compared with {te_v} of {te_n} splits with the "
      "audited labels. Class-conditional (Mondrian) LAC sets used a separate conformal quantile for each class "
      f"{atif('vovk2022')} (Table S13).", "",
      "The threshold transferred to external data was calibrated once with LTT on all 2,095 OOF predictions, where it was "
      f"λ = {ic[1]} and accepted {y1(ic[2])} of the images at an accuracy of {y1(ic[3])}. Re-certification on new data is costly "
      "in labels. With δ = 0.1 and eleven candidate levels, the Clopper–Pearson bound reaches α = 5% only if at least "
      f"{bt['0.1'][1]} accepted calibration images contain no error, or {bt['0.1'][3]} with one error, and the corresponding "
      f"numbers without errors are {bt['0.05'][1]} at δ = 0.05 and {bt['0.01'][1]} at δ = 0.01. An error-free model therefore needs at "
      f"least {bt['0.1'][1]} labelled target images at δ = 0.1, and because the lowest candidate coverage is 50%, a model whose less "
      f"confident half is wrong needs at least {bt['0.1'][2]} to certify any coverage.", ""]

# Dış veri: c_dis (tarama, kosinüs bantları, etiket kayması), r1_b (kök, aynı kök, kök bootstrap), k_recal
rf = bul(r"^RF-Fault: n (\d+), removed (\d+) \(MD5 (\d+), pHash<=12 (\d+), cos>=0\.85 (\d+), cosine-only (\d+)", C)
rs = bul(r"^RF-Soiling: n (\d+), removed (\d+) \(MD5 (\d+), pHash<=12 (\d+), cos>=0\.85 (\d+), cosine-only (\d+)", C)
rr_ = dict(re.findall(r"^\s+(Fault_unknown_type|Healthy|Structural_Fault|Surface_Obstruction)\s+n\s+\d+ removed ([\d.]+)",
                      C.split("RF-Soiling:")[0], re.M))
band = re.findall(r"^\[([\d.]+), ([\d.]+)\)\s+([\d.]+)\s+(\d+)$", C, re.M)
assert len(band) == 5 and sum(int(b[3]) for b in band) == 1108
ayni = re.findall(r"^\d+\s+(Imgdirty_\d+_\d+)\s+(\d+)\s+([\d.]+)\s+\S+\s+(\w+)$", B, re.M)
assert len(ayni) == int(bul(r"retained RF-Fault whose stem equals a SolarFCD file stem: (\d+)", B)[1]) == 3
assert all(a[3] == "Fault_unknown_type" for a in ayni)
st = bul(r"retained RF-Fault unique stems: (\d+) for (\d+) images", B)
st4 = bul(r"four-class known images: (\d+) \| unique stems among them: (\d+)", B)
sts = bul(r"retained RF-Soiling unique stems: (\d+)", B)
stx = bul(r"RF-Soiling retained stems also among retained RF-Fault stems: (\d+)", B)
KB = {m_[0]: m_ for m_ in re.findall(
    r"^(ConvNeXt|Swin|DINOv2-392|E3)\s+4-class acc ([\d.]+)%: Wilson ([\d.]+)%-([\d.]+)% \| stem-cluster bootstrap ([\d.]+)%-([\d.]+)% "
    r"\((\d+) stems\)\n\s+fault/no-fault ([\d.]+)%: Wilson ([\d.]+)%-([\d.]+)% \| stem bootstrap ([\d.]+)%-([\d.]+)% \((\d+) stems\)",
    B, re.M)}
assert len(KB) == 4
e3k = KB["E3"]
cx = bul(r"^RF-Fault four-class n (\d+): E3 acc ([\d.]+); majority baseline ([\d.]+)", C)
ci_ = bul(r"^In-domain E3 on H/SF/SO images \(n=(\d+)\): acc ([\d.]+);.*balanced ([\d.]+); priors .*'Structural_Fault': "
          r"np\.float64\(([\d.]+)\)", C)
rw = bul(r"RF-Fault E3 recalls re-weighted by SolarFCD H/SF/SO priors: ([\d.]+)", C)
krg = bul(r"images (\d+) groups (\d+)", KR)
kr = re.findall(r"^grid ([\d.]+)-1, n_cal~(\d+): LTT coverage ([\d.]+); LAC coverage ([\d.]+), set size ([\d.]+)", KR, re.M)
assert len(kr) == 8
sf_pay = (MAN.temsilci & (MAN.label == "Structural_Fault")).sum() / MAN.temsilci.sum()
s += ["## External data", "",
      "RF-Fault images were labelled from their box classes in a fixed order of priority. An image with a physical-damage box was "
      'labelled SF. Otherwise, an image with a "defective" box was treated as a fault of unknown type that entered only the '
      "fault/no-fault analysis, an image with a bird-drop, dust, dusty or snow box was labelled SO, and an image with only "
      f"non-defective boxes was labelled H. All {n_(rs[1])} RF-Soiling images carry at least one dust or bird-dropping box and were "
      "labelled SO. The box combinations and the sensitivity of the results to this mapping are given in Table S8.", "",
      f"Overlap screening removed {n_(rf[2])} of the {n_(rf[1])} RF-Fault images ({int(rf[2]) / int(rf[1]) * 100:.1f}%), "
      f"{n_(rf[4])} of them by pHash, {n_(rf[5])} by DINOv2 cosine similarity and {n_(rf[6])} by cosine similarity alone, whereas "
      f"{'no image was' if int(rf[3]) == 0 else rf[3] + ' images were'} byte-identical to a SolarFCD file. The removal rate was "
      f"{y1(rr_['Surface_Obstruction'])} for SO, {y1(rr_['Fault_unknown_type'])} for images of unknown fault type, "
      f"{y1(rr_['Healthy'])} for H and {y1(rr_['Structural_Fault'])} for SF. In RF-Soiling, {rs[2]} of the {rs[1]} images "
      f"({int(rs[2]) / int(rs[1]) * 100:.1f}%) were removed, {rs[6]} of them by cosine similarity alone. Because the cosine "
      "criterion also removes images of similar installations, the retained images may be harder than the removed ones. Among the "
      f"retained four-class images, E3 was most accurate close to the cut-off, with an accuracy of {y1(band[0][2])} below a cosine "
      f"of {band[0][1]} ({band[0][3]} images), "
      + ", ".join(f"{y1(b[2])} at {b[0]}–{b[1]} ({b[3]})" for b in band[1:4])
      + f" and {y1(band[4][2])} at 0.80–{band[4][1]} ({band[4][3]}). Three retained RF-Fault files named Imgdirty share their "
      f"file-name stem with their nearest SolarFCD image (pHash distance {min(int(a[1]) for a in ayni)}–"
      f"{max(int(a[1]) for a in ayni)}, cosine {min(f(a[2]) for a in ayni):.3f}–{max(f(a[2]) for a in ayni):.3f}), but all "
      "three are images of unknown fault type that enter only the fault/no-fault analysis.", "",
      "Because the Roboflow exports contain several augmented copies of many source images, the source-file stem was defined as "
      f"the file name before `_jpg.rf.` or `.rf.`. The {n_(st[2])} retained RF-Fault images derive from {n_(st[1])} stems "
      f"({st4[2]} stems for the {n_(st4[1])} four-class images), and the 137 RF-Soiling images from {sts[1]} stems, {stx[1]} of "
      "which also occur among the retained RF-Fault images. A cluster bootstrap over stems with 2,000 resamples gave a 95% CI of "
      f"{f(e3k[4]):.1f}–{f(e3k[5]):.1f}% for the four-class accuracy of E3, compared with a Wilson interval over images of "
      f"{f(e3k[2]):.1f}–{f(e3k[3]):.1f}%, and {f(e3k[10]):.1f}–{f(e3k[11]):.1f}% for its fault/no-fault accuracy.", "",
      f"SF makes up {y1(cx[3])} of the four-class RF-Fault images, which is therefore also the majority-class baseline, but only "
      f"{y1(sf_pay)} of the SolarFCD evaluation set. Re-weighting the RF-Fault recalls of E3 to the SolarFCD proportions of H, SF and "
      f"SO gives {y1(rw[1])}, compared with an in-domain E3 accuracy of {y1(ci_[2])} on these three classes, so part of the "
      "external loss reflects this label shift, which can be corrected when the target class proportions are estimated "
      f"{atif('lipton2018')}.", "",
      f"Recalibration on RF-Fault with calibration and test splits made at the level of the {krg[2]} near-duplicate groups (same "
      "stem or pHash distance of at most 6) gave the same result as image-level splits. LTT certified a mean coverage of at most "
      f"{max(f(k[2]) for k in kr) * 100:.1f}%, even with the coverage grid widened to 5–100%, while the LAC sets reached "
      f"{min(f(k[3]) for k in kr) * 100:.1f}–{max(f(k[3]) for k in kr) * 100:.1f}% coverage with mean sizes of "
      f"{min(f(k[4]) for k in kr):.2f}–{max(f(k[4]) for k in kr):.2f} labels. A re-screen of the external sets with dihedral "
      "pHash, RootSIFT matching and a visual audit, as used for SolarFCD, was not performed. Its cost is estimated at about 10 min "
      "of computation and a few hours of visual checking.", ""]

# Bootstrap, etiket yumuşatma, çelişkili gruplar
e3 = bul(r"^E3: acc ([\d.]+) \[([\d.]+), ([\d.]+)\], macro-F1 ([\d.]+) \[([\d.]+), ([\d.]+)\]", E)
e3g = bul(r"^E3 without the largest group \(n=(\d+)\): acc ([\d.]+) \[([\d.]+), ([\d.]+)\], F1 ([\d.]+) \[([\d.]+), ([\d.]+)\]", E)
gv = bul(r"max confidence ([\d.]+), 99th percentile ([\d.]+)", C)
ck = MAN[MAN.celiskili & ~MAN.termal]
ckg = ck.groupby("foto_grup").celiski_etiketleri.first().value_counts()
ckn = ck.label.value_counts()
tk = MAN[MAN.celiskili & MAN.termal]
assert set(tk.celiski_etiketleri) == {"Structural_Fault+Surface_Obstruction"}
s += ["## Bootstrap intervals", "",
      "The cluster-bootstrap intervals resample scene groups, or source-file stems for external data, around fixed fitted models. "
      "They are therefore conditional on these models and do not include the variability due to retraining or to the choice of "
      f"fold partition. They are not driven by the largest scene group either, because without its {2095 - int(e3g[1])} images E3 "
      f"reached an accuracy of {yk(e3g[2], e3g[1])} (95% CI {pr1(e3g[3])}–{y1(e3g[4])}) and a macro-F1 of "
      f"{b3(e3g[5], e3g[6], e3g[7])} (Table S9).", "",
      "## Label smoothing", "",
      "All models were trained with label smoothing of 0.1, which limits the predicted confidence. The highest E3 confidence among "
      f"the 2,095 OOF predictions was {gv[1]}, and 99% of these confidences were at most {gv[2]}. This compresses the upper "
      "confidence bins and affects both the ECE (Fig. S1) and the LTT thresholds.", "",
      "## Label conflicts", "",
      "Photo groups whose files carried more than one class label were removed as a whole. Among the RGB images, these were "
      f"{ck.foto_grup.nunique()} photo groups with {len(ck)} files ({ckn['Surface_Obstruction']} SO, {ckn['Healthy']} H, "
      f"{ckn['Structural_Fault']} SF and {ckn['Electrical_Fault']} EF), each pairing SO with H "
      f"({ckg['Healthy+Surface_Obstruction']} groups), SF ({ckg['Structural_Fault+Surface_Obstruction']}) or EF "
      f"({ckg['Electrical_Fault+Surface_Obstruction']}). Separately, {len(tk)} of the {int(MAN.termal.sum())} thermal files lay in "
      f"{tk.foto_grup.nunique()} photo groups labelled both SF and SO, before all thermal images were excluded.", ""]

# Uzman uyumu: 20_etiket_denetimi_formu.py (form metni, çeviri), a_uzman, h_kappa_ci
kc = bul(r"Fleiss all\s+complete n=(\d+): ([\d.]+) \[([-\d.]+), ([-\d.]+)\]", U)
s += ["## Expert agreement", "",
      "The experts used a browser form in Turkish that showed each image without its label or a model prediction. In translation, "
      "the six answer options were electrical fault (EF), structural or physical fault (SF), healthy, surface covering (SO), more "
      "than one class, and not sure or image insufficient. The form gave the following class definitions.", "",
      "> EF, electrical defect (hot spot, burn or browning, diode or junction box, cell discolouration). SF, physical damage "
      "(broken or cracked glass, broken frame, delamination, impact). SO, a factor covering the surface (dust, dirt, snow, bird "
      "droppings, leaves, shade). Healthy, no evident defect or covering.", "",
      "The instruction above the images read as follows.", "",
      "> For each image, select the main class of the panel. Decide independently, without consulting the other experts.", "",
      "No severity scale, reference images or calibration round were provided. Fleiss' κ was computed over the six answer options "
      "and allowed a variable number of answers per image, because Expert 1 left two images unanswered. Restricted to the "
      f"{kc[1]} images with three answers, it was {kc[2]} (95% CI {kc[3]}–{kc[4]}). Krippendorff's α {atif('hayes2007')} treats "
      'unanswered images as missing and was computed over the six answer options and over the four classes with "mixed" and '
      '"not sure" treated as missing. Gwet\'s AC1 is less affected than κ by the frequent use of one category, such as SO by '
      "Expert 2. Confidence intervals came from 2,000 (Fleiss' κ) or 1,000 (Krippendorff's α and Gwet's AC1) bootstrap resamples "
      "of the images, and all measures are listed in Table S12.", ""]

# ================================================================== S1-S7 (önceki sürüm; yalnız S1 ve S3 notları eklendi)
# S1 maliyet
S = json.load(open("makale/ek_sayilar.json", encoding="utf-8"))["maliyet"]
rows = [("ResNetV2-101", "ImageNet", "ResNetV2-101", "1 × 10⁻⁴", "224"), ("EfficientNetV2-S", "ImageNet-21k", "EfficientNetV2-S", "1 × 10⁻⁴", "224"),
        ("ConvNeXt-T", "ImageNet-22k", "ConvNeXt-T", "1 × 10⁻⁴", "224"), ("ConvNeXt-T (384 px)", "ImageNet-22k", "ConvNeXt-T (384)", "1 × 10⁻⁴", "384"),
        ("ConvNeXt V2-T", "FCMAE + ImageNet-22k", "ConvNeXt V2-T", "1 × 10⁻⁴", "224"), ("Swin-T", "ImageNet-22k", "Swin-T", "5 × 10⁻⁵", "224"),
        ("DINOv2 ViT-B/14", "self-supervised (LVD-142M)", "DINOv2 ViT-B/14 (224)", "2 × 10⁻⁵", "224"),
        ("DINOv2 ViT-B/14 (392 px)", "self-supervised (LVD-142M)", "DINOv2 ViT-B/14 (392)", "2 × 10⁻⁵", "392")]
t1 = pd.DataFrame([[a, p, S[k]["param_M"], lr, i, S[k]["egitim_30ep_s"], S[k]["gecikme_ms_tta"]] for a, p, k, lr, i in rows],
                  columns=["Model", "Pre-training", "Parameters (M)", "Base learning rate", "Input (px)", "Training time (s)", "Latency (ms)"])
e3_ms = S["ConvNeXt-T"]["gecikme_ms_tta"] + S["Swin-T"]["gecikme_ms_tta"] + S["DINOv2 ViT-B/14 (392)"]["gecikme_ms_tta"]
s += ["# Table S1. Backbones, settings and computational cost", "",
      "Training time refers to one 30-epoch run (fold 0, seed 42). Latency is given per image at batch size 1 with bfloat16 and three-pass test-time augmentation on an NVIDIA RTX 4080 Laptop GPU (12 GB). The classification runs other than the few-shot experiments took 23.8 GPU-hours in total."
      f" The latency of E3, which combines ConvNeXt-T, Swin-T and DINOv2 (392 px), is approximately the sum of the member latencies, about {e3_ms:.0f} ms per seed with TTA."
      f" This gives about {e3_ms * 3:.0f} ms per image for the nine in-domain models and about {e3_ms * 15 / 1000:.1f} s for the 45 fold and seed models applied to external data, as upper estimates for sequential inference, since the in-domain E3 used single-view predictions.", "",
      md(t1), ""]

# S2 az örnek: oturum/alt koleksiyon gruplu tekrar (28b, ana sonuç) ve yalnız kopya gruplu eski bölme (28, karşılaştırma)
azs = pd.read_csv("asama2_sonuclar/az_ornek_gruplu.csv")
YON = {"ince_ayar_solarfcd": "Fine-tuned from SolarFCD", "ince_ayar_imagenet": "Fine-tuned from ImageNet",
       "dogrusal_solarfcd": "Logistic regression, SolarFCD features", "dogrusal_imagenet": "Logistic regression, ImageNet features",
       "1nn_solarfcd": "1-NN, SolarFCD features", "1nn_imagenet": "1-NN, ImageNet features"}
g2 = azs.groupby(["butce", "yontem"]).agg(n=("n_egitim", "mean"), acc=("acc", "mean"), sd=("acc", "std"), lo=("acc", "min"),
                                          hi=("acc", "max"), f1=("f1", "mean"), rH=("rec_H", "mean"), rSF=("rec_SF", "mean"),
                                          rSO=("rec_SO", "mean")).reset_index()
g2["sira"] = g2.yontem.map(list(YON).index)
g2 = g2.sort_values(["butce", "sira"])
lst = lambda xs: ", ".join(xs[:-1]) + " and " + xs[-1]
t2 = pd.DataFrame({"Labelled images": g2.n.round().astype(int), "Method": [("SolarFCD model without adaptation" if b == 0 else YON[y]) for b, y in zip(g2.butce, g2.yontem)],
                   "Accuracy, mean ± SD (range)": [f"{100 * m:.1f} ± {100 * d:.1f}% ({100 * l:.1f}–{100 * h:.1f}%)"
                                                   for m, d, l, h in zip(g2.acc, g2.sd, g2.lo, g2.hi)],
                   "Macro-F1": g2.f1.map(lambda v: f"{v:.3f}"),
                   "H / SF / SO recall": [f"{x:.2f} / {z:.2f} / {w:.2f}" for x, z, w in zip(g2.rH, g2.rSF, g2.rSO)]})
tn = azs.groupby("tekrar").n_test.first()
eski = pd.read_csv("asama2_sonuclar/az_ornek.csv").groupby(["oran", "baslangic"]).agg(n=("n_egitim", "mean"), acc=("acc", "mean")).reset_index()
es = eski[eski.baslangic == "solarfcd"].sort_values("oran")
ei = eski[eski.baslangic == "imagenet"].sort_values("oran")
s += ["# Table S2. Few-shot adaptation on RF-Fault with held-out acquisition sessions", "",
      f"Ten repetitions, each splitting the 1,108 four-class RF-Fault images, grouped into 39 acquisition sessions and sub-collections, "
      f"into a test half ({tn.min()}–{tn.max()} images) and an adaptation half, with 30–70% of every class in the test half. "
      f"Adaptation images were drawn as whole copy groups (same source-file stem or pHash distance ≤ 6), and the row with 0 images evaluates "
      f"the SolarFCD model without adaptation. The mean majority-class baseline of the test halves was {100 * azs.groupby('tekrar').test_cogunluk.first().mean():.1f}%. "
      f"Recalls are means over the repetitions. For comparison, the earlier split, which separated copies and near-duplicates (613 groups) "
      f"but not sessions and used one fixed test half of 552 images with three samples per budget, gave "
      f"{100 * es.acc.iloc[0]:.1f}% without target images and "
      + lst([f"{100 * v:.1f}%" for v in es.acc.iloc[1:]]) + " with about "
      + lst([f"{n:.0f}" for n in es.n.iloc[1:]]) + " images from the SolarFCD initialisation, against "
      + lst([f"{100 * v:.1f}%" for v in ei.acc]) + " from ImageNet.", "",
      md(t2), ""]

# S3 α duyarlılığı
rows = []
for et, ad in [("orij", "Original (2,095)"), ("orijbel", "Ambiguous removed (2,049)"), ("temiz", "Audited, E3 retrained (2,049)")]:
    for r in ("0.02", "0.05", "0.10"):
        t = open(f"asama2_sonuclar/secici_ek/{et}_r{r}.txt", encoding="utf-8").read()
        m = re.search(r"kapsam ([\d.]+) \(±([\d.]+)\), seçilenlerde doğruluk ([\d.na]+).*?ihlali oranı ([\d.]+)", t)
        acc = float(m.group(3)) if m.group(3) != "nan" else float("nan")
        rows.append([ad, f"{float(r):.0%}", yuzde(float(m.group(1)), float(m.group(2))), yuzde(acc), f"{float(m.group(4)):.3f}"])
s += ["# Table S3. Sensitivity of the LTT selective classifier to the target error α", "",
      "Ensemble E3, δ = 0.1, 1,000 scene-level 50/50 calibration/test splits. A dash means that no image was accepted in any split. The violation rate is the share of splits in which the error on the accepted test images exceeded α."
      " Splits without accepted images count as non-violations."
      f" At α = 2%, no split accepted any image with the original labels, all {ob_n} splits that accepted any image violated the bound with the ambiguous images removed, and {te_v} of the {te_n} accepting splits violated it with the audited labels."
      f" At α = 5% with the original labels, {cb_v} of the {cb_n} accepting splits ({cb_v / cb_n:.3f}) violated the bound.", "",
      md(pd.DataFrame(rows, columns=["Labels", "α", "Coverage", "Accuracy on accepted", "Violation rate"])), ""]

# S4 öğrenme hızı
t = open("asama2_sonuclar/ek_analiz.txt", encoding="utf-8").read()
lr = re.findall(r"^(ConvNeXt|DINOv2|Swin)\s+F1 ([\d.]+) vs ([\d.]+)\s+fark ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]\s+acc ([\d.]+)\s+SF ([\d.]+)", t, re.M)
ad = {"ConvNeXt": "ConvNeXt-T", "Swin": "Swin-T", "DINOv2": "DINOv2 (224 px)"}
sel = {"ConvNeXt": "×0.5 (5 × 10⁻⁵)", "Swin": "×3 (1.5 × 10⁻⁴)", "DINOv2": "×0.5 (1 × 10⁻⁵)"}
s += ["# Table S4. Learning-rate tuning by inner grouped validation", "",
      "Candidate multipliers ×0.5, ×1 and ×3 of the base learning rate were compared by the mean last-epoch macro-F1 on an inner scene-grouped validation split (12.5%), after which the model was retrained without validation (seed 42, pooled OOF).", "",
      md(pd.DataFrame([[ad[m], sel[m], b, a_, eksi(f"{float(d):+.3f} [{float(lo):+.3f}, {float(hi):+.3f}]"), yuzde(float(acc)), sf] for m, a_, b, d, lo, hi, acc, sf in lr],
                      columns=["Model", "Selected learning rate", "Macro-F1, base", "Macro-F1, tuned", "Difference [95% CI]", "Accuracy, tuned", "SF recall, tuned"])), ""]

# S5 çözünürlük
c = open("asama2_sonuclar/cozunurluk.txt", encoding="utf-8").read()
rr = re.findall(r"^(ConvNeXt-224|ConvNeXt-384|DINOv2-224|DINOv2-392)\s+acc ([\d.]+)\s+F1 ([\d.]+)\s+EF ([\d.]+) H ([\d.]+) SF ([\d.]+) SO ([\d.]+)", c, re.M)
ad5 = {"ConvNeXt-224": "ConvNeXt-T (224 px)", "ConvNeXt-384": "ConvNeXt-T (384 px)", "DINOv2-224": "DINOv2 (224 px)", "DINOv2-392": "DINOv2 (392 px)"}
s += ["# Table S5. Effect of input resolution (P3, three seeds, pooled OOF)", "",
      md(pd.DataFrame([[ad5[m], yuzde(float(acc)), f1, ef, h, sf, so] for m, acc, f1, ef, h, sf, so in rr],
                      columns=["Model", "Accuracy", "Macro-F1", "EF recall", "H recall", "SF recall", "SO recall"])), "",
      f"Paired macro-F1 differences (cluster bootstrap) were {M}0.013 [{M}0.043, +0.015] for ConvNeXt-T (384 px) relative to ConvNeXt-T (224 px), and +0.025 [+0.003, +0.049] for DINOv2 (392 px) relative to DINOv2 (224 px).", ""]

# S6 ResNet
r5 = pd.read_csv("asama2_sonuclar/resnet_duyarlilik.csv")
r5["Variant"] = r5.varyant.map({"dA": "A, inner grouped validation and F1-based selection", "dB": "B, 60 epochs", "dC": "C, learning rate 3 × 10⁻⁴"})
r5["acc"] = r5.acc.map(lambda v: yuzde(v))
for k in ["f1", "rec_El", "rec_He", "rec_St", "rec_Su"]:
    r5[k] = r5[k].map(lambda v: f"{v:.3f}")
s += ["# Table S6. ResNetV2-101 sensitivity (seed 42, P3, pooled OOF)", "",
      "Under the main protocol (three seeds), ResNetV2-101 reached an accuracy of 81.0% and a macro-F1 of 0.435.", "",
      md(r5[["Variant", "acc", "f1", "rec_El", "rec_He", "rec_St", "rec_Su"]].rename(columns={"acc": "Accuracy", "f1": "Macro-F1", "rec_El": "EF recall", "rec_He": "H recall", "rec_St": "SF recall", "rec_Su": "SO recall"})), ""]

# S7 tohumlar
t6 = pd.read_csv("asama2_sonuclar/tohum_bazinda.csv")
ad7 = {"convnext_tiny": "ConvNeXt-T", "convnextv2_tiny": "ConvNeXt V2-T", "resnetv2_101": "ResNetV2-101", "swin_tiny_patch4_window7": "Swin-T",
       "tf_efficientnetv2_s": "EfficientNetV2-S", "vit_base_patch14_dinov2": "DINOv2 (224 px)"}
t6["Model"] = t6.model.map(ad7).fillna(t6.model)
t6["acc"] = t6.acc.map(lambda v: yuzde(v))
for k in ["f1", "f1_lo", "f1_hi", "rec_El", "rec_He", "rec_St", "rec_Su"]:
    t6[k] = t6[k].map(lambda v: f"{v:.3f}")
t6["Macro-F1 [95% CI]"] = t6.f1 + " [" + t6.f1_lo + ", " + t6.f1_hi + "]"
s += ["# Table S7. Seed-level results under P3 (pooled OOF, 2,095 images)", "",
      md(t6[["Model", "tohum", "acc", "Macro-F1 [95% CI]", "rec_El", "rec_He", "rec_St", "rec_Su"]].rename(columns={"tohum": "Seed", "acc": "Accuracy", "rec_El": "EF recall", "rec_He": "H recall", "rec_St": "SF recall", "rec_Su": "SO recall"})), ""]

# ================================================================== S8 RF-Fault eşlemesi (r1_f, r1_b, r1_c, c_dis)
ETK = {"Structural_Fault": "SF", "Fault_unknown_type": "Fault of unknown type", "Surface_Obstruction": "SO", "Healthy": "H"}


def kombin(blok):
    r_ = re.findall(r"^\s*(Fault_unknown_type|Healthy|Structural_Fault|Surface_Obstruction)\s+([A-Z][A-Za-z +]*?)\s+(\d+)$", blok, re.M)
    return {(l_, cmb): int(n) for l_, cmb, n in r_}


f_tum = kombin(F.split("== retained")[0])
f_kal = kombin(F.split("== retained")[1].split("abstract words")[0])
assert sum(f_tum.values()) == 2763 and sum(f_kal.values()) == 1478 and set(f_kal) <= set(f_tum)
cok = sum(n for (_, cmb), n in f_kal.items() if "+" in cmb)
assert cok == int(bul(r"retained images with >1 box class: (\d+) of (\d+)", B)[1])
sat8 = []
for etk in ETK:
    ks = sorted([k for k in f_tum if k[0] == etk], key=lambda k: (-f_kal.get(k, 0), -f_tum[k]))
    for i, k in enumerate(ks):
        sat8.append([f"{ETK[etk]} ({n_(sum(v for (l_, _), v in f_kal.items() if l_ == etk))} retained)" if i == 0 else "",
                     k[1], n_(f_kal.get(k, 0)), n_(f_tum[k])])
sat8.append(["Total", "", n_(sum(f_kal.values())), n_(sum(f_tum.values()))])
sat8.append(["With more than one box class", "", n_(cok), n_(sum(n for (_, cmb), n in f_tum.items() if "+" in cmb))])
FN = {m_[0]: m_ for m_ in re.findall(
    r"^(ConvNeXt|Swin|DINOv2-392|E3)\s+fault/no-fault: manuscript mapping ([\d.]+)% \(baseline ([\d.]+)%, n=(\d+)\) \| "
    r"Defective\+soil/snow as no-fault ([\d.]+)% \(baseline ([\d.]+)%\) \| unknown excluded ([\d.]+)% \(baseline ([\d.]+)%, "
    r"n=(\d+)\) \| 4-class ([\d.]+)% \(n=(\d+)\) \| 4-class with Defective\+soil/snow as SO ([\d.]+)% \(n=(\d+), SO recall ([\d.]+)\)",
    B, re.M)}
assert len(FN) == 4
BAL = dict(re.findall(r"^\s+(ConvNeXt|Swin|DINOv2-392)\s+acc [\d.]+ \[Wilson [\d.]+, [\d.]+\] balanced ([\d.]+)", C, re.M))
BAL["E3"] = bul(r"E3 external recalls: .*balanced acc ([\d.]+)", C)[1]
MOD = ["ConvNeXt", "Swin", "DINOv2-392", "E3"]
e3fn = FN["E3"]


def pci(v, lo, hi):              # yüzde değerlerinden
    return f"{f(v):.1f}% [{f(lo):.1f}, {f(hi):.1f}]"


sat8b = [["Fault/no-fault, unknown type as fault (main analysis)", n_(e3fn[3]), f"{f(e3fn[2]):.1f}%"]
         + [pci(FN[m_][1], KB[m_][10], KB[m_][11]) for m_ in MOD],
         ["Fault/no-fault, defective with soiling or snow as no fault", n_(e3fn[3]), f"{f(e3fn[5]):.1f}%"]
         + [f"{f(FN[m_][4]):.1f}%" for m_ in MOD],
         ["Fault/no-fault, unknown type excluded", n_(e3fn[8]), f"{f(e3fn[7]):.1f}%"] + [f"{f(FN[m_][6]):.1f}%" for m_ in MOD],
         ["Four-class accuracy (main analysis)", n_(e3fn[10]), y1(cx[3])] + [pci(FN[m_][9], KB[m_][4], KB[m_][5]) for m_ in MOD],
         ["Four-class accuracy, defective with soiling or snow as SO", n_(e3fn[12]), "–"] + [f"{f(FN[m_][11]):.1f}%" for m_ in MOD],
         ["Balanced accuracy over H, SF and SO (main analysis)", n_(e3fn[10]), "–"] + [y1(BAL[m_]) for m_ in MOD]]
bil = dict(re.findall(r"'(Defective\+snow|Defective\+soiling|Defective only|Defective\+NonDefective)': (\d+)",
                      bul(r"unknown-type \(n=370\) composition: (\{[^}]*\})", B)[1]))
e3d = ast.literal_eval(bul(r"E3 predictions on Defective\+soiling/snow images \(n=(\d+)\): (\{[^}]*\})", B)[2])
sf_def = sum(v for (l_, cmb), v in f_kal.items() if l_ == "Structural_Fault" and "Defective" in cmb.split(" + "))
idr = bul(r"Imgdirty_\* in all RF-Fault: (\d+) removed (\d+) retained (\d+)\n with Defective box: (\d+)\n box combos: (\{[^}]*\})", C1)
idr_c = ast.literal_eval(idr[5])
s += ["# Table S8. RF-Fault label mapping and its sensitivity", "",
      "Box classes were mapped in the order physical damage (SF), defective (fault of unknown type, fault/no-fault analysis only), "
      "bird drop, dust, dusty or snow (SO) and non-defective (H). Of the "
      f"{n_(sum(f_kal.values()))} images retained after overlap screening, {cok} carry more than one box class.", "",
      f"*(a) Box-class combinations by mapped label, for the retained images and all {n_(sum(f_tum.values()))} RF-Fault images*", "",
      tablo(["Mapped label", "Box classes", "Retained images", "All images"], sat8), "",
      "*(b) Accuracy of the external predictions under alternative mappings*", "",
      tablo(["Mapping and metric", "Images", "Baseline", "ConvNeXt-T", "Swin-T", "DINOv2 (392 px)", "E3"], sat8b), "",
      "The baseline is the accuracy of always predicting the majority class, which is fault in the first row, no fault in the "
      "second and third rows and SF in the fourth row. Brackets give 95% CIs from a cluster bootstrap over source-file stems "
      "(2,000 resamples). Each model averages its 15 fold and seed models with TTA, as in Table 7 of the main text. "
      f"E3 predicted SO for {e3d['Surface_Obstruction']} of the {sum(e3d.values())} retained images with a defective and a "
      f"soiling or snow box, and with this remapping its SO recall was {f(e3fn[13]):.3f}. Of the 370 retained images of unknown "
      f"fault type, {bil['Defective+snow']} also carry a snow box, {bil['Defective+soiling']} a dust, dusty or bird-drop box, "
      f"{bil['Defective only']} only a defective box and {bil['Defective+NonDefective']} a defective and a non-defective box, "
      f"and {sf_def} of the 527 SF images also carry a defective box. RF-Fault also contains {idr[1]} files named like the K-dirty "
      f"(SO) images of SolarFCD (Imgdirty), of which {idr[4]} carry a defective box, {idr_c['1,3']} of them with a dusty box as "
      f"the only other class, and screening removed {idr[2]} of these {idr[1]} files. Re-weighting the RF-Fault recalls of E3 to the SolarFCD "
      f"proportions of H, SF and SO gives {y1(rw[1])}, compared with an in-domain E3 accuracy of {y1(ci_[2])} and a balanced "
      f"accuracy of {y1(ci_[3])} on the same three classes.", ""]

# ================================================================== S9 gruplama, denetim geçmişi, katlar
sab = {k: bul(rf"^{k} = ([\d.]+)", M8)[1] for k in ("UZUN", "NKP", "MAGSAC_PX", "ALTBILGI")}
lowe = math.sqrt(f(bul(r"vr\[:, 0\] < ([\d.]+) \* vr\[:, 1\]", M8)[1]))
umin_foto = bul(r'foto = \(x\["gecerli"\] & \(x\["inl"\] >= args\.foto_inlier\) & \(x\["umin"\] >= (\d+)\)', M8)[1]
alt_esik = f(bul(r'alt = \(x\["alt_pay"\] >= ([\d.]+)\)', M8)[1])
kat = pd.read_csv("asama2/kat_sinif_v1.csv", encoding="utf-8-sig")
for k_ in ("tarama_r70", "tarama_r80", "dosyano_kenar"):
    assert k_ in PR
assert PR["dosyano_kenar"] is False and KS["dosyano"] == 0


def oran(x):
    return f"{f(x) * 100:.0f}%"


ofs = PR["aday_ofset"].replace("-", "–")
sat9a = [
    ["Strict duplicates", f"MD5 equality, or pHash Hamming distance ≤ {PR['phash_esik']} over all 64 pairs of the eight dihedral "
     "transformations", f"{n_(KS['kati(md5∪phash)'])} links, {n_(OZ['kati_grup_rgb']['grup'])} RGB strict groups"],
    ["Candidate pairs", f"DINOv2 top {PR['aday_knn']} per strict group, pHash ≤ {PR['aday_phash']}, file numbers ≤ {PR['aday_no']} "
     f"apart within a family, top {PR['aday_capraz']} H–SO neighbours, Kaggle clean–dirty number offsets of {ofs} with cosine ≥ "
     f"{PR['aday_ofset_tau']:.2f}, exhaustive RootSIFT screen of all RGB pairs, all visually reviewed pairs", f"{n_(OZ['aday_cift'])} pairs"],
    ["Geometric matching", f"RootSIFT ({n_(sab['UZUN'])} px, {n_(sab['NKP'])} keypoints), mutual nearest neighbours with ratio test "
     f"{lowe:.1f}, MAGSAC++ homography ({f(sab['MAGSAC_PX']):.0f} px), mirrored image also matched, both matching directions "
     "combined, valid if the determinant is positive and the image corners map to a convex quadrilateral", "every candidate pair"],
    ["Same-photograph link", f"valid homography, ≥ {PR['foto_inlier']} inliers at ≥ {umin_foto} distinct positions, inlier hull ≥ "
     f"{oran(PR['foto_kapsam'])} of the smaller image and high-pass NCC ≥ {PR['foto_ncc']:.2f} (≥ {PR['foto_ncc_capraz']:.2f} if the "
     f"labels differ), or same-label copies with ≥ {PR['foto_filigranli_inlier']} inliers, hull ≥ {oran(PR['foto_filigranli_kapsam'])} "
     f"and NCC ≥ {PR['foto_filigranli_ncc']}, or a visual same-photograph decision",
     f"{KS['foto_geometrik']} geometric ({KS['foto_filigranli']} of them watermarked copies) and {KS['foto_gorsel']} visual links, "
     f"{n_(OZ['foto_grup_rgb']['grup'])} RGB photo groups"],
    ["Same-scene link", f"geometric (valid homography, ≥ {PR['sahne_inlier']} inliers at ≥ {PR['sahne_ayrik']} distinct positions, "
     f"{'the same' if PR['azinlik_inlier'] == PR['sahne_inlier'] == PR['sahne_ayrik'] else PR['azinlik_inlier']} for pairs with "
     f"SF or EF images, no footer or watermark match), DINOv2 (mutual {PR['dino_k']} nearest strict groups over dihedral "
     f"embeddings, cosine ≥ {PR['dino_tau']}), EXIF (same camera model, ≤ {PR['exif_dk']:.0f} min apart) or a visual same-scene or "
     "uncertain decision", f"{KS['sahne_geometrik']} geometric, {KS['dino_mknn']} DINOv2, {KS['exif']} EXIF and {KS['sahne_gorsel']} "
     f"visual links, {n_(OZ['sahne_grup_rgb']['grup'])} RGB scene groups"],
    ["Vetoes", f"footer match if ≥ {alt_esik * 100:.0f}% of the inliers lie in the bottom {f(sab['ALTBILGI']) * 100:.0f}% of both "
     f"images, watermark or template match if the frames coincide (corner deviation < {PR['filigran_kimlik']:.2f}) with NCC < "
     f"{PR['filigran_ncc']} and either ≥ {oran(PR['filigran_altpay'])} of the inliers in the footer, hull < "
     f"{oran(PR['filigran_kapsam'])} or cosine < {PR['filigran_s']}, a definite visual unrelated decision blocks photo and "
     "geometric scene links, and a visual same-scene decision blocks photo links",
     f"{KS['foto_yasagi_ile_dusen']} photo and {KS['ilgisiz_vetosu_ile_dusen_sahne']} scene links blocked by visual decisions"],
    ["File-number links", "unverified links between nearby Kaggle file numbers switched off", f"{KS['dosyano']} links"],
    ["Exclusions and representatives", "thermal files and photo groups with conflicting labels removed, one representative per "
     "remaining photo group (largest pixel area, then shortest path, then name)",
     f"{OZ['termal_dosya']} thermal and {OZ['celiskili_rgb_dosya']} conflicting RGB files removed, {n_(OZ['v1_ornek'])} "
     f"representatives of {n_(OZ['uygun_dosya'])} eligible files"],
    ["Folds", f"StratifiedGroupKFold with {PR['kat']} folds, shuffled with seed {PR['tohum']}, strata class × source, groups = scene "
     "groups", f"{n_(OZ['sahne_grup_v1']['grup'])} scene groups, {kat.toplam.min()}–{kat.toplam.max()} images per fold"]]
# NOTLAR.md §9: r1 satır 184-187 (2.275; 11/60), r2 222-244 (2.106; en büyük 118; 14/120), r3 279-315 (2.105; 72; 15/160),
# r4 322-347 (2.095; 226; 5/160, yalnız belirsizlerle 9/160), r4.1 366-385 (sahne görsel kenarı 469 -> 619; kör denetim yok)
TUR = [("r1", "MD5 and dihedral pHash ≤ 4 over 16 of the 64 orientation pairs, SIFT matching of DINOv2 candidates", 2275, "n.r.",
        "60 random images", 11, 60),
       ("r2", "all 64 orientation pairs, model-independent candidates, RootSIFT with MAGSAC++ and NCC, EXIF, file-number and "
        "visual links", 2106, 118, "120 random queries", 14, 120),
       ("r3", "deeper symmetric matcher at 1,400 px, further candidate sources, unverified file-number links switched off",
        2105, 72, "160 random queries", 15, 160),
       ("r4", "matching in both directions, narrower watermark veto, uncertain visual decisions as scene links", 2095, 226,
        "160 random queries from 1,816 images not queried before", 5, 160),
       ("r4.1 (final)", f"{KS['sahne_gorsel'] - 469} visual scene links added from the r4 audit and a targeted screen of 2,121 "
        "Kaggle pairs, folds redrawn", int(OZ["v1_ornek"]), int(OZ["sahne_grup_v1"]["en_buyuk"]), "not audited", None, None)]
sat9b = []
for tur, deg, n, eb, orn, k_, nn in TUR:
    if k_ is None:
        sz = "–"
    else:
        lo, hi = wilson(k_, nn)
        sz = f"{k_}/{nn}, {k_ / nn * 100:.1f}% ({lo * 100:.1f}–{hi * 100:.1f}%)"
    sat9b.append([tur, deg, n_(n), eb if isinstance(eb, str) else n_(eb), orn, sz])
# görsel çift kararları (e_protokol) ve insan kontrolü (insan_kontrol/sonuc: ozet.txt, cift_karsilastirma.csv)
vp = bul(r"visual_pair_reviews: (\d+) decisions; \{'unrelated': (\d+), 'same_scene': (\d+), 'uncertain': (\d+), 'same_photo': (\d+)\}; "
         r"single decision (\d+); certainty \{'definite': (\d+), 'probable': (\d+)\}", E)
ve = bul(r"edges from visual review: photo_visual (\d+) scene_visual (\d+)", E)
assert int(ve[1]) == KS["foto_gorsel"] and int(ve[2]) == KS["sahne_gorsel"]
ik_neg = bul(r"Kaçırılan bağ .*?: (\d+)/(\d+) = [\d.]+% \(GA ([\d.]+)%-([\d.]+)%\)", IK)
ik_poz = bul(r"doğrulanması \(kesinlik\): (\d+)/(\d+) = ([\d.]+)% \(GA ([\d.]+)%-([\d.]+)%\)", IK)
ik_a = bul(r"Bölüm A .*?: (\d+)/(\d+) = ", IK)
CK = pd.read_csv("insan_kontrol/sonuc/cift_karsilastirma.csv", encoding="utf-8-sig")
iki = {"ayni_foto": "iliskili", "ayni_sahne": "iliskili", "ilgisiz": "ilgisiz"}
CB = CK[CK.bolum == "B"]
ckd = CB[CB.yz.isin(iki) & CB.insan.isin(iki)]
kap = cohen_kappa_score(ckd.yz.map(iki), ckd.insan.map(iki))
assert abs(kap - f(bul(r"İkili Cohen kappa .*?n=(\d+)\): ([\d.]+)", IK)[2])) < 5e-4
nB = (CK.bolum == "B").sum()
nB_neg = (CB.yz == "ilgisiz").sum()
# kat bazında E3 (r1_a) ve en büyük sahne grubu (r1_a, e_protokol, buyuk_gruplar.csv)
kt = re.findall(r"^\s+fold (\d): n=(\d+) acc ([\d.]+) macroF1 ([\d.]+) SF (\d+) EF (\d+)", A, re.M)
assert len(kt) == 5
g36 = bul(r"largest scene group id (\d+): size (\d+) folds \[np\.int64\((\d+)\)\] share of fold ([\d.]+) \(fold size (\d+)\)", A)
g36a = bul(r"families: (\{[^}]*\}) labels: (\{[^}]*\})", A)
g36aile, g36et = ast.literal_eval(g36a[1]), ast.literal_eval(g36a[2])
g36i = bul(r"E3 acc inside ([\d.]+); pooled without it n=(\d+) acc ([\d.]+) macroF1 ([\d.]+)", A)
g36n = ast.literal_eval(bul(r"next largest groups: (\{[^}]*\})", A)[1])
f14 = bul(r"E3 fold 0: acc ([\d.]+) \(n=(\d+)\); folds 1-4: acc ([\d.]+); macro-F1 without fold 0 ([\d.]+)", E)
gr_t = bul(r"^all edges\s*: groups among the 2,095 = (\d+), singletons (\d+), largest (\d+)", E)
gr_v = bul(r"^without scene_visual\s*: groups among the 2,095 = (\d+), singletons (\d+), largest (\d+)", E)
bg = pd.read_csv("asama2/buyuk_gruplar.csv", encoding="utf-8-sig")
bg36 = ast.literal_eval(bg.loc[bg.sahne_grup == int(g36[1]), "kenar"].iloc[0])
assert int(g36[2]) == int(OZ["sahne_grup_v1"]["en_buyuk"]) == int(gr_t[3])
ef_t, sf_t = sum(int(k[5]) for k in kt), sum(int(k[4]) for k in kt)
assert [int(k[1]) for k in kt] == kat.toplam.tolist() and int(f14[2]) == int(kt[0][1])
sat9c = [[k[0], k[1], k[5], k[4], yk(k[2], k[1]), d3(k[3])] for k in kt]
sat9c += [["1–4 pooled", n_(2095 - int(f14[2])), str(ef_t - int(kt[0][5])), str(sf_t - int(kt[0][4])),
           yk(f14[3], 2095 - int(f14[2])), d3(f14[4])],
          ["All pooled", n_(2095), str(ef_t), str(sf_t), pb(e3[1], e3[2], e3[3], 2095), b3(e3[4], e3[5], e3[6])],
          [f"All pooled without scene group {g36[1]}", n_(e3g[1]), str(ef_t), str(sf_t), pb(e3g[2], e3g[3], e3g[4], e3g[1]),
           b3(e3g[5], e3g[6], e3g[7])],
          [f"Scene group {g36[1]} only", n_(g36[2]), "0", "0", yk(g36i[1], g36[2]), "–"]]
AIL = {"Imgclean": "K-clean", "Imgdirty": "K-dirty"}
s += ["# Table S9. Duplicate and scene grouping, audit history and per-fold results of E3", "",
      "*(a) Final grouping rules and thresholds (round r4.1), with the links and groups they produced*", "",
      tablo(["Step", "Rule (final parameters)", "Result in the final manifest"], sat9a), "",
      f"Visual decisions on {n_(vp[1])} candidate pairs ({n_(vp[2])} unrelated, {vp[3]} same scene, {vp[4]} uncertain and "
      f"{vp[5]} same photograph, {n_(vp[7])} of them definite and {n_(vp[8])} probable) were all made by multimodal "
      "language-model reviewers (Claude Opus 5.5, Anthropic) under the author's supervision, and "
      f"{n_(vp[6])} pairs received a single decision. They contributed the {ve[1]} visual photo links and {ve[2]} visual scene "
      f"links in (a). The author checked a sample of {len(CK)} of these decisions by hand without seeing the reviewers' verdicts, "
      f"namely {nB} decisions drawn at random ({nB - nB_neg} related or uncertain and {nB_neg} definitely unrelated) and the "
      f"{len(CK) - nB} leakage or uncertain pairs adjudicated in the last blinded audit. None of the {ik_neg[2]} pairs judged "
      f"definitely unrelated was related ({ik_neg[1]}/{ik_neg[2]}, 95% CI {ik_neg[3]}–{ik_neg[4]}%), whereas only {ik_poz[1]} of "
      f"the {ik_poz[2]} pairs judged related were confirmed ({ik_poz[3]}%, 95% CI {ik_poz[4]}–{ik_poz[5]}%), with Cohen's "
      f"κ = {d2(kap)} over the {len(ckd)} pairs with a definite verdict. The author judged only {ik_a[1]} of the {ik_a[2]} audit "
      "pairs related. Thresholds were set on visually labelled positive and hard-negative pairs from earlier audits. When "
      "unverified links between nearby Kaggle file numbers were enabled in an earlier round, they chained unrelated jobs into a "
      "group of 343 images, so they were switched off. A complete rebuild of the grouping takes about one hour without caches.", "",
      "*(b) Grouping rounds and blinded audits of residual same-scene leakage*", "",
      tablo(["Round", "Main change", "Evaluation set", "Largest scene group", "Audit sample", "Residual leakage (95% CI)"], sat9b), "",
      "Each audit drew new random queries. From round r2 onwards, two blinded language-model reviewers and an adjudicator "
      "inspected every candidate partner of each query in other folds, and the author checked all flagged pairs of the last "
      "audit. Counting queries with only "
      "uncertain partners as leaks would give 9/160 (5.6%, 95% CI 3.0–10.3%) in the r4 audit, and the r3 and r4 audits found no "
      "same-photograph leakage. The final round r4.1 was not audited again, so the r4 estimate indicates rather than strictly "
      "bounds its residual leakage. n.r., not recorded.", "",
      "*(c) E3 by fold and without the largest scene group (pooled OOF, original labels)*", "",
      tablo(["Fold", "Images", "EF", "SF", "Accuracy [95% CI]", "Macro-F1 [95% CI]"], sat9c), "",
      f"Scene group {g36[1]} holds {g36[2]} images ({g36et['H']} H and {g36et['SO']} SO) from five source families ("
      + ", ".join(f"{k_} {v}" for k_, v in g36aile.items())
      + f"). It lies entirely in fold {g36[3]}, where it makes up {y1(g36[4])} of the {g36[5]} images, and the next largest "
      f"groups have {', '.join(str(v) for v in list(g36n.values())[:-1])} and {list(g36n.values())[-1]} images. Across all RGB "
      f"files, the group is held together mainly by {bg36['sahne_geometrik']} geometric and {bg36['sahne_gorsel']} visual scene "
      "links among job series of cleaning companies, and in the r4 audit, when it held 226 images, 6 of 20 sampled links were "
      "judged wrong. It is therefore partly a chain of similar templates and textures rather than one installation. Without the "
      f"visual scene links, the evaluation set would form {n_(gr_v[1])} scene groups ({n_(gr_v[2])} singletons) of at most "
      f"{gr_v[3]} images, compared with {n_(gr_t[1])} groups ({n_(gr_t[2])} singletons) and a largest group of {gr_t[3]} images "
      "with them.", ""]

# ================================================================== S10 P1 ayrıştırma (r1_a4, r1_a)
P1 = {m_[0]: m_ for m_ in re.findall(
    r"^(\w+)\s+P1 no-TTA: all ([\d.]+) \(n=375\) \| RGB ([\d.]+) \(n=364\) \| leak-free files ([\d.]+) \(n=162\) \| leak-free reps "
    r"([\d.]+) \(n=99\) \| test-file reps [\d.]+ \(n=189\)\n\s+P3 s42 OOF: all ([\d.]+)", A4, re.M)}
AY = {m_[0]: m_ for m_ in re.findall(r"^(\w+)\s+same leak-free images \(n=99\): P1 ([\d.]+) vs P3 ([\d.]+)", A4, re.M)}
assert len(P1) == len(AY) == 5
lf = bul(r"RGB test files with no same-photo copy in train: (\d+) (\{[^}]*\})", A)
lfd = ast.literal_eval(lf[2])
lfg = bul(r"leak-free by photo group \(no same-photo copy in train/valid\) (\d+) files, (\d+) photo groups; eligible \d+; "
          r"representatives (\d+)", A4)
ad10 = {"resnetv2_101": "ResNetV2-101", "efficientnetv2_s": "EfficientNetV2-S", "convnext_tiny": "ConvNeXt-T",
        "swin_tiny": "Swin-T", "dinov2": "DINOv2 (224 px)"}
for k_ in ad10:
    assert P1[k_][4] == AY[k_][1]          # aynı 99 fotoğrafta P1 değeri iki satırda aynı
NS = (375, 364, int(lf[1]), int(lfg[3]))       # P1 alt kümeleri; P3 aynı 99 fotoğraf ve 2.095 görüntü
sat10 = [[ad10[k_], *[yk(P1[k_][i], n) for i, n in zip((1, 2, 3, 4), NS)], yk(AY[k_][2], NS[3]), yk(P1[k_][5], 2095)]
         for k_ in ad10]
dn = P1["dinov2"]


def kes(x, n):                  # sayımdan kesin oran
    return round(f(x) * n) / n


kop = (kes(dn[1], 375) - kes(dn[4], NS[3])) * 100
kaz = (kes(AY["dinov2"][2], NS[3]) - kes(dn[4], NS[3])) * 100
nuf = (kes(AY["dinov2"][2], NS[3]) - kes(dn[5], 2095)) * 100
term = abs(kes(dn[1], 375) - kes(dn[2], 364)) * 100
assert abs(kop - kaz + nuf - (kes(dn[1], 375) - kes(dn[5], 2095)) * 100) < 1e-9
s += ["# Table S10. Decomposition of the P1 accuracy with single-view predictions (seed 42)", "",
      "Accuracy of the seed-42 models under P1 on nested subsets of the official test split, and of the corresponding P3 models "
      "on the same photographs and on the evaluation set. All values are single-view, whereas Table 3 of the main text gives P1 "
      "with TTA, which raises DINOv2 from 97.3% to 97.6% on all test files.", "",
      tablo(["Model", "P1, all 375 test files", "P1, 364 RGB test files", f"P1, {lf[1]} RGB files without a training copy",
             f"P1, {lfg[3]} photographs", f"P3, same {lfg[3]} photographs", "P3, all 2,095 images"], sat10), "",
      f"The {lf[1]} RGB test files without a same-photograph copy in the training or validation split ({lfd['H']} H, "
      f"{lfd['SO']} SO, {lfd['SF']} SF and {lfd['EF']} EF) form {lfg[2]} photo groups, and {lfg[3]} of these groups are "
      "represented in the evaluation set by an official test file, which is counted once. For DINOv2, removing the 11 thermal "
      f"test files changed the P1 accuracy by {'less than 0.1' if term < 0.1 else f'{term:.1f}'} points. Duplicates account for "
      f"{kop:.1f} points of the drop from {yk(dn[1], 375)} (P1, all test files) to {yk(dn[5], 2095)} (P3), the P3 model was "
      f"{kaz:.1f} point more "
      f"accurate than the P1 model on the same {lfg[3]} photographs, and the harder evaluation population accounts for the "
      f"remaining {nuf:.1f} points. The comparisons on {lfg[3]} photographs rest on few images and are indicative only.", ""]

# ================================================================== S11 kaynak aileleri (r1_a, ek_sayilar, doğrulayıcı taslağı R1.2)
AL = {m_[0]: m_ for m_ in re.findall(r"^\s+(K-clean|K-dirty|PD-\w+)\s+n=\s*(\d+) acc ([\d.]+) pred (\{[^}]*\})", A, re.M)}
BOY = {m_[0]: m_ for m_ in re.findall(r"^\s+family (\S+)\s+n=\s*(\d+) median WxH (\d+)x(\d+)", A, re.M)}
AAC = EKS["aile_acc"]
SN = {"Healthy": "H", "Surface_Obstruction": "SO", "Electrical_Fault": "EF", "Structural_Fault": "SF"}
AILE = [("K-clean", "Imgclean", "H"), ("PD-clean", "Clean", "H"), ("K-dirty", "Imgdirty", "SO"), ("PD-dust", "Dust", "SO"),
        ("PD-bird", "Bird", "SO"), ("PD-snow", "Snow", "SO"), ("PD-electrical", "Electrical", "EF"), ("PD-physical", "Physical", "SF")]
sat11a = []
for ai, dosya, sin in AILE:
    n, acc, pr = AL[ai][1], AL[ai][2], ast.literal_eval(AL[ai][3])
    assert int(n) == AAC[ai]["n"] and abs(f(acc) - AAC[ai]["acc"]) < 1e-3 and int(BOY[ai][1]) == int(n)
    hata = sorted([(v, k_) for k_, v in pr.items() if k_ != sin and v > 0], reverse=True)
    sat11a.append([f"{ai} ({dosya})", sin, n_(n), pb(AAC[ai]["acc"], AAC[ai]["lo"], AAC[ai]["hi"]),
                   ", ".join(f"{k_} {v}" for v, k_ in hata) if hata else "none",
                   f"{n_(BOY[ai][2])} × {n_(BOY[ai][3])}"])
# doğrulayıcının taslağı (hakem_sonuc.json, R1.2 proposed_fix (i)); yazar IEA PVPS T13-01, IEC 61215-2 MQT 01 ve
# IEC TS 62446-3 ile karşılaştırmalı (plan O3)
sat11b = [["H (K-clean, PD-clean)", "no visible anomaly", "none detectable visually", "only the absence of visible anomalies",
           "none or routine inspection"],
          ["SO (K-dirty, PD-dust)", "dust and dirt films", "soiling, an operational condition rather than a module failure",
           "visible, severity not measurable", "cleaning when the soiling ratio justifies it"],
          ["SO (PD-bird)", "bird droppings", "localised hard shading, possible hot spots and bypass-diode activation", "visible",
           "spot cleaning, IR check"],
          ["SO (PD-snow)", "snow cover", "seasonal cover", "visible", "usually none"],
          ["EF (PD-electrical)", "burn marks, browning, discolouration",
           "burn marks and hot spots, encapsulant discolouration, junction-box failure",
           "only when externally visible, hot spots and diode faults need IR or electrical tests",
           "IR or electrical confirmation, repair or replacement"],
          ["SF (PD-physical)", "broken glass, frame damage, delamination",
           "glass and frame breakage, delamination, cell cracks that need EL", "glass and frame damage visible",
           "replacement and safety check of the insulation"]]
RZ = {m_[0]: m_ for m_ in re.findall(
    r"^\s+(all|Electrical_Fault|Healthy|Structural_Fault|Surface_Obstruction)\s+n=\s*(\d+) median WxH (\d+)x(\d+) \| median long "
    r"side \d+ \| min width \d+ \| short side <224: ([\d.]+), <392: ([\d.]+) \| short side p10/p50/p90 \[\d+, (\d+), \d+\]", A, re.M)}
assert len(RZ) == 5
sat11c = [[{"all": "All", **SN}[k_], n_(RZ[k_][1]), f"{n_(RZ[k_][2])} × {n_(RZ[k_][3])}", n_(RZ[k_][6]), y1(RZ[k_][4]),
           y1(RZ[k_][5])] for k_ in ("all", "Healthy", "Surface_Obstruction", "Electrical_Fault", "Structural_Fault")]
e3a = bul(r"^n=2095 acc ([\d.]+) macroF1 ([\d.]+)", A)
ag3 = bul(r"3 action groups \(H / SO / EF\+SF\): acc ([\d.]+) macroF1 ([\d.]+) recalls none ([\d.]+) clean ([\d.]+) repair ([\d.]+)", A)
ag2 = bul(r"action needed \(SO/EF/SF\) vs no action \(H\): acc ([\d.]+) baseline ([\d.]+) recall ([\d.]+) precision ([\d.]+) "
          r"missed (\d+) false alarms (\d+)", A)
fa = bul(r"fault \(EF\+SF\) vs no-fault \(H\+SO\): acc ([\d.]+) baseline ([\d.]+) TP (\d+) FP (\d+) FN (\d+) recall ([\d.]+) "
         r"precision ([\d.]+)", A)
prv = dict(re.findall(r"precision at fault prevalence (\d+)%: ([\d.]+)", A))
kac = {(a_, b_): int(n) for a_, b_, n, _ in re.findall(r"true (SF|EF) -> pred (SO|H): (\d+) of (\d+)", A)}
sn1 = bul(r"SO recall all ([\d.]+) \(n=(\d+)\), without snow ([\d.]+) \(n=(\d+)\)", A)
sn2 = bul(r"SO precision all ([\d.]+), without snow images ([\d.]+)", A)
sn3 = bul(r"without snow: n=(\d+) acc ([\d.]+) macroF1 ([\d.]+)", A)
cm0 = EKS["cm_0"]
n_eylem = sum(cm0[0]) + sum(cm0[2]) + sum(cm0[3])
cz = {m_[0]: m_ for m_ in re.findall(r"^(DINOv2-224|DINOv2-392)\s+acc [\d.]+\s+F1 [\d.]+\s+EF [\d.]+ H [\d.]+ SF ([\d.]+) SO", c, re.M)}
sat11d = [["Three action groups (H no action, SO cleaning, EF or SF repair)", y1(ag3[1]), "–", d3(ag3[2]),
           f"no action {d3(ag3[3])}, cleaning {d3(ag3[4])}, repair {d3(ag3[5])}", "–"],
          ["Action needed (SO, EF or SF) versus H", y1(ag2[1]), y1(ag2[2]), "–", f"{d3(ag2[3])}", f"{d3(ag2[4])}"],
          ["Fault (EF or SF) versus no fault (H or SO)", y1(fa[1]), y1(fa[2]), "–", d3(fa[6]),
           f"{d3(fa[7])} at {y1(1 - f(fa[2]))}, {d3(prv['2'])} at 2% and {d3(prv['1'])} at 1% prevalence"],
          [f"All classes without the {int(sn1[2]) - int(sn1[4])} PD-snow images", f"{y1(sn3[2])} (all images {y1(e3a[1])})", "–",
           f"{d3(sn3[3])} (all images {d3(e3a[2])})", f"SO {d3(sn1[3])} (all images {d3(sn1[1])})",
           f"SO {d3(sn2[2])} (all images {d3(sn2[1])})"]]
s += ["# Table S11. Source families, failure modes, native resolution and action-oriented results of E3", "",
      "*(a) Source families of the evaluation set with the E3 accuracy (95% cluster-bootstrap CI, as in Fig. 7c of the main text), "
      "the classes predicted for misclassified images and the median native size*", "",
      tablo(["Family (file names)", "Class", "Images", "E3 accuracy [95% CI]", "E3 errors by predicted class",
             "Median width × height (px)"], sat11a), "",
      "*(b) Indicative mapping of the classes and families to failure modes, visibility and O&M response*", "",
      tablo(["Class (families)", "Visible content", "Related failure mode or condition", "Visibility in front-side RGB",
             "Typical O&M response"], sat11b), "",
      "*(c) Native resolution of the evaluation images by class*", "",
      tablo(["Class", "Images", "Median width × height (px)", "Median short side (px)", "Short side < 224 px",
             "Short side < 392 px"], sat11c), "",
      "*(d) E3 under action-oriented groupings and without snow images (pooled OOF, original labels)*", "",
      tablo(["Grouping", "Accuracy", "Majority baseline", "Macro-F1", "Recall", "Precision"], sat11d), "",
      "Every source family maps to a single class, so the families also show where the four classes merge different conditions "
      "with different O&M responses. The mapping in (b) is indicative and was not used for labelling. Of the "
      f"{int(fa[5]) + int(fa[3])} fault images, {fa[5]} were missed, {kac['SF', 'SO']} SF and {kac['EF', 'SO']} EF images as SO and "
      f"{kac['SF', 'H']} SF and {kac['EF', 'H']} EF images as H. In the action grouping, {ag2[5]} of the {n_(n_eylem)} images "
      f"that need action were called H and {ag2[6]} H images were flagged. The EF images are the smallest, with "
      f"{y1(RZ['Electrical_Fault'][5])} of them below 392 px on the short side, and DINOv2 at 392 px raised the SF recall from "
      f"{cz['DINOv2-224'][1]} to {cz['DINOv2-392'][1]} (Table S5).", ""]

# ================================================================== S12 uzman uyumu (h_kappa_ci, a_uzman, r1_b)
kv = {m_[0]: m_ for m_ in re.findall(r"^(all|candidates|controls)\s+n=(\d+) Fleiss \(variable n\) ([-\d.]+) \[([-\d.]+), ([-\d.]+)\]", K, re.M)}
kcc = {m_[0]: m_ for m_ in re.findall(r"Fleiss (all|candidates|controls)\s+complete n=(\d+): ([-\d.]+) \[([-\d.]+), ([-\d.]+)\]", U)}
kra = {m_[0]: m_ for m_ in re.findall(
    r"^\s+(all|candidates|controls)\s+Krippendorff alpha \(6 options, missing allowed\) ([-\d.]+); \(4 classes, other answers "
    r"missing\) ([-\d.]+) \[([-\d.]+), ([-\d.]+)\]; Gwet AC1 \(6 options\) ([-\d.]+) \[([-\d.]+), ([-\d.]+)\]; unanimous ([\d.]+)", U, re.M)}
ac4 = {"all": bul(r"Gwet AC1: 6 categories [\d.]+ \(items \d+\); mixed/not_sure as missing ([\d.]+)", B)[1]}
for g_ in ("candidates", "controls"):
    ac4[g_] = bul(rf"^\s+{g_}: Krippendorff alpha \(4 cls, others missing\) [\d.]+, Gwet AC1 ([\d.]+)", B)[1]
dur = ast.literal_eval(bul(r"status by candidate: (\{.*\})", B)[1])
GR = ("all", "candidates", "controls")


def durum(g_):
    ad_ = {"all": (True, False), "candidates": (True,), "controls": (False,)}[g_]
    return ", ".join(str(sum(dur[(a_, d_)] for a_ in ad_)) for d_ in ("kept", "changed", "ambiguous"))


sat12a = [["Images", *[kv[g_][1] for g_ in GR]],
          ["Fleiss' κ, six answers, variable number of answers [95% CI]", *[b3(*kv[g_][2:5]) for g_ in GR]],
          ["Fleiss' κ, six answers, images with three answers [95% CI] (images)", *[f"{b3(*kcc[g_][2:5])} ({kcc[g_][1]})" for g_ in GR]],
          ["Krippendorff's α, six answers", *[d3(kra[g_][1]) for g_ in GR]],
          ['Krippendorff\'s α, four classes, "mixed" and "not sure" missing [95% CI]', *[b3(*kra[g_][2:5]) for g_ in GR]],
          ["Gwet's AC1, six answers [95% CI]", *[b3(*kra[g_][5:8]) for g_ in GR]],
          ['Gwet\'s AC1, four classes, "mixed" and "not sure" missing', *[d3(ac4[g_]) for g_ in GR]],
          ["Unanimous answers", *[y1(kra[g_][8]) for g_ in GR]],
          ["Consensus kept, changed and ambiguous", *[durum(g_) for g_ in GR]]]
assert [int(kv[g_][1]) for g_ in GR] == [150, 112, 38]
cs6 = dict(re.findall(r"'(\w+)': np\.float64\(([-\d.]+)\)", bul(r"category-specific kappa \(6 cat\): (\{.*\})", B)[1]))
cs4 = dict(re.findall(r"'(\w+)': np\.float64\(([-\d.]+)\)", bul(r"category-specific kappa \(4 cls\): (\{.*\})", B)[1]))
n6 = bul(r"Fleiss kappa, 6 categories, images without blank answers \(n=(\d+)\): ([\d.]+)", B)
n4 = bul(r"Fleiss kappa, 4 classes, images where all three gave a class \(n=(\d+)\): ([\d.]+)", B)
hs = bul(r"Fleiss kappa H vs SO, images where all three answered H or SO \(n=(\d+)\): ([\d.]+)", B)
cif = re.findall(r"^\s+expert(\d)-expert(\d): 6 cat kappa ([\d.]+), raw agreement ([\d.]+) \(n=(\d+)\) \| 4 cls kappa ([\d.]+), "
                 r"raw ([\d.]+) \(n=(\d+)\)", B, re.M)
assert len(cif) == 3
spa = dict(re.findall(r"specific agreement for (Healthy|Surface_Obstruction): ([\d.]+)", U))
sat12b = [[f"Experts {a_} and {b_}, Cohen's κ (raw agreement, images)", f"{d3(k6)} ({y1(r6)}, {n6_})", f"{d3(k4)} ({y1(r4)}, {n4_})"]
          for a_, b_, k6, r6, n6_, k4, r4, n4_ in cif]
sat12b += [[f"All three, Fleiss' κ (images)", f"{d3(n6[2])} ({n6[1]})", f"{d3(n4[2])} ({n4[1]})"]]
sat12b += [[f"Category-specific κ, {SN[k_]}", d3(cs6[k_]), d3(cs4[k_])]
           for k_ in ("Electrical_Fault", "Structural_Fault", "Surface_Obstruction", "Healthy")]
sat12b += [["Category-specific κ, mixed and not sure", f"{d3(cs6['mixed'])} and {d3(cs6['not_sure'])}", "–"],
           [f"Fleiss' κ for H versus SO only (images answered H or SO by all three)", "–", f"{d3(hs[2])} ({hs[1]})"],
           ["Specific agreement, H and SO", "–", f"{d3(spa['Healthy'])} and {d3(spa['Surface_Obstruction'])}"]]
ED = {u: ast.literal_eval(bul(rf"^expert{u} (\{{[^}}]*\}})", B)[1]) for u in (1, 2, 3)}
CEV = [("Healthy", "H"), ("Surface_Obstruction", "SO"), ("Electrical_Fault", "EF"), ("Structural_Fault", "SF"), ("mixed", "Mixed"),
       ("not_sure", "Not sure"), ("blank", "Unanswered")]
sat12c = [[ad_, *[str(ED[u].get(k_, 0)) for u in (1, 2, 3)]] for k_, ad_ in CEV]
for u in (1, 2, 3):
    assert sum(ED[u].values()) == 150
HC = {u: ast.literal_eval(bul(rf"^H controls expert{u} (\{{[^}}]*\}})", B)[1]) for u in (1, 2, 3)}
SOK = ast.literal_eval(bul(r"SO consensus labels: (\d+) (\{[^}]*\})", U)[2])
nso = bul(r"SO consensus labels: (\d+)", U)[1]
nhc = sum(HC[1].values())
sat12c += [[f"H answers on the {nhc} H controls", *[str(HC[u].get("Healthy", 0)) for u in (1, 2, 3)]],
           [f"SO answers behind the {nso} SO consensus labels", *[str(SOK[f"uzman{u}"]) for u in (1, 2, 3)]]]
loo = {(m_[0], m_[1]): m_[2:] for m_ in re.findall(r"drop uzman(\d), (controls|candidates): changed (\d+), kept (\d+), ambiguous (\d+)", U)}
sat12d = [["None (all three experts)",
           ", ".join(str(dur[(True, d_)]) for d_ in ("changed", "kept", "ambiguous")),
           ", ".join(str(dur[(False, d_)]) for d_ in ("changed", "kept", "ambiguous"))]]
sat12d += [[f"Expert {u}", ", ".join(loo[str(u), "candidates"]), ", ".join(loo[str(u), "controls"])] for u in (1, 2, 3)]
efsf = {k_: ast.literal_eval(bul(rf"^\s+{k_} expert2 (\{{[^}}]*\}})", B)[1]) for k_ in ("Electrical_Fault",)}
kon = {k_: ast.literal_eval(bul(rf"^\s+{k_} consensus (\{{[^}}]*\}})", B)[1]) for k_ in ("Electrical_Fault", "Structural_Fault")}
orj = ast.literal_eval(bul(r"EF/SF originals: (\{[^}]*\})", B)[1])
mx = bul(r"mixed answers total (\d+) \| on EF/SF originals (\d+) \| EF/SF images (\d+)", B)
kont = ast.literal_eval(bul(r"controls by original label: (\{[^}]*\})", B)[1])
# E3 ile uzman uzlaşısı: adaylar tam sayım, kontroller sınıfa göre aday olmayan 1.983 görüntüye ağırlıklı (a_uzman)
ua = bul(r"Non-ambiguous candidates (\d+) \| consensus == E3: (\d+)", U)
uk = bul(r"Controls: E3 == original (\d+) / (\d+) \| E3 == consensus (\d+) / (\d+) \| retrained E3 == consensus (\d+) / (\d+)", U)
unc = ast.literal_eval(bul(r"Non-candidate population by class: (\{[^}]*\})", U)[1])
ust = bul(r"Stratified E3 accuracy vs expert consensus \(non-ambiguous\): ([\d.]+) \[([\d.]+), ([\d.]+)\] \(estimated "
          r"non-ambiguous population (\d+)\)", U)
ustT = bul(r"Same for retrained E3 \(audited labels\): ([\d.]+)", U)
uch = bul(r"Check: same estimator with original labels = ([\d.]+) vs true full-data E3 accuracy ([\d.]+)", U)
uca = bul(r"Candidates: n (\d+) \| E3 == original label: (\d+)", U)
uam = bul(r"Ambiguous \(all 150\): (\d+) \| of which candidates (\d+) \| E3 errors among ambiguous: (\d+)", U)
assert uam[2] == uam[3] and int(uca[2]) == 0      # belirsiz E3 hatalarının hepsi aday
assert sum(unc.values()) + 112 == 2095
sat12e = [[f"Non-ambiguous candidates, fully enumerated", f"{ua[2]} of {ua[1]}", "–"],
          ["Non-ambiguous random controls", f"{uk[3]} of {uk[4]}", f"{uk[5]} of {uk[6]}"],
          ["Stratified estimate for all non-ambiguous images [95% CI]", pb(ust[1], ust[2], ust[3]), y1(ustT[1])]]
s += ["# Table S12. Agreement among the three experts", "",
      "*(a) Overall agreement on all images, the 112 model-selected candidates and the 38 random controls*", "",
      tablo(["Measure", "All", "Candidates", "Controls"], sat12a), "",
      "*(b) Pairwise and category-specific agreement*", "",
      tablo(["Measure", "Six answers", "Four classes only"], sat12b), "",
      "*(c) Answers of each expert on the 150 images*", "",
      tablo(["Answer", "Expert 1", "Expert 2", "Expert 3"], sat12c), "",
      "*(d) Consensus labels when one expert is left out (changed, kept and ambiguous)*", "",
      tablo(["Expert left out", "Candidates", "Controls"], sat12d), "",
      "Fleiss' κ for a variable number of answers keeps the two unanswered images of Expert 1. Category-specific κ was computed "
      "on the images with three answers, and specific agreement on all answers in the four classes. With two experts, the "
      f"consensus requires both to agree. Expert 2 labelled {efsf['Electrical_Fault']['Structural_Fault']} of the "
      f"{orj['Electrical_Fault']} audited EF images as SF, and the consensus confirmed {kon['Electrical_Fault']['Electrical_Fault']} "
      f"of the {orj['Electrical_Fault']} EF and {kon['Structural_Fault']['Structural_Fault']} of the {orj['Structural_Fault']} SF "
      f"originals, while {mx[2]} of the {mx[1]} \"mixed\" answers concerned these {mx[3]} images. The random controls contained "
      f"only {kont['Electrical_Fault']} EF and {kont['Structural_Fault']} SF image ({kont['Healthy']} H and "
      f"{kont['Surface_Obstruction']} SO), so the dataset-wide label-change rate describes mainly the H and SO classes.", "",
      "*(e) Agreement of E3 with the expert consensus on non-ambiguous images*", "",
      tablo(["Images", "E3, original labels", "E3 retrained on audited labels"], sat12e), "",
      "The stratified estimate enumerates the candidates fully and weights the controls by class to the "
      f"{n_(sum(unc.values()))} non-candidate images ({n_(unc['Healthy'])} H, {unc['Surface_Obstruction']} SO, "
      f"{unc['Electrical_Fault']} EF and {unc['Structural_Fault']} SF), which corresponds to an estimated {n_(ust[4])} "
      "non-ambiguous images. Its CI comes from 2,000 bootstrap resamples of the controls within each class. As a check, the same "
      f"estimator with the original labels as the reference gives {y1(uch[1])}, against the actual E3 accuracy of {y1(uch[2])}. "
      f"E3 disagreed with the original label on all {uca[1]} candidates, and {uam[3]} of the {uam[1]} ambiguous images were E3 "
      "errors, all of them candidates.", ""]

# ================================================================== S13 sınıf bazında LTT ve LAC (r1_a, r1_h, b_secici, secici_ek)
LT = {}
for et in ("orij", "orijbel", "temiz"):
    blok = A.split(f"[{et}]")[1].split("\n[")[0]
    LT[et] = dict(
        kab=dict(re.findall(r"(EF|H|SF|SO) ([\d.]+)", bul(r"mean acceptance rate by true class \(all splits\): (.*?) \|", blok)[1])),
        fkab=bul(r"mean acceptance rate by true class \(all splits\): .*? \| faults \(EF\+SF\) ([\d.]+)", blok)[1],
        hata={k_: int(a_) / int(b_) for k_, a_, b_ in re.findall(r"(EF|H|SF|SO) (\d+)/(\d+)", bul(r"errors among accepted by true "
                                                                                                 r"class, total over splits / accepted: (.*)", blok)[1])},
        lac=bul(r"LAC \(eps 0\.05\) -> action: no action \{H\} ([\d.]+), cleaning check \(H/SO sets\) ([\d.]+), priority inspection "
                r"\(set has EF or SF\) ([\d.]+), empty ([\d.]+); true faults flagged ([\d.]+); non-faults flagged ([\d.]+); coverage "
                r"([\d.]+)", blok),
        d05=bul(r"delta=0\.05: coverage ([\d.]+) \+- ([\d.]+) acc ([\d.]+) violation ([\d.]+) zero-coverage ([\d.]+)", blok),
        d01=bul(r"delta=0\.01: coverage ([\d.]+) \+- ([\d.]+) acc ([\d.]+) violation ([\d.]+) zero-coverage ([\d.]+)", blok),
        ana=bul(rf"^\[{et}\] n=(\d+) coverage ([\d.]+) \+- ([\d.]+) acc ([\d.]+) violation ([\d.]+) zero-coverage splits ([\d.]+)", A),
        h=bul(rf"^\[{et}\] per 1,000 test images: reviewed (\d+); faults accepted automatically ([\d.]+); accepted faults labelled "
              rf"no-fault ([\d.]+); non-faults accepted as faults ([\d.]+) \| mean test faults per split ([\d.]+)", H))
    assert abs(LT[et]["hata"]["EF"] - f(bul(rf"^\[{et}\].*error among accepted by class EF ([\d.]+)", H)[1])) < 6e-4
ET3 = ("orij", "orijbel", "temiz")


def kz(x, sd, z):
    return f"{yuzde(f(x), f(sd))} ({f(z) * 100:.1f}%)"


sat13a = [["Coverage at δ = 0.1 (splits without acceptance)", *[kz(LT[e]["ana"][2], LT[e]["ana"][3], LT[e]["ana"][6]) for e in ET3]],
          *[[f"Acceptance, {k_}", *[y1(LT[e]["kab"][k_]) for e in ET3]] for k_ in ("H", "SO", "EF", "SF")],
          ["Acceptance, faults (EF or SF)", *[y1(LT[e]["fkab"]) for e in ET3]],
          *[[f"Error among accepted, {k_}", *[y1(LT[e]["hata"][k_]) for e in ET3]] for k_ in ("H", "SO", "EF", "SF")],
          ["Per 1,000 images, reviewed manually", *[LT[e]["h"][1] for e in ET3]],
          ["Per 1,000 images, faults (mean)", *[LT[e]["h"][5] for e in ET3]],
          ["Per 1,000 images, faults accepted automatically", *[LT[e]["h"][2] for e in ET3]],
          ["Per 1,000 images, accepted faults labelled as no fault", *[LT[e]["h"][3] for e in ET3]],
          ["Per 1,000 images, non-faults accepted as faults", *[LT[e]["h"][4] for e in ET3]],
          ["Coverage at δ = 0.05 (splits without acceptance)", *[kz(LT[e]["d05"][1], LT[e]["d05"][2], LT[e]["d05"][5]) for e in ET3]],
          ["Violation rate at δ = 0.05", *[d3(LT[e]["d05"][4]) for e in ET3]],
          ["Coverage at δ = 0.01 (splits without acceptance)", *[kz(LT[e]["d01"][1], LT[e]["d01"][2], LT[e]["d01"][5]) for e in ET3]],
          ["Violation rate at δ = 0.01", *[d3(LT[e]["d01"][4]) for e in ET3]],
          ["LAC sets, share with no action, cleaning check and priority inspection",
           *[", ".join(y1(LT[e]["lac"][i]) for i in (1, 2, 3)) for e in ET3]],
          ["Priority inspection, share of faults and of other images", *[f"{y1(LT[e]['lac'][5])}, {y1(LT[e]['lac'][6])}" for e in ET3]]]
for e in ET3:
    assert int(LT[e]["ana"][1]) == {"orij": 2095, "orijbel": 2049, "temiz": 2049}[e]
LC = {m_[0]: m_ for m_ in re.findall(r"^\s+(\w+)\s+LAC cov ([\d.]+) size ([\d.]+) \| Mondrian cov ([\d.]+) \| LTT accepted share ([\d.]+)", BS, re.M)}
mon = bul(r"Mondrian LAC: marginal cov ([\d.]+), mean set size ([\d.]+)", BS)
lacb = bul(r"LAC marginal set size ([\d.]+)", BS)
sat13b = [[SN[k_], d3(LC[k_][1]), f"{f(LC[k_][2]):.2f}", d3(LC[k_][3]), y1(LC[k_][4])]
          for k_ in ("Healthy", "Surface_Obstruction", "Electrical_Fault", "Structural_Fault")]
sat13b.append(["All", d3(LT["orij"]["lac"][7]), d3(lacb[1]), f"{d3(mon[1])} (mean set size {d3(mon[2])})", y1(LT["orij"]["ana"][2])])
af = bul(r"Accepted fault \(EF\+SF\) test images per split ([\d.]+); of which predicted H/SO ([\d.]+) \(missed-fault rate among "
         r"accepted fault images ([\d.]+)\); among images auto-labelled non-fault, share truly faulty ([\d.]+)", BS)
s += ["# Table S13. Class-wise behaviour of the LTT selective classifier and of the LAC sets", "",
      "*(a) LTT (α = 5%) and LAC sets (ε = 0.05) of E3 by true class and label condition, means over the 1,000 scene-level "
      "calibration and test splits*", "",
      tablo(["Quantity", "Original labels (2,095)", "Ambiguous removed (2,049)", "Audited labels, E3 retrained (2,049)"], sat13a), "",
      "*(b) Class-conditional coverage of LAC and Mondrian LAC sets and LTT acceptance (original labels, δ = 0.1)*", "",
      tablo(["Class", "LAC coverage", "LAC mean set size", "Mondrian LAC coverage", "LTT acceptance"], sat13b), "",
      "The LTT guarantee holds for the class mix as a whole, so acceptance differs by class, and the review share of a plant with "
      r"class shares $\pi_c$ follows from $\sum_c \pi_c (1 - a_c)$ with the class-wise acceptance rates $a_c$ in (a). The LAC "
      "sets were read as actions, with a singleton H set meaning no action, a set of H and SO a cleaning check and any set "
      f"containing EF or SF a priority inspection. With the original labels, on average {f(af[1]):.1f} fault images per split "
      f"were accepted, {f(af[2]):.2f} of them with a no-fault label ({y1(af[3])}), and {f(af[4]) * 100:.2f}% of the images "
      "automatically labelled as no fault carried a fault label.", ""]

# ================================================================== S14 kontrol listesi
kg = bul(r"Dış set (\d+); örtüşen (\d+)", KG)
dc = bul(r"Etiket gürültüsü \(değişen\): .*?tüm veri tahmini ([\d.]+) \[([\d.]+)-([\d.]+)\]", DS)
ece = EKS["ece"]
acc4 = [f(FN[m_][9]) for m_ in MOD[:3]]
cov5 = f(bul(r"kapsam ([\d.]+) \(±", open("asama2_sonuclar/secici_ek/orij_r0.05.txt", encoding="utf-8").read())[1])
oz = OZ["dosya"] - OZ["md5_tekil"]
# az örnek: oturum/alt koleksiyon gruplu yeni deney (28b, ana metin Şekil 9 ve Bulgular); Tablo S2'yi yazar değiştirecek
azg = pd.read_csv("asama2_sonuclar/az_ornek_gruplu.csv")
azs = azg[azg.yontem == "ince_ayar_solarfcd"].groupby("butce").agg(n=("n_egitim", "mean"), acc=("acc", "mean"))
az_taban = azg.groupby("tekrar").test_cogunluk.first().mean()
assert azg.tekrar.nunique() == 10 and list(azs.index) == [0, 55, 111, 222]
s4l, s4u = wilson(5, 160)
sat14 = [["1. Byte-level and perceptual duplicate rates, within and across datasets",
          "MD5, dihedral pHash and geometric matching, screening of external sets",
          f"{oz / OZ['dosya'] * 100:.1f}% redundant files, {160 / 375 * 100:.1f}% of test images with a byte-identical training "
          f"copy, {int(kg[2]) / int(kg[1]) * 100:.1f}% of the Kaggle set overlapping"],
         ["2. Grouped evaluation with audited residual leakage",
          "scene- or site-grouped CV with the split made before augmentation, blinded audit of random queries",
          f"scene-grouped five-fold CV, residual same-scene leakage {5 / 160 * 100:.1f}% (95% CI {s4l * 100:.1f}–{s4u * 100:.1f}%)"],
         ["3. Label definitions and inter-annotator agreement",
          "written definitions with severity thresholds aligned with failure taxonomies, blinded relabelling of a random sample",
          f"no severity scale, Fleiss' κ {d2(kv['all'][2])}, label-change rate {y1(dc[1])} (95% CI {f(dc[2]) * 100:.1f}–"
          f"{y1(dc[3])})"],
         ["4. Calibration and selective accuracy with a stated guarantee, by class",
          "ECE, certified coverage with α and δ, class-wise acceptance",
          f"ECE {ece['SolarFCD OOF']:.3f}, coverage {y1(cov5)} at α = 5% and δ = 0.1, {y1(LT['orij']['fkab'])} of faults accepted "
          "automatically"],
         ["5. External tests screened for overlap", "MD5, pHash and embedding similarity, clustering by source image",
          f"{int(rf[2]) / int(rf[1]) * 100:.1f}% (RF-Fault) and {int(rs[2]) / int(rs[1]) * 100:.1f}% (RF-Soiling) removed, "
          f"accuracy {min(acc4):.0f}–{max(acc4):.0f}%"],
         ["6. Target labels needed for new data", "learning curve with adaptation and test images from different sessions",
          f"about {azs.n[111]:.0f} labelled images raised the accuracy on held-out RF-Fault sessions only from "
          f"{y1(azs.acc[0])} to {y1(azs.acc[111])}, against a majority baseline of {y1(az_taban)} "
          "(Fig. 12 of the main text)"],
         ["7. Acquisition metadata and unit of analysis",
          "platform, viewing geometry, native resolution or ground sampling distance, irradiance, module technology and site "
          "identity, analysis at image, module or string level",
          f"web photographs without site identity or acquisition data, median {n_(RZ['all'][2])} × {n_(RZ['all'][3])} px (EF "
          f"{RZ['Electrical_Fault'][2]} × {RZ['Electrical_Fault'][3]} px), unit is the photograph"],
         ["8. Physical or electrical ground truth", "soiling ratio (IEC 61724-1), I–V, IR or EL confirmation",
          "none, visual labels only"],
         ["9. Metrics at realistic prevalence", "precision and recall at the expected fault prevalence",
          f"fault recall {d2(fa[6])}, precision {d2(fa[7])} at {y1(1 - f(fa[2]))} and {d2(prv['1'])} at 1% prevalence"]]
s += ["# Table S14. Reporting checklist for PV image benchmarks with the values found for SolarFCD", "",
      "Items 1–6 are the recommendations of the main text, and items 7–9 are further items needed for O&M use.", "",
      tablo(["What to report", "How", "SolarFCD in this study"], sat14), ""]

# ================================================================== Şekiller
ece_ci = {k_: bul(rf"^ECE {k_} ([\d.]+) \[(?:cluster|i\.i\.d\.) bootstrap ([\d.]+), ([\d.]+)\]", C) for k_ in ("in-domain", "RF-Fault", "RF-Soiling")}
eg = re.findall(r"^(\w+)\s+15\s+30\.0\s+([\d.]+)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+([\d.]+)$", G, re.M)
assert len(eg) == 8
eg_acc = [f(x[2]) for x in eg if x[0].endswith("224") and not x[0].startswith("resnet")]   # şekil: 224 px koşuları
assert len(eg_acc) == 5 and f"{min(eg_acc) * 100:.1f}–{max(eg_acc) * 100:.1f}" == "99.3–99.8"
assert max(abs(f(x[3]) - f(x[4])) for x in eg) < 0.01
s += ["# Supplementary figures", "",
      "![**Fig. S1.** Reliability diagrams of the ensemble E3 in-domain (SolarFCD out-of-fold predictions) and on the two external "
      "datasets, with the expected calibration error (ECE) over ten equal-width bins and histograms of the number of images per "
      f"confidence bin. The ECE (95% CI) was {ece_ci['in-domain'][1]} ({ece_ci['in-domain'][2]}–{ece_ci['in-domain'][3]}) "
      f"in-domain with a scene-level cluster bootstrap, and {ece_ci['RF-Fault'][1]} ({ece_ci['RF-Fault'][2]}–"
      f"{ece_ci['RF-Fault'][3]}) on RF-Fault and {ece_ci['RF-Soiling'][1]} ({ece_ci['RF-Soiling'][2]}–{ece_ci['RF-Soiling'][3]}) "
      "on RF-Soiling with an image-level bootstrap, each with 2,000 resamples. Because training used label smoothing of 0.1, the "
      f"highest in-domain E3 confidence was {gv[1]}.](fig08_reliability.png){{width=15cm}}", "",
      "![**Fig. S2.** Overlap screening of the external datasets against all 3,742 SolarFCD images, showing the minimum pHash distance versus the maximum DINOv2 cosine similarity. Dashed lines mark the removal thresholds (pHash ≤ 12, cosine ≥ 0.85), and red points are byte-identical files.](fig09_overlap_screening.png){width=15cm}", "",
      "![**Fig. S3.** Training accuracy and loss under P3 (mean of the 15 runs at 224 px per backbone, loss on a logarithmic "
      "scale, insets for epochs 21–30 without ResNetV2-101). All backbones except ResNetV2-101 reached 99.3–99.8% training "
      "accuracy by epoch 30, and the training loss changed by less than 0.01 over the last five epochs.]"
      "(figS3_training_curves.png){width=15cm}", "",
      '![**Fig. S4.** Answers of each expert on the 150 audit images, with the counts listed in Table S12(c).](figS4_expert_labels.png){width=9cm}', ""]

# ================================================================== Ek kaynakça (makale/kaynakca.py HAND, makale/ek_kaynak_meta.json)
HAND = next(ast.literal_eval(n.value) for n in ast.parse(oku("makale/kaynakca.py")).body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "HAND")
EM = json.load(open("makale/ek_kaynak_meta.json", encoding="utf-8"))


def ini(given):                 # makale/kaynakca.py ile aynı kural
    out = []
    for p in re.split(r"\s+", given.replace(".", ". ").strip()):
        if p:
            out.append("-".join(x[0] + "." for x in p.split("-") if x))
    return "".join(out)


BASLIK = {"hayes2007": "Answering the call for a standard reliability measure for coding data",
          "lipton2018": "Detecting and correcting for label shift with black box predictors"}
DERGI = {"Communication Methods and Measures": "Commun. Methods Meas."}


def kunye(k):
    if k in HAND:
        return HAND[k]
    if k == "hayes2007":
        m_ = EM[k]
        assert BASLIK[k].lower() == m_["title"][0].lower()
        au = ", ".join(f"{ini(a_['given'])} {a_['family']}" for a_ in m_["author"])
        return (f"{au}, {BASLIK[k]}, {DERGI[m_['container-title'][0]]} {m_['volume']} ({m_['published-print']['date-parts'][0][0]}) "
                f"{m_['page'].replace('-', '–')}. https://doi.org/{m_['DOI']}.")
    if k == "lipton2018":
        m_ = EM[k]
        cm = m_["citation_meta"]
        assert BASLIK[k].lower() == cm["citation_title"][0].lower() == m_["datacite"]["titles"][0]["title"].lower()
        assert [c_["familyName"] for c_ in m_["datacite"]["creators"]] == [a_.split()[-1] for a_ in cm["citation_author"]]
        au = ", ".join(f"{ini(c_['givenName'])} {c_['familyName']}" for c_ in m_["datacite"]["creators"])
        no = bul(r"Proceedings of the (\d+(?:st|nd|rd|th)) International Conference on Machine Learning", m_["booktitle"])[1]
        assert m_["volume"] == m_["url_volume"]
        return (f"{au}, {BASLIK[k]}, in: Proc. {no} Int. Conf. Mach. Learn., PMLR {m_['volume']}, "
                f"{cm['citation_publication_date'][0][:4]}, pp. {cm['citation_firstpage'][0]}–{cm['citation_lastpage'][0]}.")
    raise KeyError(k)


s += ["# Supplementary references {.unnumbered}", ""]
for i, k in enumerate(SIRA, 1):
    s += [f"[S{i}] {kunye(k)}", ""]
open("makale/supplementary.md", "w", encoding="utf-8").write("\n".join(s).rstrip() + "\n")
print("ok", len(s), "satır; ek kaynaklar:", SIRA)
