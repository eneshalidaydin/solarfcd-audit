"""Adım 15 - Ek analizler (kullanıcı kararı 20 Kasım), yalnız mevcut tahminlerden (yeni eğitim yok).
  1) Modeller arası eşleştirilmiş küme bootstrap (sahne_grup, 2.000 tekrar): makro-F1 farkı (3 tohum ortalaması).
  2) Topluluk: ConvNeXt+Swin+DINOv2 olasılık ortalaması (aynı tohum), ayrıca 5 modelin tümü.
  3) EF/SF duyarlılığı: insan kontrolünde tartışmalı 4 görüntü (a) çıkarılır, (b) uzman etiketiyle değerlendirilir.
     Eğitim etiketleri değişmez; yalnız değerlendirme duyarlılığı.
  4) Karışıklık matrisleri (3 tohum toplamı).
Ayrıca runs_rastgele/ varsa sızıntılı (rastgele bölme) ve sızıntısız sonuç aynı görüntülerde karşılaştırılır (tohum 42).
Çıktı: asama2_sonuclar/ek_analiz.txt"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
PC = [f"p_{c}" for c in SIN]
m = pd.read_csv("asama2/manifest_asama2.csv")
sg = m.set_index("path")["sahne_grup"]
rng = np.random.default_rng(0)
KIS = {"convnext_tiny": "ConvNeXt", "swin_tiny_patch4_window7": "Swin", "vit_base_patch14_dinov2": "DINOv2",
       "tf_efficientnetv2_s": "EffNetV2", "resnetv2_101": "ResNetV2",
       "convnextv2_tiny": "ConvNeXtV2"}


def oku(kok, desen):
    P = {}
    for d in sorted(Path(kok).glob(desen)):
        mdl = KIS[d.name.split("_224_")[0][len("grouped0_"):]]
        tohum = int(d.name.split("_s")[-1].split("_")[0])
        if not all((Path(kok) / d.name.replace("grouped0_", f"grouped{k}_") / "test_predictions.csv").exists()
                   for k in range(5)):
            continue
        D = pd.concat([pd.read_csv(Path(kok) / d.name.replace("grouped0_", f"grouped{k}_") / "test_predictions.csv")
                       for k in range(5)]).set_index("path").sort_index()
        P[(mdl, tohum)] = D
    return P


def f1(y, p, w=None):
    """y, p: sınıf indeksleri; w: örnek ağırlığı (bootstrap sayımı)."""
    w = np.ones(len(y)) if w is None else w
    c = np.zeros((4, 4))
    np.add.at(c, (y, p), w)
    tp = np.diag(c)
    den = c.sum(0) + c.sum(1)
    return np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0))


P = oku("runs_asama2", "grouped0_*_none")
yol = next(iter(P.values())).index
y = next(iter(P.values()))["label"].map(SIN.index).to_numpy()
tah = {}
for (mdl, t), D in P.items():
    assert (D.index == yol).all()
    tah.setdefault(mdl, {})[t] = D[PC].to_numpy()
for ad, uye in [("Topluluk-3 (ConvNeXt+Swin+DINOv2)", ["ConvNeXt", "Swin", "DINOv2"]),
                ("Topluluk-5", ["ConvNeXt", "Swin", "DINOv2", "EffNetV2", "ResNetV2"])]:
    tah[ad] = {t: np.mean([tah[u][t] for u in uye], axis=0) for t in (42, 43, 44)}
pred = {k: {t: v.argmax(1) for t, v in d.items()} for k, d in tah.items()}
g = yol.map(sg).to_numpy()
gu = np.unique(g)
gi = np.searchsorted(gu, g)
W = np.stack([np.bincount(rng.choice(len(gu), len(gu)), minlength=len(gu))[gi] for _ in range(2000)])
s = ["== Makro-F1 (3 tohum ort.) [%95 küme bootstrap GA] ve sınıf duyarlılıkları ==", ""]
bs = {}
for k, d in pred.items():
    bs[k] = np.mean([[f1(y, p, w) for w in W] for p in d.values()], axis=0)
    nok = np.mean([f1(y, p) for p in d.values()])
    acc = np.mean([(p == y).mean() for p in d.values()])
    rec = np.mean([[(p[y == i] == i).mean() for i in range(4)] for p in d.values()], axis=0)
    lo, hi = np.percentile(bs[k], [2.5, 97.5])
    s.append(f"{k:36s} acc {acc:.4f}  F1 {nok:.4f} [{lo:.3f}-{hi:.3f}]  EF {rec[0]:.3f} H {rec[1]:.3f} "
             f"SF {rec[2]:.3f} SO {rec[3]:.3f}")
s += ["", "== Eşleştirilmiş fark (satır - sütun), makro-F1, %95 GA; * = GA 0'ı içermiyor ==", ""]
for a, b in combinations(bs, 2):
    fark = bs[a] - bs[b]
    lo, hi = np.percentile(fark, [2.5, 97.5])
    nok = np.mean([f1(y, p) for p in pred[a].values()]) - np.mean([f1(y, p) for p in pred[b].values()])
    s.append(f"{a} - {b}: {nok:+.4f} [{lo:+.3f}, {hi:+.3f}]{' *' if lo > 0 or hi < 0 else ''}")
# EF/SF duyarlılığı
E = pd.read_csv("insan_kontrol/sonuc/efsf_karsilastirma.csv", encoding="utf-8-sig")
E = E[E["uyum"] == "FARKLI"]
yolu = m.set_index("rel_path")["path"]
ix = {yol.get_loc(yolu[r]): u for r, u in zip(E["rel"], E["uzman"]) if yolu[r] in yol}
s += ["", f"== EF/SF duyarlılığı: tartışmalı {len(E)} görüntünün {len(ix)}'ü değerlendirme setinde ==",
      "model: asıl / (a) çıkarılınca / (b) uzman etiketiyle (makro-F1, 3 tohum ort.)", ""]
maske = np.ones(len(y), bool)
maske[list(ix)] = False
y2 = y.copy()
for i, u in ix.items():
    y2[i] = SIN.index(u)
for k, d in pred.items():
    a0 = np.mean([f1(y, p) for p in d.values()])
    a1 = np.mean([f1(y[maske], p[maske]) for p in d.values()])
    a2 = np.mean([f1(y2, p) for p in d.values()])
    s.append(f"{k:36s} {a0:.4f} / {a1:.4f} / {a2:.4f}")
s += ["", "== Karışıklık matrisleri (satır gerçek, sütun tahmin; EF H SF SO; 3 tohum toplamı) ==", ""]
for k, d in pred.items():
    c = sum(pd.crosstab(pd.Categorical(y, range(4)), pd.Categorical(p, range(4)), dropna=False).to_numpy()
            for p in d.values())
    s.append(k + "\n" + "\n".join("  " + " ".join(f"{v:5d}" for v in r) for r in c))
R = oku("runs_rastgele", "grouped0_*_none") if Path("runs_rastgele").exists() else {}
tam = {k: v for k, v in R.items() if len(v) == len(yol)}
if tam:
    s += ["", "== Sızıntılı (rastgele bölme) vs sızıntısız (sahne grubu), aynı 2.095 görüntü, tohum 42 ==", ""]
    for (mdl, t), D in sorted(tam.items()):
        pr = D.loc[yol, PC].to_numpy().argmax(1)
        pg = pred[mdl][42]
        s.append(f"{mdl:10s} acc {(pr == y).mean():.4f} vs {(pg == y).mean():.4f}   F1 {f1(y, pr):.4f} vs "
                 f"{f1(y, pg):.4f}   fark F1 {f1(y, pr) - f1(y, pg):+.4f}")
(Path("asama2_sonuclar") / "ek_analiz.txt").write_text("\n".join(s), encoding="utf-8")
print("\n".join(s))
L = oku("runs_lr", "grouped0_*_none_lrsec") if Path("runs_lr").exists() else {}
if L:
    s += ["", "== Lr ayarı (iç doğrulamayla seçilen lr, iç doğrulamasız, tohum 42) vs temel lr (tohum 42) ==", ""]
    for (mdl, t), D in sorted(L.items()):
        pl = D.loc[yol, PC].to_numpy().argmax(1)
        pg = pred[mdl][42]
        fark = np.array([f1(y, pl, w) - f1(y, pg, w) for w in W])
        lo, hi = np.percentile(fark, [2.5, 97.5])
        s.append(f"{mdl:10s} F1 {f1(y, pl):.4f} vs {f1(y, pg):.4f}  fark {f1(y, pl) - f1(y, pg):+.4f} "
                 f"[{lo:+.3f}, {hi:+.3f}]  acc {(pl == y).mean():.4f}  SF {(pl[y == 2] == 2).mean():.3f}")
    (Path("asama2_sonuclar") / "ek_analiz.txt").write_text("\n".join(s), encoding="utf-8")
    print("\n".join(s[-len(L) - 2:]))
