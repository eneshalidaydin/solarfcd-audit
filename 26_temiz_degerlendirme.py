"""Adım 26 - Temiz etiket değerlendirmesi (önceden sabit plan, 21 Kasım). Değerlendirme kümesi: uzman denetimi sonrası
belirsiz olmayan 2.049 görüntü (asama2_temiz). Karşılaştırmalar (havuzlanmış OOF, 3 tohum olasılık ortalaması):
(a) temiz etiketle eğitilen modeller, temiz etiketle; (b) orijinal etiketle eğitilen modeller, temiz etiketle; (a-b) eşli
sahne_grup bootstrap (2.000); (c) orijinal modeller orijinal etiketle: tüm 2.095 ve belirsizler çıkarılmış 2.049 (ikincil
"kararsızlar hariç" analiz). Uyarı: adaylar model-etiket uyuşmazlığıyla seçildiğinden temiz etikette değerlendirme
iyimser yanlı olabilir; kontrol grubu taban değişim oranını verir. Çıktı: asama2_sonuclar/temiz.txt, temiz_etiketler.csv"""
import numpy as np
import pandas as pd

SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
PC = [f"p_{c}" for c in SIN]
M = pd.read_csv("asama2_temiz/manifest_asama2.csv")
M = M[M.label_orijinal.notna()].set_index("path")
pd.DataFrame({"path": M.index, "etiket": np.where(M.kat_v1 >= 0, M.label, "belirsiz")}).to_csv(
    "asama2_sonuclar/temiz_etiketler.csv", index=False)
UY = {"ConvNeXt": "convnext_tiny_224", "Swin": "swin_tiny_patch4_window7_224_224", "DINOv2-392": "vit_base_patch14_dinov2_392"}


def oof(kok, u, son):
    return pd.concat([pd.read_csv(f"{kok}/grouped{k}_{u}_s{s}_{son}/test_predictions.csv") for s in (42, 43, 44)
                      for k in range(5)]).groupby("path")[PC].mean()


T = {a: oof("runs_temiz", u, "kat_v1_asama2_temiz_none") for a, u in UY.items()}
O = {a: oof("runs_asama2", u, "kat_v1_asama2_none") for a, u in UY.items()}
yol = T["ConvNeXt"].index
T = {a: d.loc[yol].to_numpy() for a, d in T.items()}
O2 = {a: d.loc[yol].to_numpy() for a, d in O.items()}
T["Topluluk-3"] = np.mean([T[a] for a in UY], 0)
O2["Topluluk-3"] = np.mean([O2[a] for a in UY], 0)
yc = M.loc[yol, "label"].map(SIN.index).to_numpy()
yo = M.loc[yol, "label_orijinal"].map(SIN.index).to_numpy()
g = M.loc[yol, "sahne_grup"].to_numpy()
gu, gi = np.unique(g, return_inverse=True)
rng = np.random.default_rng(0)
B = [np.bincount(rng.choice(len(gu), len(gu)), minlength=len(gu))[gi] for _ in range(2000)]


def f1(y, p, w=None):
    w = np.ones(len(y)) if w is None else w
    c = np.zeros((4, 4))
    np.add.at(c, (y, p), w)
    den = c.sum(0) + c.sum(1)
    return np.mean(np.where(den > 0, 2 * np.diag(c) / np.maximum(den, 1e-12), 0))


def ms(y, p, w=None):
    w = np.ones(len(y)) if w is None else w
    return np.array([np.average(p == y, weights=w), f1(y, p, w)])


s = [f"Değerlendirme: {len(yol)} görüntü (temiz), değişen etiket {(yc != yo).sum()}", "",
     "Model | (a) temiz model/temiz etiket acc F1 | (b) orijinal model/temiz etiket acc F1 | a-b acc [%95 GA] F1 [%95 GA] "
     "| (c) orijinal model/orijinal etiket 2.049 acc F1"]
for a in T:
    pa, pb = T[a].argmax(1), O2[a].argmax(1)
    ra, rb, rc = ms(yc, pa), ms(yc, pb), ms(yo, pb)
    d = np.array([ms(yc, pa, w) - ms(yc, pb, w) for w in B])
    lo, hi = np.percentile(d, [2.5, 97.5], 0)
    s.append(f"{a:11s} | {ra[0]:.4f} {ra[1]:.4f} | {rb[0]:.4f} {rb[1]:.4f} | {ra[0]-rb[0]:+.4f} [{lo[0]:+.4f},{hi[0]:+.4f}] "
             f"{ra[1]-rb[1]:+.4f} [{lo[1]:+.4f},{hi[1]:+.4f}] | {rc[0]:.4f} {rc[1]:.4f}")
    s.append("            sınıf duyarlılığı (a): " + " ".join(f"{c[:2]} {(pa[yc == i] == i).mean():.3f}" for i, c in enumerate(SIN)))
# (c) tüm 2.095, orijinal etiket
yol_t = O["ConvNeXt"].index
yt = M.loc[yol_t, "label_orijinal"].map(SIN.index).to_numpy()
Pt = np.mean([O[a].loc[yol_t].to_numpy() for a in UY], 0).argmax(1)
s += ["", f"Topluluk-3 orijinal model/orijinal etiket, tüm {len(yol_t)}: acc {(Pt == yt).mean():.4f} F1 {f1(yt, Pt):.4f}"]
A = pd.read_csv("insan_kontrol/etiket/sonuc.csv", encoding="utf-8-sig")
s.append(f"Yanlılık kontrolü: kontrol grubunda etiket değişim oranı {(A[~A.aday].durum == 'degisti').mean():.3f}, "
         f"adaylarda {(A[A.aday].durum == 'degisti').mean():.3f}")
open("asama2_sonuclar/temiz.txt", "w", encoding="utf-8").write("\n".join(s))
print("\n".join(s))
