"""Adım 21 - Yüksek çözünürlük (19) sonuçları: 3 tohum havuzlanmış OOF, 224 px karşılığına göre eşleştirilmiş küme
bootstrap farkı (sahne_grup, 2.000), ayrıca topluluk (ConvNeXt-384 + DINOv2-392 + Swin-224) ve seçici sınıflandırma
önizlemesi. Çıktı: asama2_sonuclar/cozunurluk.txt"""
from pathlib import Path

import numpy as np
import pandas as pd

SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
PC = [f"p_{c}" for c in SIN]
m = pd.read_csv("asama2/manifest_asama2.csv")
rng = np.random.default_rng(0)


def oku(ad, s):
    return pd.concat([pd.read_csv(f"runs_asama2/grouped{k}_{ad}_s{s}_kat_v1_asama2_none/test_predictions.csv")
                      for k in range(5)]).set_index("path").sort_index()


def f1(y, p, w=None):
    w = np.ones(len(y)) if w is None else w
    c = np.zeros((4, 4))
    np.add.at(c, (y, p), w)
    den = c.sum(0) + c.sum(1)
    return np.mean(np.where(den > 0, 2 * np.diag(c) / np.maximum(den, 1e-12), 0))


AD = {"ConvNeXt-224": "convnext_tiny_224", "ConvNeXt-384": "convnext_tiny_384",
      "DINOv2-224": "vit_base_patch14_dinov2_224", "DINOv2-392": "vit_base_patch14_dinov2_392",
      "Swin-224": "swin_tiny_patch4_window7_224_224"}
pr = {k: {s: oku(v, s) for s in (42, 43, 44)} for k, v in AD.items()}
ref = pr["ConvNeXt-224"][42]
y = ref["label"].map(SIN.index).to_numpy()
olas = {k: {s: d.loc[ref.index, PC].to_numpy() for s, d in v.items()} for k, v in pr.items()}
for ad, uye in [("Topluluk-3 224", ["ConvNeXt-224", "Swin-224", "DINOv2-224"]),
                ("Topluluk-3 yuksek", ["ConvNeXt-384", "Swin-224", "DINOv2-392"])]:
    olas[ad] = {s: np.mean([olas[u][s] for u in uye], 0) for s in (42, 43, 44)}
g = ref.index.map(m.set_index("path")["sahne_grup"]).to_numpy()
gu, gi = np.unique(g, return_inverse=True)
W = [np.bincount(rng.choice(len(gu), len(gu)), minlength=len(gu))[gi] for _ in range(2000)]
s = []
bs = {}
for k, d in olas.items():
    p = {t: v.argmax(1) for t, v in d.items()}
    bs[k] = np.mean([[f1(y, q, w) for w in W] for q in p.values()], 0)
    rec = np.mean([[(q[y == i] == i).mean() for i in range(4)] for q in p.values()], 0)
    s.append(f"{k:18s} acc {np.mean([(q == y).mean() for q in p.values()]):.4f}  F1 "
             f"{np.mean([f1(y, q) for q in p.values()]):.4f}  EF {rec[0]:.3f} H {rec[1]:.3f} SF {rec[2]:.3f} SO {rec[3]:.3f}")
s.append("")
for a, b in [("ConvNeXt-384", "ConvNeXt-224"), ("DINOv2-392", "DINOv2-224"), ("Topluluk-3 yuksek", "Topluluk-3 224"),
             ("Topluluk-3 yuksek", "ConvNeXt-224")]:
    fark = bs[a] - bs[b]
    lo, hi = np.percentile(fark, [2.5, 97.5])
    s.append(f"{a} - {b}: F1 farkı {fark.mean():+.4f} [{lo:+.3f}, {hi:+.3f}]{' *' if lo > 0 or hi < 0 else ''}")
P = np.mean(list(olas["Topluluk-3 yuksek"].values()), 0)
o = np.argsort(-P.max(1))
s.append("\nTopluluk-3 yüksek (9 üye ort.) seçici: " + ", ".join(
    f"%{int(c * 100)} kapsam acc {(P.argmax(1)[o[:int(len(y) * c)]] == y[o[:int(len(y) * c)]]).mean():.3f}"
    for c in (1, .9, .85, .8)))
Path("asama2_sonuclar/cozunurluk.txt").write_text("\n".join(s), encoding="utf-8")
print("\n".join(s))
