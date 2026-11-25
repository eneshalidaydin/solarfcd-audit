"""İç değerlendirme R1 kontrolü - A4: P1 (resmi bolme) ayristirma. Yalniz okur."""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
PIL = Path(r"C:\solarfcd\pilot")
SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
AB = ["EF", "H", "SF", "SO"]
L = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    L.append(s)


M = pd.read_csv(PIL / "asama2" / "manifest_asama2.csv").set_index("path")
P1 = {"convnext_tiny": "official_convnext_tiny_224_s42", "swin_tiny": "official_swin_tiny_patch4_window7_224_224_s42",
      "dinov2": "official_vit_base_patch14_dinov2_224_s42", "efficientnetv2_s": "official_tf_efficientnetv2_s_224_s42",
      "resnetv2_101": "official_resnetv2_101_224_s42"}
P3 = {"convnext_tiny": "convnext_tiny_224", "swin_tiny": "swin_tiny_patch4_window7_224_224",
      "dinov2": "vit_base_patch14_dinov2_224", "efficientnetv2_s": "tf_efficientnetv2_s_224", "resnetv2_101": "resnetv2_101_224"}
fg = M.groupby("foto_grup").split.agg(lambda s: set(s))
test = M[M.split == "test"]
rgb = test[~test.termal.astype(bool)]
lf = rgb[rgb.foto_grup.map(lambda g: not ({"train", "valid"} & fg[g]))]
P(f"RGB test {len(rgb)}; leak-free by photo group (no same-photo copy in train/valid) {len(lf)} files, "
  f"{lf.foto_grup.nunique()} photo groups; eligible {int(lf.uygun.sum())}; representatives {int(lf.temsilci.sum())}")
ev = M[M.kat_v1 >= 0]
ev_lf = ev[ev.foto_grup.isin(set(lf.foto_grup))]
ev_t = ev[ev.split == "test"]
P(f"eval-set reps from leak-free test photo groups: {len(ev_lf)} {ev_lf.label.map(dict(zip(SIN, AB))).value_counts().to_dict()}; "
  f"eval-set reps that are test files: {len(ev_t)} {ev_t.label.map(dict(zip(SIN, AB))).value_counts().to_dict()}")
# sinif bilesimi agirlikli P3: resmi RGB test sinif karisimi
mix = rgb.label.value_counts(normalize=True)
P("official RGB test class mix:", mix.round(3).to_dict())
for k, run in P1.items():
    t1 = pd.read_csv(PIL / "runs" / run / "test_predictions.csv").set_index("path")
    c1 = (t1.label == t1.tahmin)
    term = np.asarray(t1.index.map(M["termal"]).astype(bool))
    lfm = np.asarray(t1.index.isin(lf.index))
    evm = np.asarray(t1.index.isin(ev_lf.index))
    etm = np.asarray(t1.index.isin(ev_t.index))
    d3 = pd.concat([pd.read_csv(PIL / "runs_asama2" / f"grouped{f}_{P3[k]}_s42_kat_v1_asama2_none" / "test_predictions.csv")
                    for f in range(5)]).set_index("path")
    c3 = (d3.label == d3.tahmin)
    m3l = np.asarray(d3.index.isin(ev_lf.index))
    m3t = np.asarray(d3.index.isin(ev_t.index))
    # P3 sinif karisimi P1 RGB testine agirliklandirilmis
    rec3 = d3.assign(c=c3).groupby("label").c.mean()
    w3 = float((rec3 * mix).sum())
    P(f"{k:17s} P1 no-TTA: all {c1.mean():.4f} (n={len(c1)}) | RGB {c1[~term].mean():.4f} (n={int((~term).sum())}) | "
      f"leak-free files {c1[lfm].mean():.4f} (n={int(lfm.sum())}) | leak-free reps {c1[evm].mean():.4f} (n={int(evm.sum())}) | "
      f"test-file reps {c1[etm].mean():.4f} (n={int(etm.sum())})")
    P(f"{'':17s} P3 s42 OOF: all {c3.mean():.4f} | leak-free reps {c3[m3l].mean():.4f} (n={int(m3l.sum())}) | test-file reps "
      f"{c3[m3t].mean():.4f} (n={int(m3t.sum())}) | class-recall weighted to official RGB test mix {w3:.4f}")
(OUT / "r1_a4_sonuc.txt").write_text("\n".join(L), encoding="utf-8")

# kesisim: ayni goruntuler (P1 test dosyasi = P3 temsilcisi), sizintisiz foto gruplari
L2 = []
t0 = pd.read_csv(PIL / "runs" / P1["dinov2"] / "test_predictions.csv").set_index("path")
miss = ev_lf.index[~ev_lf.index.isin(t0.index)]
L2.append(f"leak-free reps not in P1 test predictions: {len(miss)}; their splits: {M.loc[miss, 'split'].value_counts().to_dict()}; "
          f"foto groups' splits: {[sorted(fg[g]) for g in M.loc[miss, 'foto_grup']][:5]}")
for k, run in P1.items():
    t1 = pd.read_csv(PIL / "runs" / run / "test_predictions.csv").set_index("path")
    d3 = pd.concat([pd.read_csv(PIL / "runs_asama2" / f"grouped{f}_{P3[k]}_s42_kat_v1_asama2_none" / "test_predictions.csv")
                    for f in range(5)]).set_index("path")
    ids = sorted(set(t1.index) & set(ev_lf.index) & set(d3.index))
    a1 = (t1.loc[ids].label == t1.loc[ids].tahmin).mean()
    a3 = (d3.loc[ids].label == d3.loc[ids].tahmin).mean()
    ids2 = sorted(set(t1.index) & set(ev_t.index) & set(d3.index))
    b1 = (t1.loc[ids2].label == t1.loc[ids2].tahmin).mean()
    b3 = (d3.loc[ids2].label == d3.loc[ids2].tahmin).mean()
    L2.append(f"{k:17s} same leak-free images (n={len(ids)}): P1 {a1:.4f} vs P3 {a3:.4f} | same test-file reps (n={len(ids2)}): "
              f"P1 {b1:.4f} vs P3 {b3:.4f}")
print("\n".join(L2))
with open(OUT / "r1_a4_sonuc.txt", "a", encoding="utf-8") as fh:
    fh.write("\n" + "\n".join(L2))
