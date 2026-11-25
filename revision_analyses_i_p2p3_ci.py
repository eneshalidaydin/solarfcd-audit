"""R2.8: paired scene-cluster bootstrap CIs (2,000 resamples, rng 0) for the P2 minus P3 accuracy and macro-F1 differences
(seed 42, same 2,095 images). Read-only inputs."""
import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
K = "C:/solarfcd/pilot/"
SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
man = pd.read_csv(K + "asama2/manifest_asama2.csv").set_index("path")


def rd(kok, name):
    return pd.concat([pd.read_csv(f"{K}{kok}/{name.replace('grouped0_', f'grouped{k}_')}/test_predictions.csv")
                      for k in range(5)]).set_index("path").sort_index()


def f1(y, p, w):
    c = np.zeros((4, 4)); np.add.at(c, (y, p), w)
    den = c.sum(0) + c.sum(1)
    return np.mean(np.where(den > 0, 2 * np.diag(c) / np.maximum(den, 1e-12), 0))


ref = rd("runs_asama2", "grouped0_convnext_tiny_224_s42_kat_v1_asama2_none")
paths = ref.index
g = paths.map(man["sahne_grup"]).to_numpy()
gu, gi = np.unique(g, return_inverse=True)
rng = np.random.default_rng(0)
W = np.stack([np.bincount(rng.choice(len(gu), len(gu)), minlength=len(gu))[gi] for _ in range(2000)])
y = ref.label.map(SIN.index).to_numpy()
for mdl in ["convnext_tiny_224", "swin_tiny_patch4_window7_224_224", "vit_base_patch14_dinov2_224", "tf_efficientnetv2_s_224",
            "resnetv2_101_224"]:
    p3 = rd("runs_asama2", f"grouped0_{mdl}_s42_kat_v1_asama2_none").loc[paths].tahmin.map(SIN.index).to_numpy()
    p2 = rd("runs_rastgele", f"grouped0_{mdl}_s42_kat_rastgele_asama2_rastgele_none").loc[paths].tahmin.map(SIN.index).to_numpy()
    da = np.array([np.average(p2 == y, weights=w) - np.average(p3 == y, weights=w) for w in W])
    dfm = np.array([f1(y, p2, w) - f1(y, p3, w) for w in W])
    one = np.ones(len(y))
    print(f"{mdl:34s} P2-P3 acc {100 * ((p2 == y).mean() - (p3 == y).mean()):+.2f} pp [{100 * np.percentile(da, 2.5):+.2f}, "
          f"{100 * np.percentile(da, 97.5):+.2f}]; macro-F1 {f1(y, p2, one) - f1(y, p3, one):+.4f} [{np.percentile(dfm, 2.5):+.3f}, "
          f"{np.percentile(dfm, 97.5):+.3f}]")
