"""İç değerlendirme R1 kontrolü - D: P2/P3 havuzlanmis OOF metrikleri TTA'siz mi? results.json test/test_tta karisiklik matrislerini
katlar uzerinden toplar. Yalniz okur."""
import json
from pathlib import Path

import numpy as np

PIL = Path(r"C:\solarfcd\pilot")


def mf1(c):
    c = np.array(c, float)
    den = c.sum(0) + c.sum(1)
    return np.mean(np.where(den > 0, 2 * np.diag(c) / np.maximum(den, 1e-12), 0))


for kok, son in [("runs_asama2", "kat_v1_asama2_none"), ("runs_rastgele", "kat_rastgele_asama2_rastgele_none")]:
    for u in ["convnext_tiny_224", "swin_tiny_patch4_window7_224_224", "vit_base_patch14_dinov2_224",
              "vit_base_patch14_dinov2_392", "tf_efficientnetv2_s_224", "resnetv2_101_224"]:
        for s in (42, 43, 44):
            fs = [PIL / kok / f"grouped{k}_{u}_s{s}_{son}" / "results.json" for k in range(5)]
            if not all(f.exists() for f in fs):
                continue
            C0 = sum(np.array(json.load(open(f, encoding="utf-8"))["test"]["confusion_matrix"]) for f in fs)
            C1 = sum(np.array(json.load(open(f, encoding="utf-8"))["test_tta"]["confusion_matrix"]) for f in fs)
            print(f"{kok:13s} {u:34s} s{s}: no TTA acc {np.trace(C0) / C0.sum():.4f} F1 {mf1(C0):.4f} | TTA acc "
                  f"{np.trace(C1) / C1.sum():.4f} F1 {mf1(C1):.4f} (n={C0.sum()})")
