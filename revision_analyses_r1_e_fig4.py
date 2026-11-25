"""İç değerlendirme R1 kontrolü - E: Sekil 4 dis cubuklari (DINOv2-224, P3, tohum 42, kat toplulugu) icin kok-ad kume bootstrap GA. Yalniz okur."""
import re
from pathlib import Path
import numpy as np
import pandas as pd
DEP = Path(r"<DEPOSIT_DIR>")
SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
stem = lambda p: re.split(r"[._]rf[._]|_jpg\.rf", Path(p).name)[0]
rng = np.random.default_rng(20261124)
def boot(c, s, B=2000):
    g = pd.DataFrame(dict(c=c.astype(float), s=s)).groupby("s").c.agg(["sum", "count"])
    i = rng.integers(0, len(g), size=(B, len(g)))
    v = g["sum"].to_numpy()[i].sum(1) / g["count"].to_numpy()[i].sum(1)
    return np.percentile(v, [2.5, 97.5]), len(g)
Q = pd.read_csv(DEP / "external_predictions_rf_fault.csv")
for m in ["DINOv2-224_P3_seed42", "DINOv2-224_P2_seed42"]:
    pr = Q[[f"{m}__p_{c}" for c in SIN]].to_numpy().argmax(1)
    k = Q.mapped_label.isin(SIN).to_numpy()
    y = Q.mapped_label[k].map(SIN.index).to_numpy()
    c = pr[k] == y
    ci, ng = boot(c, Q.rel_path[k].map(stem).to_numpy())
    print(f"RF-Fault {m}: 4-class acc {c.mean():.4f} (n={k.sum()}) stem bootstrap {ci[0]:.4f}-{ci[1]:.4f} ({ng} stems)")
S = pd.read_csv(DEP / "external_predictions_rf_soiling.csv")
for m in ["DINOv2-224_P3_seed42", "DINOv2-224_P2_seed42"]:
    pr = S[[f"{m}__p_{c}" for c in SIN]].to_numpy().argmax(1)
    c = pr == 3
    ci, ng = boot(c, S.rel_path.map(stem).to_numpy())
    print(f"RF-Soiling {m}: SO recall {c.mean():.4f} (n={len(c)}) stem bootstrap {ci[0]:.4f}-{ci[1]:.4f} ({ng} stems)")
