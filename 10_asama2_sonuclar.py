"""Adım 10 - 2. aşama sonuçları: model × tohum için 5 katın test tahminleri birleştirilir (havuzlanmış OOF, TTA'lı
test_predictions.csv). Metrikler: doğruluk, makro-F1, sınıf duyarlılıkları. %95 GA: sahne_grup düzeyinde küme bootstrap
(2.000 tekrar, tohum 0). Tohumlar arası ortalama ± ss; ayrıca 0. kat hariç duyarlılık (büyük grup 36 0. katta).
Pilotun resmî bölme sonucu karşılaştırma için. Çıktı: asama2_sonuclar/ (tablo.csv, tablo.txt, tohum_bazinda.csv)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
m = pd.read_csv("asama2/manifest_asama2.csv")
sg = m.set_index("path")["sahne_grup"]
kat = m.set_index("path")["kat_v1"]
rng = np.random.default_rng(0)


def metrik(y, p):
    return dict(acc=float((y == p).mean()), f1=float(f1_score(y, p, labels=SIN, average="macro", zero_division=0)),
                **{f"rec_{c[:2]}": float((p[y == c] == c).mean()) for c in SIN})


def bootstrap(d, B=2000):
    g = d["sg"].to_numpy()
    gu = np.unique(g)
    ix = {k: np.flatnonzero(g == k) for k in gu}
    y, p = d["label"].to_numpy(), d["tahmin"].to_numpy()
    f = []
    for _ in range(B):
        s = np.concatenate([ix[k] for k in rng.choice(gu, len(gu))])
        f.append(f1_score(y[s], p[s], labels=SIN, average="macro", zero_division=0))
    return np.percentile(f, [2.5, 97.5])


satir = []
for d in sorted(Path("runs_asama2").glob("grouped0_*_none")):
    model, tohum = d.name.split("_224_")[0][len("grouped0_"):], d.name.split("_s")[-1].split("_")[0]
    parca = [pd.read_csv(Path("runs_asama2") / d.name.replace("grouped0_", f"grouped{k}_") / "test_predictions.csv")
             for k in range(5)]
    D = pd.concat(parca, ignore_index=True)
    assert len(D) == (m["kat_v1"] >= 0).sum() and D["path"].is_unique
    D["sg"], D["kat"] = D["path"].map(sg), D["path"].map(kat)
    r = dict(model=model, tohum=int(tohum), **metrik(D["label"].to_numpy(), D["tahmin"].to_numpy()))
    lo, hi = bootstrap(D)
    r.update(f1_lo=lo, f1_hi=hi)
    D0 = D[D["kat"] != 0]
    r["f1_kat0_haric"] = metrik(D0["label"].to_numpy(), D0["tahmin"].to_numpy())["f1"]
    satir.append(r)
T = pd.DataFrame(satir)
out = Path("asama2_sonuclar")
out.mkdir(exist_ok=True)
T.to_csv(out / "tohum_bazinda.csv", index=False, encoding="utf-8-sig")
O = T.groupby("model").agg(["mean", "std"])
cols = ["acc", "f1", "f1_lo", "f1_hi", "rec_El", "rec_He", "rec_St", "rec_Su", "f1_kat0_haric"]
ozet = pd.DataFrame({c: O[c]["mean"].map("{:.4f}".format) + " ± " + O[c]["std"].map("{:.4f}".format) for c in cols})
pilot = {}
for mdl in T["model"].unique():
    f = Path(f"runs/official_{mdl}_224_s42/results.json")
    if f.exists():
        t = json.load(open(f))["test_tta"]
        pilot[mdl] = (t.get("accuracy"), t.get("macro_f1"))
ozet["pilot_resmi_acc/f1"] = [str(pilot.get(i)) for i in ozet.index]
ozet.to_csv(out / "tablo.csv", encoding="utf-8-sig")
(out / "tablo.txt").write_text(ozet.to_string(), encoding="utf-8")
print(ozet.to_string())
