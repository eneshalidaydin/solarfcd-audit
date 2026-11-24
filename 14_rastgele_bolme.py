"""Adım 14 - Sızıntılı karşılaştırma (kullanıcı kararı 20 Kasım: eksik analizler). Aynı 2.095 temsilci görüntü,
grupları YOK SAYAN rastgele StratifiedKFold(5, tohum 42; label|kaynak). Diğer her şey 09 ile aynı (5 model, 30 epoch,
iç doğrulama yok, TTA), tohum 42. Böylece aynı test görüntülerinde sızıntılı/sızıntısız fark doğrudan ölçülür.
Manifest: asama2_rastgele/manifest_asama2.csv (asama2 kopyası + kat_rastgele). Çıktı: runs_rastgele/. Sürdürülebilir."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

MAN = Path("asama2_rastgele/manifest_asama2.csv")
if not MAN.exists():
    m = pd.read_csv("asama2/manifest_asama2.csv")
    t = m[m["kat_v1"] >= 0]
    m["kat_rastgele"] = -1
    for k, (_, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42)
                                .split(t, t["label"] + "|" + t["kaynak"])):
        m.loc[t.index[te], "kat_rastgele"] = k
    MAN.parent.mkdir(exist_ok=True)
    m.to_csv(MAN, index=False)
    t = m[m["kat_rastgele"] >= 0]
    for g in ["foto_grup", "sahne_grup"]:
        cok = t.groupby(g)["kat_rastgele"].transform("nunique") > 1
        print(f"{g}: başka katta eşi olan görüntü oranı {cok.mean():.1%}")
MODELLER = [("resnetv2_101.a1h_in1k", "1e-4"), ("convnext_tiny.fb_in22k_ft_in1k", "1e-4"),
            ("tf_efficientnetv2_s.in21k_ft_in1k", "1e-4"), ("swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", "5e-5"),
            ("vit_base_patch14_dinov2.lvd142m", "2e-5")]
ortam = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
Path("loglar/rastgele").mkdir(parents=True, exist_ok=True)
for mdl, lr in MODELLER:
    for k in range(5):
        ad = f"grouped{k}_{mdl.split('.')[0]}_224_s42_kat_rastgele_asama2_rastgele_none"
        if (Path("runs_rastgele") / ad / "results.json").exists():
            continue
        komut = [sys.executable, "-B", "02b_egit_asama2.py", "--manifest", str(MAN), "--protocol", "grouped",
                 "--fold-col", "kat_rastgele", "--group-col", "rel_path", "--fold", str(k), "--val-mode", "none",
                 "--select", "last", "--model", mdl, "--lr", lr, "--epochs", "30", "--tta", "--batch-size", "32",
                 "--workers", "8", "--seed", "42", "--out", "runs_rastgele"]
        with open(f"loglar/rastgele/{ad}.log", "w", encoding="utf-8") as f:
            r = subprocess.run(komut, stdout=f, stderr=subprocess.STDOUT, env=ortam)
        print(ad, "kod", r.returncode, flush=True)
