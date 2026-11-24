"""Adım 16 - Modele göre öğrenme hızı ayarı (kullanıcı kararı 20 Kasım: iç doğrulamayla seç).
Topluluk anlamlı iyileşme vermediği için. ConvNeXt, Swin, DINOv2; ızgara = temel lr × {0,5; 1; 3}.
1. aşama (seçim): her lr × 5 kat, iç doğrulama sgkf (%12,5), 30 epoch, son epoch; ölçüt = iç doğrulama makro-F1'inin
   5 kat ortalaması. Test katları seçimde KULLANILMAZ.
2. aşama: seçilen lr temelden farklıysa iç doğrulamasız (09 ile aynı protokol) tohum 42, 5 kat; çıktı adı _lrsec.
Sürdürülebilir. Çıktı: runs_lr/ ; özet asama2_sonuclar/lr_ayari.csv"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

MODELLER = [("convnext_tiny.fb_in22k_ft_in1k", 1e-4), ("swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", 5e-5),
            ("vit_base_patch14_dinov2.lvd142m", 2e-5)]
CARPAN = [0.5, 1, 3]
OUT = Path("runs_lr")
ortam = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
Path("loglar/lr").mkdir(parents=True, exist_ok=True)


def kos(mdl, lr, k, mod, tag):
    kisa = mdl.split(".")[0]
    ad = f"grouped{k}_{kisa}_224_s42_kat_v1_asama2_{mod}_{tag}"
    if not (OUT / ad / "results.json").exists():
        komut = [sys.executable, "-B", "02b_egit_asama2.py", "--manifest", "asama2/manifest_asama2.csv", "--protocol",
                 "grouped", "--fold-col", "kat_v1", "--group-col", "sahne_grup", "--fold", str(k), "--val-mode", mod,
                 "--select", "last", "--model", mdl, "--lr", f"{lr:g}", "--epochs", "30", "--tta", "--batch-size",
                 "32", "--workers", "8", "--seed", "42", "--out", str(OUT), "--tag", tag]
        with open(f"loglar/lr/{ad}.log", "w", encoding="utf-8") as f:
            r = subprocess.run(komut, stdout=f, stderr=subprocess.STDOUT, env=ortam)
        print(ad, "kod", r.returncode, flush=True)
    return json.load(open(OUT / ad / "results.json"))


satir = []
for mdl, lr0 in MODELLER:
    for c in CARPAN:
        v = [kos(mdl, lr0 * c, k, "sgkf", f"lr{c:g}")["best_val_macro_f1"] for k in range(5)]
        satir.append(dict(model=mdl.split(".")[0], carpan=c, lr=lr0 * c, ic_dogrulama_f1=sum(v) / 5,
                          kat_f1=" ".join(f"{x:.3f}" for x in v)))
T = pd.DataFrame(satir)
sec = T.loc[T.groupby("model")["ic_dogrulama_f1"].idxmax()].set_index("model")
T["secildi"] = [sec.at[m, "carpan"] == c for m, c in zip(T.model, T.carpan)]
T.to_csv("asama2_sonuclar/lr_ayari.csv", index=False, encoding="utf-8-sig")
print(T.to_string(), flush=True)
for mdl, lr0 in MODELLER:
    c = sec.at[mdl.split(".")[0], "carpan"]
    if c != 1:
        for k in range(5):
            kos(mdl, lr0 * c, k, "none", "lrsec")
print("bitti", flush=True)
