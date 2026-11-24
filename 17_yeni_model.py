"""Adım 17 - Referans makalede olmayan yeni model (kullanıcı kararı 20 Kasım: lr ayarı tatmin etmezse; model seçimi
bana bırakıldı). ConvNeXt-V2 Tiny (FCMAE ön eğitim + IN-22k ince ayar, 28 M parametre; ConvNeXt-T ile aynı boyut,
daha güçlü ön eğitim). Ana protokolle birebir (09): iç doğrulama yok, 30 epoch, lr 1e-4, 224 px, TTA, 3 tohum × 5 kat.
Çıktı runs_asama2/ (10 ve 15 tarafından otomatik okunur). Sürdürülebilir."""
import os
import subprocess
import sys
from pathlib import Path

MDL = "convnextv2_tiny.fcmae_ft_in22k_in1k"
ortam = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
for s in (42, 43, 44):
    for k in range(5):
        ad = f"grouped{k}_convnextv2_tiny_224_s{s}_kat_v1_asama2_none"
        if (Path("runs_asama2") / ad / "results.json").exists():
            continue
        komut = [sys.executable, "-B", "02b_egit_asama2.py", "--manifest", "asama2/manifest_asama2.csv", "--protocol",
                 "grouped", "--fold-col", "kat_v1", "--group-col", "sahne_grup", "--fold", str(k), "--val-mode", "none",
                 "--select", "last", "--model", MDL, "--lr", "1e-4", "--epochs", "30", "--tta", "--batch-size", "32",
                 "--workers", "8", "--seed", str(s), "--out", "runs_asama2"]
        with open(f"loglar/asama2_egitim/{ad}.log", "w", encoding="utf-8") as f:
            r = subprocess.run(komut, stdout=f, stderr=subprocess.STDOUT, env=ortam)
        print(ad, "kod", r.returncode, flush=True)
