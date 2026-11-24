"""Adım 24 - Temiz (uzman uzlaşılı) etiketlerle yeniden eğitim (önceden sabit plan, 21 Kasım). Manifest
asama2_temiz/manifest_asama2.csv (23). Topluluk-3 yüksek üyeleri: ConvNeXt-T 224, Swin-T 224, DINOv2 392; ana protokol,
3 tohum × 5 kat. Çıktı runs_temiz/. Sürdürülebilir."""
import os
import subprocess
import sys
from pathlib import Path

ISLER = [("convnext_tiny.fb_in22k_ft_in1k", "1e-4", 224), ("swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", "5e-5", 224),
         ("vit_base_patch14_dinov2.lvd142m", "2e-5", 392)]
ortam = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
Path("loglar/temiz").mkdir(parents=True, exist_ok=True)
for s in (42, 43, 44):
    for mdl, lr, S in ISLER:
        for k in range(5):
            ad = f"grouped{k}_{mdl.split('.')[0]}_{S}_s{s}"
            if list(Path("runs_temiz").glob(ad + "*/results.json")):
                continue
            komut = [sys.executable, "-B", "02b_egit_asama2.py", "--manifest", "asama2_temiz/manifest_asama2.csv",
                     "--protocol", "grouped", "--fold-col", "kat_v1", "--group-col", "sahne_grup", "--fold", str(k),
                     "--val-mode", "none", "--select", "last", "--model", mdl, "--lr", lr, "--epochs", "30",
                     "--img-size", str(S), "--tta", "--batch-size", "32", "--workers", "8", "--seed", str(s),
                     "--out", "runs_temiz"]
            with open(f"loglar/temiz/{ad}.log", "w", encoding="utf-8") as f:
                r = subprocess.run(komut, stdout=f, stderr=subprocess.STDOUT, env=ortam)
            print(ad, "kod", r.returncode, flush=True)
print("TEMIZ BITTI", flush=True)
