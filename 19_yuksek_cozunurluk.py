"""Adım 19 - Yüksek çözünürlük (kullanıcı kararı 21 Kasım; öneri B). Ana protokol (09) aynı, yalnız giriş boyutu:
ConvNeXt-T 384 px; DINOv2 ViT-B/14 392 px (14'ün katı). 3 tohum × 5 kat. Swin-T (pencere 7) 384'te ızgara uyumsuz -> dışarıda.
Bellek yetmezse (CUDA OOM) yalnız o koşu batch 16 ile yinelenir ve loga yazılır. Çıktı runs_asama2/ (ad: _384_/_392_)."""
import os
import subprocess
import sys
from pathlib import Path

ISLER = [("convnext_tiny.fb_in22k_ft_in1k", "1e-4", 384), ("vit_base_patch14_dinov2.lvd142m", "2e-5", 392)]
ortam = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
for s in (42, 43, 44):
    for mdl, lr, S in ISLER:
        for k in range(5):
            ad = f"grouped{k}_{mdl.split('.')[0]}_{S}_s{s}_kat_v1_asama2_none"
            if (Path("runs_asama2") / ad / "results.json").exists():
                continue
            for bs in ("32", "16"):
                komut = [sys.executable, "-B", "02b_egit_asama2.py", "--manifest", "asama2/manifest_asama2.csv",
                         "--protocol", "grouped", "--fold-col", "kat_v1", "--group-col", "sahne_grup", "--fold", str(k),
                         "--val-mode", "none", "--select", "last", "--model", mdl, "--lr", lr, "--epochs", "30",
                         "--img-size", str(S), "--tta", "--batch-size", bs, "--workers", "8", "--seed", str(s),
                         "--out", "runs_asama2"]
                log = Path(f"loglar/asama2_egitim/{ad}_bs{bs}.log")
                with open(log, "w", encoding="utf-8") as f:
                    r = subprocess.run(komut, stdout=f, stderr=subprocess.STDOUT, env=ortam)
                print(ad, "bs", bs, "kod", r.returncode, flush=True)
                if r.returncode == 0 or "out of memory" not in log.read_text(encoding="utf-8", errors="ignore"):
                    break
