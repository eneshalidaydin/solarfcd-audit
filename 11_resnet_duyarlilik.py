"""Adım 11 - ResNetV2-101 duyarlılık (kullanıcı kararı 19 Kasım: farklı protokoller dene; sonuç değişmezse ek
duyarlılık analizi). Ana protokol (09) değişmez; bunlar ek tablo içindir. Tohum 42, 5 kat, r4.1, kat_v1.
  A: katmanlı grup iç doğrulama (sgkf, %12,5) + doğrulama makro-F1 ile en iyi epoch (sabır 8), 30 epoch
  B: iç doğrulama yok, 60 epoch (son epoch)
  C: iç doğrulama yok, 30 epoch, lr 3e-4
Sürdürülebilir (results.json varsa atlar). Çıktı runs_asama2/ (adında _dA/_dB/_dC)."""
import os
import subprocess
import sys
from pathlib import Path

VAR = {"dA": ["--val-mode", "sgkf", "--select", "f1", "--epochs", "30", "--lr", "1e-4"],
       "dB": ["--val-mode", "none", "--select", "last", "--epochs", "60", "--lr", "1e-4"],
       "dC": ["--val-mode", "none", "--select", "last", "--epochs", "30", "--lr", "3e-4"]}
ortam = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
Path("loglar/asama2_egitim").mkdir(parents=True, exist_ok=True)
for v, ek in VAR.items():
    for k in range(5):
        mod = "sgkf" if v == "dA" else "none"
        ad = f"grouped{k}_resnetv2_101_224_s42_kat_v1_asama2_{mod}_{v}"
        if (Path("runs_asama2") / ad / "results.json").exists():
            continue
        komut = [sys.executable, "-B", "02b_egit_asama2.py", "--manifest", "asama2/manifest_asama2.csv", "--protocol",
                 "grouped", "--fold-col", "kat_v1", "--group-col", "sahne_grup", "--fold", str(k), "--model",
                 "resnetv2_101.a1h_in1k", "--tta", "--batch-size", "32", "--workers", "8", "--seed", "42",
                 "--out", "runs_asama2", "--tag", v] + ek
        with open(f"loglar/asama2_egitim/{ad}.log", "w", encoding="utf-8") as f:
            r = subprocess.run(komut, stdout=f, stderr=subprocess.STDOUT, env=ortam)
        print(ad, "kod", r.returncode, flush=True)
