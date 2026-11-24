"""Adım 9 - 2. aşama eğitimleri (kullanıcı kararları, 18 Kasım: iç doğrulama yok + sabit 30 epoch (son epoch),
pilottaki 5 model 224 px, 3 tohum, sınıf ağırlığı yok (pilotla aynı), TTA açık).
Manifest: asama2/manifest_asama2.csv (r4.1), kat sütunu kat_v1, grup sahne_grup. Eğitici: 02b_egit_asama2.py
(oneri/02_egit_asama2_oneri.py'nin kopyası; 02_egit.py değişmedi). Çıktı: runs_asama2/.
Sürdürülebilir: results.json'ı olan koşu atlanır; yarım kalan koşu baştan çalışır. Sıra: tohum → model → kat.
Kullanım: python 09_asama2_egitim.py   [--smoke]"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODELLER = [("resnetv2_101.a1h_in1k", "1e-4"), ("convnext_tiny.fb_in22k_ft_in1k", "1e-4"),
            ("tf_efficientnetv2_s.in21k_ft_in1k", "1e-4"), ("swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", "5e-5"),
            ("vit_base_patch14_dinov2.lvd142m", "2e-5")]
TOHUMLAR = [42, 43, 44]
KATLAR = [0, 1, 2, 3, 4]
OUT = Path("runs_asama2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    Path("loglar/asama2_egitim").mkdir(parents=True, exist_ok=True)
    ortam = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    isler = [(s, m, lr, k) for s in TOHUMLAR for m, lr in MODELLER for k in KATLAR]
    if a.smoke:
        isler = isler[:1]
    for n, (s, m, lr, k) in enumerate(isler, 1):
        kisa = m.split(".")[0]
        ad = f"grouped{k}_{kisa}_224_s{s}_kat_v1_asama2_none"
        if not a.smoke and (OUT / ad / "results.json").exists():
            continue
        komut = [sys.executable, "-B", "02b_egit_asama2.py", "--manifest", "asama2/manifest_asama2.csv",
                 "--protocol", "grouped", "--fold-col", "kat_v1", "--group-col", "sahne_grup", "--fold", str(k),
                 "--val-mode", "none", "--select", "last", "--model", m, "--lr", lr, "--epochs", "30", "--tta",
                 "--batch-size", "32", "--workers", "8", "--seed", str(s), "--out", str(OUT)]
        if a.smoke:
            komut.append("--smoke")
        t0 = time.time()
        print(f"[{n}/{len(isler)}] {ad}", flush=True)
        with open(f"loglar/asama2_egitim/{ad}.log", "w", encoding="utf-8") as f:
            r = subprocess.run(komut, stdout=f, stderr=subprocess.STDOUT, env=ortam)
        print(f"    kod={r.returncode}  {(time.time() - t0) / 60:.1f} dk", flush=True)
        if r.returncode != 0:
            print("    HATA: log'a bakın; sonraki koşuya geçiliyor", flush=True)
    biten = len(list(OUT.glob("*/results.json"))) if OUT.exists() else 0
    print(json.dumps({"biten_kosu": biten, "toplam": len(isler)}))


if __name__ == "__main__":
    main()
