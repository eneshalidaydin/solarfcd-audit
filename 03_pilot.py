"""Adım 3 - Pilot akışını tek komutla çalıştır (1. gün).

Kullanım ("Anaconda Prompt" içinde, ortam etkinken, pilot klasöründe):
    python 03_pilot.py --veri-kok "C:\\solarfcd\\data\\SolarFCD-Final"

Seçenekler:
    --batch 32        RTX 4080 (12 GB) için 32; RTX 4070 (8 GB) için 16; RTX 4090 (16 GB) için 64
    --epoch 30        Her model için en fazla epoch (erken durdurma var)
    --isci 8          Veri yükleme işçisi; donma/hata olursa 0 dene
    --atla-denetim    01_veri_denetimi.py daha önce çalıştıysa tekrar etme
    --yalniz MODEL    Yalnız adı verilen modeli/modelleri çalıştır (ör. --yalniz convnext_tiny)

Her adımın konsol çıktısı loglar/ klasörüne de yazılır. Bir adım hata verirse akış durur.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from ortak import utf8_konsol

MODELLER = [
    ("resnetv2_101.a1h_in1k", "1e-4"),                          # makaledeki en iyi modelin (ResNet101V2) karşılığı
    ("convnext_tiny.fb_in22k_ft_in1k", "1e-4"),
    ("tf_efficientnetv2_s.in21k_ft_in1k", "1e-4"),
    ("swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", "5e-5"),
    ("vit_base_patch14_dinov2.lvd142m", "2e-5"),
]


def calistir(ad, argumanlar, log_klasoru):
    komut = [sys.executable] + argumanlar
    print(f"\n>>> {ad}\n    {' '.join(komut[1:])}", flush=True)
    ortam = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    log = Path(log_klasoru) / f"{ad}.log"
    with open(log, "w", encoding="utf-8") as f, subprocess.Popen(
            komut, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=ortam,
            encoding="utf-8", errors="replace", bufsize=1) as p:
        for satir in p.stdout:
            print(satir, end="", flush=True)
            f.write(satir)
        p.wait()
    if p.returncode != 0:
        raise SystemExit(f"\n[!] '{ad}' adımı hata verdi (çıkış kodu {p.returncode}). Ayrıntı: {log}")


def main():
    utf8_konsol()
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri-kok", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epoch", type=int, default=30)
    ap.add_argument("--isci", type=int, default=8)
    ap.add_argument("--atla-denetim", action="store_true")
    ap.add_argument("--yalniz", nargs="*", default=None)
    ap.add_argument("--test-modu", action="store_true", help=argparse.SUPPRESS)  # yalnız paket testi için
    args = ap.parse_args()

    burasi = Path(__file__).resolve().parent
    os.chdir(burasi)
    loglar = burasi / "loglar"
    loglar.mkdir(exist_ok=True)
    manifest = str(Path("denetim_final") / "manifest.csv")
    t0 = time.time()

    modeller = MODELLER
    ek, img, smoke_model, lp_model = [], "224", "convnext_tiny.fb_in22k_ft_in1k", "vit_base_patch14_dinov2.lvd142m"
    if args.test_modu:
        modeller = [("resnet18", "1e-3")]
        ek, img, smoke_model, lp_model = ["--no-pretrained"], "64", "resnet18", "resnet18"
    if args.yalniz:
        modeller = [m for m in modeller if any(y in m[0] for y in args.yalniz)]

    if not args.atla_denetim:
        calistir("00_ortam", ["00_ortam_kontrol.py", "--veri-klasoru", args.veri_kok], loglar)
        denetim = ["01_veri_denetimi.py", "--kok", args.veri_kok, "--cikti", "denetim_final", "--derin",
                   "--isci", str(args.isci)]
        if args.test_modu:
            denetim += ["--derin-onegitimsiz", "--derin-model", "resnet18"]
        calistir("01_denetim", denetim, loglar)

    ortak = ["--manifest", manifest, "--batch-size", str(args.batch), "--workers", str(args.isci),
             "--img-size", img] + ek
    calistir("02_smoke", ["02_egit.py", "--model", smoke_model, "--smoke"] + ortak, loglar)
    for model, lr in modeller:
        calistir("02_" + model.split(".")[0],
                 ["02_egit.py", "--model", model, "--lr", lr, "--epochs", str(args.epoch), "--tta"] + ortak, loglar)
    if not args.yalniz:
        calistir("02_dinov2_lp", ["02_egit.py", "--model", lp_model, "--linear-probe", "--lr", "1e-3",
                                  "--epochs", str(min(15, args.epoch)), "--tta"] + ortak, loglar)
    calistir("04_ozet", ["04_sonuclari_topla.py"], loglar)

    print(f"\nPilot tamamlandı. Toplam süre: {(time.time() - t0) / 60:.0f} dk")
    print("Bana gönder: ozet_sonuclar.md, denetim_final/ozet.json, denetim_final/sayim_bolme_sinif.csv,\n"
          "             denetim_final/meta_dosyalari.txt, denetim_final/yakin_kopya_ornekleri.png,\n"
          "             runs/*/confusion_matrix.png")


if __name__ == "__main__":
    main()
