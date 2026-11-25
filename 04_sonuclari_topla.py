"""Adım 4 - Sonuçları tek tabloda topla.

runs/*/results.json dosyalarını okur, SolarFCD makalesindeki referans sonucu da ekleyerek
ozet_sonuclar.csv ve ozet_sonuclar.md üretir. Grup bazlı (grouped) koşularda katların
ortalama ± standart sapmasını da verir.

Kullanım:  python 04_sonuclari_topla.py [--runs runs]
"""
import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from ortak import utf8_konsol

REFERANS = dict(kosu="SolarFCD makalesi (ResNet101V2)", model="ResNet101V2 (Keras)", protokol="official",
                img=224, acc=0.8668, macro_f1=0.8817, mcc=None, kappa=None, acc_tta=None)


def main():
    utf8_konsol()
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    satirlar = []
    for f in sorted(glob.glob(str(Path(args.runs) / "*" / "results.json"))):
        if "SMOKE_" in f:
            continue
        r = json.loads(Path(f).read_text(encoding="utf-8"))
        c, t = r["config"], r["test"]
        tt = r.get("test_tta") or {}
        satirlar.append(dict(
            kosu=Path(f).parent.name, model=c["model"], protokol=c["protocol"],
            fold=(c["fold"] if c["protocol"] == "grouped" else None), img=c["img_size"],
            linear_probe=c.get("linear_probe", False),
            acc=t["accuracy"], macro_f1=t["macro_f1"], mcc=t["mcc"], kappa=t["kappa"],
            balanced_acc=t["balanced_accuracy"], acc_tta=tt.get("accuracy"), macro_f1_tta=tt.get("macro_f1"),
            best_epoch=r["best_epoch"], dakika=r["train_minutes"], n_test=c["n_test"]))
    if not satirlar:
        raise SystemExit("Hiç sonuç bulunamadı (runs/*/results.json).")
    df = pd.DataFrame(satirlar)

    resmi = df[df["protokol"] == "official"].sort_values("acc", ascending=False)
    resmi = pd.concat([pd.DataFrame([REFERANS]), resmi], ignore_index=True)
    sira = ["kosu", "img", "linear_probe", "acc", "acc_tta", "macro_f1", "macro_f1_tta", "mcc", "kappa",
            "balanced_acc", "best_epoch", "dakika", "n_test"]
    resmi = resmi[[c for c in sira if c in resmi.columns]]
    gruplu = df[df["protokol"] == "grouped"]

    parcalar = ["## Resmî bölme (yazarların train/val/test ayrımı)\n", resmi.round(4).to_string(index=False)]
    if len(gruplu):
        agg = (gruplu.groupby(["model", "img", "linear_probe"])
               .agg(kat=("fold", "nunique"), acc_ort=("acc", "mean"), acc_ss=("acc", "std"),
                    f1_ort=("macro_f1", "mean"), f1_ss=("macro_f1", "std"), mcc_ort=("mcc", "mean"))
               .reset_index().sort_values("acc_ort", ascending=False))
        parcalar += ["\n\n## Grup bazlı (sızıntısız) protokol: katların ortalaması ± ss\n",
                     agg.round(4).to_string(index=False)]
        agg.to_csv("ozet_gruplu.csv", index=False, encoding="utf-8-sig")
    metin = "\n".join(parcalar)
    print(metin)
    df.to_csv("ozet_sonuclar.csv", index=False, encoding="utf-8-sig")
    Path("ozet_sonuclar.md").write_text(metin, encoding="utf-8")
    print("\nKaydedildi: ozet_sonuclar.csv, ozet_sonuclar.md" + (", ozet_gruplu.csv" if len(gruplu) else ""))


if __name__ == "__main__":
    main()
