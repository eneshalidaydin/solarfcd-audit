"""Adım 6 (ek) - Final görüntülerinin ham kaynaklardaki kökenini bayt düzeyinde (MD5) bulur.

Ham zip'ler AÇILMAZ ve DEĞİŞTİRİLMEZ: zipfile ile bellekte okunur. Termal görüntüler (C#, S#) ham tarafta
iç içe .rar içinde olduğundan bu betikte eşlenmez.

Kullanım:
    python 06_ham_kaynak_eslesme.py --final C:\\solarfcd\\data\\SolarFCD-Final --ham C:\\solarfcd\\data\\Raw --cikti denetim_ek
Çıktı: <cikti>/ham_kaynak_eslesme.csv, <cikti>/ham_kaynak_ozet.csv
"""
import argparse
import hashlib
import zipfile
from pathlib import Path

import pandas as pd

from ortak import IMG_EXTS, infer_label, infer_split, list_images, utf8_konsol

HAM_ZIPLER = ["PV Panel Defect Dataset.zip", "Solar Panel dust detection.zip"]


def main():
    utf8_konsol()
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", required=True)
    ap.add_argument("--ham", required=True)
    ap.add_argument("--cikti", default="denetim_ek")
    args = ap.parse_args()
    cikti = Path(args.cikti)
    cikti.mkdir(parents=True, exist_ok=True)

    ham = {}
    for ad in HAM_ZIPLER:
        with zipfile.ZipFile(Path(args.ham) / ad) as z:
            for info in z.infolist():
                if info.is_dir() or Path(info.filename).suffix.lower() not in IMG_EXTS:
                    continue
                h = hashlib.md5(z.read(info)).hexdigest()
                parca = info.filename.split("/")
                ham.setdefault(h, []).append(dict(ham_zip=ad, ham_yol=info.filename,
                                                  ham_bolme=(parca[0] if ad.startswith("PV Panel") else ""),
                                                  ham_dosya=parca[-1]))
        print(f"{ad}: okundu")
    tekrar = sum(len(v) > 1 for v in ham.values())
    print(f"Ham taraftaki farklı içerik sayısı: {len(ham)} (birden çok yerde geçen içerik: {tekrar})")

    kok = Path(args.final)
    satir = []
    for p in list_images(kok):
        parts = list(p.relative_to(kok).parts[:-1])
        h = hashlib.md5(p.read_bytes()).hexdigest()
        eslesen = ham.get(h, [])
        satir.append(dict(rel_path=p.relative_to(kok).as_posix(), dosya=p.name, split=infer_split(parts),
                          label=infer_label(parts)[0], md5=h, ham_eslesme_sayisi=len(eslesen),
                          ham_zip=";".join(e["ham_zip"] for e in eslesen),
                          ham_bolme=";".join(e["ham_bolme"] for e in eslesen),
                          ham_yol=";".join(e["ham_yol"] for e in eslesen),
                          ad_ayni=any(e["ham_dosya"] == p.name for e in eslesen)))
    df = pd.DataFrame(satir)
    df.to_csv(cikti / "ham_kaynak_eslesme.csv", index=False, encoding="utf-8-sig")

    df["sonek"] = df["dosya"].str.extract(r"\)(-[A-Za-z]+)\.", expand=False).fillna("(yok)")
    pv = df[df["ham_zip"].str.contains("PV Panel", na=False)]
    ozet = pd.crosstab([pv["sonek"]], pv["ham_bolme"])
    ozet.to_csv(cikti / "ham_kaynak_ozet.csv", encoding="utf-8-sig")
    print("\nFinal'de bayt düzeyinde ham kaynağı bulunan görüntü:", int((df["ham_eslesme_sayisi"] > 0).sum()), "/", len(df))
    print("Final içinde bayt düzeyinde birebir aynı dosya çifti (farklı adlarla):",
          int(df.duplicated("md5", keep=False).sum()))
    print("\nPV Panel Defect kaynaklı Final görüntüleri: sonek × ham bölme klasörü")
    print(ozet.to_string())
    print("\nFinal bölmesi × ham bölme (PV Panel Defect kaynaklı):")
    print(pd.crosstab(pv["split"], pv["ham_bolme"]).to_string())


if __name__ == "__main__":
    main()
