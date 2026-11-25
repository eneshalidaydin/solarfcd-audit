"""Adım 5 (ek) - Kaynak dağılımı, sızıntı ayrıştırması ve hata analizi.

Resmî değerlendirme protokolünü DEĞİŞTİRMEZ: yalnızca 01 ve 02'nin ürettiği dosyaları yeniden okur.

1. Dosya ailesi (kaynak) x bölme x sınıf sayımı: resmî bölmenin kaynak/alt tür dengesizliği.
2. Bölmeler arası yakın kopyalar (capraz_yakin_kopyalar.csv): yöntem, bölme çifti, etiket uyumu, aile bazında.
3. (runs/ varsa) Her modelin test tahminleri: aile bazında doğruluk; eğitimde yakın kopyası OLAN / OLMAYAN
   test alt kümelerinde doğruluk ve makro-F1; en sık karışan sınıf çiftleri.

Kullanım:
    python 05_ek_analiz.py --denetim denetim_final --runs runs --cikti denetim_ek
"""
import argparse
import glob
import json
import re
from pathlib import Path

import pandas as pd

from ortak import utf8_konsol

SINIFLAR = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]


def aile(dosya):
    """Dosya adından kaynak ailesini ve (varsa) ad-çakışması sonekini çıkarır."""
    f = str(dosya)
    if f.startswith("Imgclean_"):
        return "Imgclean (toz seti)", ""
    if f.startswith("Imgdirty_"):
        return "Imgdirty (toz seti)", ""
    if re.match(r"^S\d+\.", f):
        return "S# (termal)", ""
    if re.match(r"^C\d+\.", f):
        return "C# (termal)", ""
    m = re.match(r"^([A-Za-z]+) \((\d+)\)(?:-([A-Za-z]+))?\.", f)
    if m:
        return m.group(1) + " (PV Panel Defect)", ("-" + m.group(3)) if m.group(3) else ""
    return "diğer", ""


def md_tablo(df, index=True):
    try:
        return df.to_markdown(index=index)
    except Exception:  # tabulate yoksa
        return "```\n" + df.to_string(index=index) + "\n```"


def main():
    utf8_konsol()
    ap = argparse.ArgumentParser()
    ap.add_argument("--denetim", default="denetim_final")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--cikti", default="denetim_ek")
    args = ap.parse_args()

    cikti = Path(args.cikti)
    cikti.mkdir(parents=True, exist_ok=True)
    den = Path(args.denetim)
    m = pd.read_csv(den / "manifest.csv")
    m["aile"], m["sonek"] = zip(*m["dosya"].map(aile))
    rapor = ["# Ek analiz: kaynak dağılımı, sızıntı ve hata ayrıştırması\n",
             "Resmî protokol değiştirilmedi; bu dosya yalnızca mevcut çıktıların yeniden okunmasıyla üretildi.\n"]

    # ------------------------------------------------------------ 1. aile x bölme x sınıf
    sayim = pd.crosstab([m["label"], m["aile"]], m["split"]).reindex(columns=["train", "val", "test"], fill_value=0)
    sayim["toplam"] = sayim.sum(axis=1)
    sayim.to_csv(cikti / "aile_bolme_sinif_sayim.csv", encoding="utf-8-sig")
    rapor += ["## 1. Kaynak ailesi × bölme × sınıf\n", md_tablo(sayim), ""]

    # ------------------------------------------------------------ 2. bölmeler arası yakın kopyalar
    cy = den / "capraz_yakin_kopyalar.csv"
    sizinti = {}
    if cy.exists():
        c = pd.read_csv(cy)
        c["cift"] = c.apply(lambda r: "-".join(sorted([r["split_i"], r["split_j"]])), axis=1)
        c["etiket_ayni"] = c["label_i"] == c["label_j"]
        ozet = c.groupby(["yontem", "cift", "etiket_ayni"]).size().unstack(fill_value=0)
        rapor += ["## 2. Bölmeler arası yakın kopya çiftleri (yöntem × bölme çifti × etiket aynı mı)\n",
                  md_tablo(ozet), ""]
        rel2aile = dict(zip(m["rel_path"], m["aile"]))
        rel2split = dict(zip(m["rel_path"], m["split"]))
        # test/val tarafındaki görüntüler (eğitimde yakın kopyası olan)
        for hedef in ("test", "val"):
            for yontem in ("phash", "derin"):
                alt = c[(c["yontem"] == yontem) & (c["cift"] == "-".join(sorted([hedef, "train"])))]
                yollar = set(alt.apply(lambda r: r["yol_i"] if r["split_i"] == hedef else r["yol_j"], axis=1))
                sizinti[(hedef, yontem)] = yollar
                # etiketi farklı olan çiftler (olası etiket çelişkisi)
                fark = alt[~alt["etiket_ayni"]]
                sizinti[(hedef, yontem, "etiket_farkli")] = set(
                    fark.apply(lambda r: r["yol_i"] if r["split_i"] == hedef else r["yol_j"], axis=1))
        satir = []
        for hedef in ("test", "val"):
            ph, de = sizinti[(hedef, "phash")], sizinti[(hedef, "derin")]
            n = int((m["split"] == hedef).sum())
            birlesik = ph | de
            satir.append(dict(bolme=hedef, goruntu=n, phash=len(ph), derin=len(de), herhangi=len(birlesik),
                              oran_yuzde=round(100 * len(birlesik) / max(1, n), 1),
                              etiketi_farkli=len(sizinti[(hedef, "phash", "etiket_farkli")] |
                                                 sizinti[(hedef, "derin", "etiket_farkli")])))
        sz = pd.DataFrame(satir)
        sz.to_csv(cikti / "sizinti_ozet.csv", index=False, encoding="utf-8-sig")
        rapor += ["### Eğitimde yakın kopyası olan değerlendirme görüntüleri\n", md_tablo(sz, index=False), ""]
        tum = sizinti[("test", "phash")] | sizinti[("test", "derin")]
        te = m[m["split"] == "test"].copy()
        te["sizinti"] = te["rel_path"].isin(tum)
        aile_sz = te.groupby(["label", "aile"])["sizinti"].agg(["size", "sum"]).rename(
            columns={"size": "test_goruntu", "sum": "egitimde_kopyasi_olan"})
        aile_sz.to_csv(cikti / "sizinti_aile.csv", encoding="utf-8-sig")
        rapor += ["### Test: aile bazında yakın kopya sayısı\n", md_tablo(aile_sz), ""]
    else:
        rapor.append("capraz_yakin_kopyalar.csv yok: bölmeler arası yakın kopya bulunmamış.\n")

    # ------------------------------------------------------------ 2b. bayt düzeyinde birebir kopyalar (MD5)
    birebir, celiski = set(), set()
    md5_dosya = Path(args.cikti) / "ham_kaynak_eslesme.csv"
    if md5_dosya.exists():
        h = pd.read_csv(md5_dosya)[["rel_path", "split", "label", "md5"]]
        tr_lab = h[h["split"] == "train"].groupby("md5")["label"].agg(set)
        te_h = h[h["split"] == "test"]
        birebir = set(te_h.loc[te_h["md5"].isin(tr_lab.index), "rel_path"])
        celiski = set(te_h.loc[te_h.apply(lambda r: bool(r["md5"] in tr_lab.index and
                                                         (tr_lab[r["md5"]] - {r["label"]})), axis=1), "rel_path"])
        mk = []
        for s in ("train", "val", "test"):
            q = h[h["split"] == s]
            mk.append(dict(bolme=s, dosya=len(q), tekil_icerik=q["md5"].nunique(),
                           egitimde_birebir_kopyasi_olan=(int(q["md5"].isin(tr_lab.index).sum()) if s != "train" else None)))
        rapor += ["### Bayt düzeyinde (MD5) birebir kopyalar\n", md_tablo(pd.DataFrame(mk), index=False),
                  f"\nTest'te eğitimdeki birebir kopyasından en az biri FARKLI etiketli olan: {len(celiski)}\n"]

    # ------------------------------------------------------------ 3. model tahminlerinin ayrıştırılması
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    kayitlar, aile_kayit, karisma = [], [], []
    for f in sorted(glob.glob(str(Path(args.runs) / "*" / "test_predictions.csv"))):
        kosu = Path(f).parent.name
        if kosu.startswith("SMOKE_"):
            continue
        p = pd.read_csv(f).merge(m[["path", "rel_path", "aile", "sonek"]], on="path", how="left")
        yakin = (sizinti[("test", "phash")] | sizinti[("test", "derin")]) if sizinti else set()
        p["alt_kume"] = "temiz"
        p.loc[p["rel_path"].isin(yakin), "alt_kume"] = "yakin_kopya"
        p.loc[p["rel_path"].isin(birebir), "alt_kume"] = "birebir_kopya"
        dogru = p["label"] == p["tahmin"]
        k = dict(kosu=kosu, n=len(p), acc=accuracy_score(p["label"], p["tahmin"]),
                 macro_f1=f1_score(p["label"], p["tahmin"], average="macro", labels=SINIFLAR, zero_division=0),
                 mcc=matthews_corrcoef(p["label"], p["tahmin"]))
        for ad in ("birebir_kopya", "yakin_kopya", "temiz"):
            q = p[p["alt_kume"] == ad]
            k[f"n_{ad}"] = len(q)
            k[f"acc_{ad}"] = accuracy_score(q["label"], q["tahmin"]) if len(q) else None
            k[f"f1_{ad}"] = (f1_score(q["label"], q["tahmin"], average="macro", labels=SINIFLAR, zero_division=0)
                             if len(q) else None)
        q = p[p["alt_kume"] != "birebir_kopya"]
        k["acc_birebir_haric"] = accuracy_score(q["label"], q["tahmin"])
        k["f1_birebir_haric"] = f1_score(q["label"], q["tahmin"], average="macro", labels=SINIFLAR, zero_division=0)
        if celiski:
            q = p[p["rel_path"].isin(celiski)]
            k["n_etiket_celiskili"] = len(q)
            k["acc_etiket_celiskili"] = accuracy_score(q["label"], q["tahmin"]) if len(q) else None
            q = p[(p["alt_kume"] == "birebir_kopya") & ~p["rel_path"].isin(celiski)]
            k["n_birebir_tutarli"] = len(q)
            k["acc_birebir_tutarli"] = accuracy_score(q["label"], q["tahmin"]) if len(q) else None
            q = p[~p["rel_path"].isin(celiski)]
            k["acc_celiski_haric"] = accuracy_score(q["label"], q["tahmin"])
            k["f1_celiski_haric"] = f1_score(q["label"], q["tahmin"], average="macro", labels=SINIFLAR, zero_division=0)
        kayitlar.append(k)
        for (lab, a), q in p.groupby(["label", "aile"]):
            aile_kayit.append(dict(kosu=kosu, label=lab, aile=a, n=len(q), dogru=int((q["label"] == q["tahmin"]).sum()),
                                   en_sik_yanlis=(q.loc[q["label"] != q["tahmin"], "tahmin"].value_counts().idxmax()
                                                  if (q["label"] != q["tahmin"]).any() else "")))
        for (g, t), n in p[~dogru].groupby(["label", "tahmin"]).size().items():
            karisma.append(dict(kosu=kosu, gercek=g, tahmin=t, n=int(n)))

    if kayitlar:
        kd = pd.DataFrame(kayitlar).round(4)
        kd.to_csv(cikti / "model_sizinti_ayristirma.csv", index=False, encoding="utf-8-sig")
        rapor += ["## 3. Model başına test: tümü / eğitimde birebir kopyası olan / yalnız yakın kopyası olan / temiz\n",
                  md_tablo(kd, index=False), ""]
        ad = pd.DataFrame(aile_kayit)
        ad["acc"] = (ad["dogru"] / ad["n"]).round(3)
        piv = ad.pivot_table(index=["label", "aile"], columns="kosu", values="acc")
        nn = ad.groupby(["label", "aile"])["n"].first()
        piv.insert(0, "n", nn)
        piv.to_csv(cikti / "model_aile_dogruluk.csv", encoding="utf-8-sig")
        rapor += ["### Aile bazında test doğruluğu (model sütunları)\n", md_tablo(piv), ""]
        kr = pd.DataFrame(karisma)
        if len(kr):
            top = (kr.groupby(["gercek", "tahmin"])["n"].agg(["sum", "mean"]).sort_values("sum", ascending=False)
                   .rename(columns={"sum": "toplam_hata(tum_modeller)", "mean": "model_basina_ort"}).round(1))
            top.to_csv(cikti / "karisan_siniflar.csv", encoding="utf-8-sig")
            rapor += ["### En sık karışan sınıf çiftleri (gerçek → tahmin, tüm modellerin toplamı)\n",
                      md_tablo(top.head(10)), ""]

    # ------------------------------------------------------------ 4. belirsizlik: bootstrap %95 GA + McNemar
    import numpy as np
    from scipy.stats import binomtest
    tahminler = {}
    for f in sorted(glob.glob(str(Path(args.runs) / "*" / "test_predictions.csv"))):
        kosu = Path(f).parent.name
        if not kosu.startswith("SMOKE_"):
            tahminler[kosu] = pd.read_csv(f).sort_values("path").reset_index(drop=True)
    if tahminler:
        rng = np.random.default_rng(0)
        ilk = next(iter(tahminler.values()))
        n = len(ilk)
        idx = rng.integers(0, n, size=(2000, n))
        ga = []
        for kosu, p in tahminler.items():
            y, t = p["label"].to_numpy(), p["tahmin"].to_numpy()
            dogru = (y == t)
            accs = dogru[idx].mean(1)
            f1s = np.array([f1_score(y[i], t[i], average="macro", labels=SINIFLAR, zero_division=0) for i in idx[:1000]])
            ga.append(dict(kosu=kosu, acc=dogru.mean(), acc_ga_alt=np.percentile(accs, 2.5), acc_ga_ust=np.percentile(accs, 97.5),
                           macro_f1=f1_score(y, t, average="macro", labels=SINIFLAR, zero_division=0),
                           f1_ga_alt=np.percentile(f1s, 2.5), f1_ga_ust=np.percentile(f1s, 97.5)))
        # Küme bootstrap'ı: test kendi içinde kopyalar içerdiğinden (375 dosya / 287 tekil içerik) i.i.d. bootstrap
        # fazla iyimser. Küme = manifestteki group_id_phash (pHash<=4 bileşenleri; birebir kopyaları kapsar).
        yol2grup = dict(zip(m["path"], m["group_id_phash"]))
        gruplar_ = ilk["path"].map(yol2grup).to_numpy()
        tekil_gruplar = np.unique(gruplar_)
        uyeler = [np.flatnonzero(gruplar_ == g) for g in tekil_gruplar]
        kidx = [np.concatenate([uyeler[j] for j in rng.integers(0, len(uyeler), size=len(uyeler))]) for _ in range(2000)]
        for satir in ga:
            p = tahminler[satir["kosu"]]
            dogru = (p["label"] == p["tahmin"]).to_numpy()
            accs = np.array([dogru[i].mean() for i in kidx])
            satir["acc_kume_ga_alt"] = np.percentile(accs, 2.5)
            satir["acc_kume_ga_ust"] = np.percentile(accs, 97.5)
        gad = pd.DataFrame(ga).sort_values("acc", ascending=False).round(4)
        gad.to_csv(cikti / "bootstrap_ga.csv", index=False, encoding="utf-8-sig")
        rapor += ["## 4. Belirsizlik (test, n=%d): bootstrap %%95 güven aralığı (2000 yeniden örnekleme; F1 için 1000)\n" % n,
                  f"`acc_ga_*`: i.i.d. bootstrap (iyimser). `acc_kume_ga_*`: {len(uyeler)} pHash kopya kümesi üzerinden "
                  "küme bootstrap'ı (önerilen).\n", md_tablo(gad, index=False), ""]
        en_iyi = gad.iloc[0]["kosu"]
        # McNemar iki biçimde: tüm 375 dosya ve MD5'e göre tekilleştirilmiş test (aynı baytlar aynı tahmini alır)
        tekil_maske = np.ones(n, dtype=bool)
        if md5_dosya.exists():
            rel2md5 = dict(zip(h["rel_path"], h["md5"]))
            path2rel = dict(zip(m["path"], m["rel_path"]))
            md5ler = ilk["path"].map(path2rel).map(rel2md5)
            tekil_maske = ~md5ler.duplicated().to_numpy()
        mc = []
        for kosu, p in tahminler.items():
            if kosu == en_iyi:
                continue
            a = (tahminler[en_iyi]["label"] == tahminler[en_iyi]["tahmin"]).to_numpy()
            b = (p["label"] == p["tahmin"]).to_numpy()
            n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
            pval = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
            u01, u10 = int((a & ~b & tekil_maske).sum()), int((~a & b & tekil_maske).sum())
            upval = binomtest(min(u01, u10), u01 + u10, 0.5).pvalue if (u01 + u10) else 1.0
            mc.append(dict(karsilastirma=f"{en_iyi} vs {kosu}", yalniz_ilki_dogru=n01, yalniz_ikincisi_dogru=n10,
                           mcnemar_kesin_p=round(pval, 5), tekil_n=int(tekil_maske.sum()),
                           tekil_yalniz_ilki=u01, tekil_yalniz_ikincisi=u10, tekil_mcnemar_p=round(upval, 5)))
        if mc:
            rapor += [f"### McNemar kesin testi (en iyi koşu: {en_iyi})\n", md_tablo(pd.DataFrame(mc), index=False), ""]
            pd.DataFrame(mc).to_csv(cikti / "mcnemar.csv", index=False, encoding="utf-8-sig")

    (cikti / "ek_analiz.md").write_text("\n".join(rapor), encoding="utf-8")
    print("\n".join(rapor))
    print(f"\nKaydedildi: {cikti.resolve()}")


if __name__ == "__main__":
    main()
