"""Adım 7 (ek) - Sağlamlık analizleri: alt kümeler, McNemar, termal görüntüler ve model seçimi etkisi.

Resmî protokolü DEĞİŞTİRMEZ. Mevcut tahminleri yeniden okur; --valid-termal verilirse her koşunun seçilen
checkpoint'ini (best.pt) Valid kümesinde yeniden çalıştırır (yalnız çıkarım).

Alt kümeler (test, n=375):
  celiskili (15)   : eğitimde birebir (MD5) kopyası olan ve bu kopyalardan en az biri farklı etiketli dosyalar
  tutarli_tekil    : çelişkili 15 çıkarıldıktan sonra MD5'e göre tekilleştirilmiş test (her içerikten bir dosya)
  sizintisiz       : Train ∪ Valid'de ne MD5 kopyası ne de dihedral pHash (8 dönüşüm, Hamming <= 4) yakın kopyası olan
                     test dosyaları (pHash'ler dogrulama/critic/k01_hashes.csv'den; uzaklık burada yeniden hesaplanır)

Kullanım:
    python 07_saglamlik_analizi.py [--valid-termal]
Çıktılar: denetim_ek/saglamlik.md ve saglamlik_*.csv
"""
import argparse
import glob
import json
import ntpath
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

from ortak import load_rgb, utf8_konsol

SINIFLAR = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
EN_IYI = "official_vit_base_patch14_dinov2_224_s42"
TERMAL = re.compile(r"^C\d+\.")


def kisa(k):
    return (k.replace("official_", "").replace("_224_s42", "").replace("_patch4_window7_224", "")
            .replace("vit_base_patch14_", "").replace("tf_", ""))


def mcnemar(a, b):
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(min(n01, n10), n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    return n01, n10, p


def md_tablo(df, index=False):
    return "```\n" + df.to_string(index=index) + "\n```"


def main():
    utf8_konsol()
    ap = argparse.ArgumentParser()
    ap.add_argument("--valid-termal", action="store_true", help="best.pt'leri Valid'de çalıştır (GPU önerilir)")
    args = ap.parse_args()
    cikti = Path("denetim_ek")
    rapor = ["# Sağlamlık analizleri (07_saglamlik_analizi.py)\n", "Resmî protokol değiştirilmedi.\n"]

    m = pd.read_csv("denetim_final/manifest.csv")
    h = pd.read_csv("denetim_ek/ham_kaynak_eslesme.csv")[["rel_path", "split", "label", "md5"]]
    p2r = dict(zip(m["path"], m["rel_path"]))
    r2m = dict(zip(h["rel_path"], h["md5"]))

    # --- çelişkili 15
    tr_lab = h[h["split"] == "train"].groupby("md5")["label"].agg(set)
    te = h[h["split"] == "test"]
    celiskili = set(te.loc[te.apply(lambda r: bool(r["md5"] in tr_lab.index and (tr_lab[r["md5"]] - {r["label"]})),
                                    axis=1), "rel_path"])

    # --- sızıntısız (MD5 ∪ dihedral pHash <= 4, Train ∪ Valid)
    k = pd.read_csv("dogrulama/critic/k01_hashes.csv", dtype=str, keep_default_na=False)
    ph = np.stack([np.array([int(x, 16) for x in k[f"ph{i}"]], dtype=np.uint64) for i in range(8)], 1)
    test_i = np.flatnonzero(k["split"].to_numpy() == "test")
    ref_i = np.flatnonzero(k["split"].to_numpy() != "test")
    D = np.full((len(test_i), len(ref_i)), 64, dtype=np.uint8)
    for t in range(8):
        D = np.minimum(D, np.bitwise_count(ph[test_i, 0][:, None] ^ ph[ref_i, t][None, :]).astype(np.uint8))
        D = np.minimum(D, np.bitwise_count(ph[test_i, t][:, None] ^ ph[ref_i, 0][None, :]).astype(np.uint8))
    md5_ref = set(k["md5"].to_numpy()[ref_i])
    yakin = (D <= 4).any(1) | np.isin(k["md5"].to_numpy()[test_i], list(md5_ref))
    sizintisiz = set(k["rel_path"].to_numpy()[test_i][~yakin])

    # --- tahminler
    P = {}
    for f in sorted(glob.glob("runs/official_*/test_predictions.csv")):
        kosu = ntpath.basename(ntpath.dirname(f))
        d = pd.read_csv(f).sort_values("path").reset_index(drop=True)
        d["rel"] = d["path"].map(p2r)
        d["md5"] = d["rel"].map(r2m)
        d["dogru"] = d["label"] == d["tahmin"]
        P[kosu] = d
    b = P[EN_IYI]
    maske = {
        "tum_test": np.ones(len(b), bool),
        "celiskili": b["rel"].isin(celiskili).to_numpy(),
        "tutarli_tekil": (~b["rel"].isin(celiskili) & ~b["md5"].duplicated()).to_numpy(),
        "sizintisiz": b["rel"].isin(sizintisiz).to_numpy(),
        "sizintisiz_tekil": (b["rel"].isin(sizintisiz) & ~b["md5"].duplicated()).to_numpy(),
    }
    tanim = pd.DataFrame([dict(alt_kume=a, n=int(v.sum()),
                               sinif_dagilimi=b.loc[v, "label"].value_counts().reindex(SINIFLAR, fill_value=0).to_dict())
                          for a, v in maske.items()])
    rapor += ["## 1. Alt kümeler\n", md_tablo(tanim), ""]

    satir = []
    for kosu, d in P.items():
        r = dict(kosu=kisa(kosu))
        for a, v in maske.items():
            r[f"acc_{a}"] = round(float(d.loc[v, "dogru"].mean()), 4)
        satir.append(r)
    acc = pd.DataFrame(satir)
    acc.to_csv(cikti / "saglamlik_alt_kume_dogruluk.csv", index=False, encoding="utf-8-sig")
    rapor += ["## 2. Alt kümelerde doğruluk\n", md_tablo(acc), ""]

    mc = []
    for kosu, d in P.items():
        if kosu == EN_IYI:
            continue
        for a in ("tum_test", "tutarli_tekil", "sizintisiz_tekil"):
            v = maske[a]
            n01, n10, p = mcnemar(b["dogru"].to_numpy()[v], d["dogru"].to_numpy()[v])
            mc.append(dict(karsilastirma=f"DINOv2 vs {kisa(kosu)}", alt_kume=a, n=int(v.sum()),
                           yalniz_dinov2=n01, yalniz_diger=n10, p=round(p, 4)))
    mcd = pd.DataFrame(mc)
    mcd.to_csv(cikti / "saglamlik_mcnemar.csv", index=False, encoding="utf-8-sig")
    rapor += ["## 3. McNemar kesin testi (DINOv2'ye karşı)\n",
              "Not: az sayıda uyuşmaz çift = düşük güç; anlamsız p eşdeğerlik kanıtı değildir.\n", md_tablo(mcd), ""]

    # --- termal test görüntüleri
    te_rows = []
    for kosu, d in P.items():
        t = d[d["rel"].map(lambda r: bool(TERMAL.match(r.split("/")[-1])))]
        te_rows.append(dict(kosu=kisa(kosu), n=len(t), SF_dedi=int((t["tahmin"] == "Structural_Fault").sum()),
                            SO_dedi=int((t["tahmin"] == "Surface_Obstruction").sum()),
                            p_SF_min=round(t["p_Structural_Fault"].min(), 3), p_SF_max=round(t["p_Structural_Fault"].max(), 3)))
    tt = pd.DataFrame(te_rows)
    tt.to_csv(cikti / "saglamlik_termal_test.csv", index=False, encoding="utf-8-sig")
    rapor += ["## 4. Test'teki 11 termal Structural görüntüsü (eğitimde 1 SF + 2 SO birebir kopyalı)\n", md_tablo(tt), ""]

    # --- Valid termal: seçilen checkpoint ne diyor? (model seçimi etkisi)
    if args.valid_termal:
        import timm
        import torch
        from torchvision import transforms
        cihaz = "cuda" if torch.cuda.is_available() else "cpu"
        va = m[m["split"] == "val"].sort_values("path").reset_index(drop=True)
        va_termal = va["dosya"].map(lambda f: bool(TERMAL.match(f))).to_numpy()
        vrows = []
        for kosu in P:
            cfg_ = json.loads((Path("runs") / kosu / "config.json").read_text(encoding="utf-8"))
            res = json.loads((Path("runs") / kosu / "results.json").read_text(encoding="utf-8"))
            kw = {"img_size": cfg_["img_size"]} if any(s in cfg_["model"] for s in ("vit", "dinov2", "deit", "eva")) else {}
            model = timm.create_model(cfg_["model"], pretrained=False, num_classes=len(cfg_["siniflar"]), **kw)
            model.load_state_dict(torch.load(Path("runs") / kosu / "best.pt", map_location="cpu"))
            model = model.to(cihaz).eval()
            dc = timm.data.resolve_data_config({}, model=model)
            S = cfg_["img_size"]
            tf = transforms.Compose([transforms.Resize((S, S)), transforms.ToTensor(),
                                     transforms.Normalize(dc["mean"], dc["std"])])
            olas = []
            with torch.no_grad():
                for i in range(0, len(va), 32):
                    x = torch.stack([tf(load_rgb(p, S)) for p in va["path"].iloc[i:i + 32]]).to(cihaz)
                    with torch.autocast(device_type=cihaz, dtype=torch.bfloat16, enabled=(cihaz == "cuda")):
                        olas.append(model(x).float().softmax(1).cpu().numpy())
            olas = np.concatenate(olas)
            sin = cfg_["siniflar"]
            tah = np.array(sin)[olas.argmax(1)]
            y = va["label"].to_numpy()
            f1 = f1_score(y, tah, average="macro", labels=sin, zero_division=0)
            alt = tah.copy()
            ters = "Surface_Obstruction" if (tah[va_termal] == "Structural_Fault").mean() >= 0.5 else "Structural_Fault"
            alt[va_termal] = ters
            f1_alt = f1_score(y, alt, average="macro", labels=sin, zero_division=0)
            isf = sin.index("Structural_Fault")
            vrows.append(dict(kosu=kisa(kosu), secilen_epoch=res["best_epoch"],
                              val_f1_kayitli=round(res["best_val_macro_f1"], 4), val_f1_yeniden=round(f1, 4),
                              valid_termal_n=int(va_termal.sum()),
                              termal_SF_dedi=int((tah[va_termal] == "Structural_Fault").sum()),
                              p_SF_min=round(float(olas[va_termal, isf].min()), 3),
                              p_SF_max=round(float(olas[va_termal, isf].max()), 3),
                              termal_tersi_olsaydi_val_f1=round(f1_alt, 4), ters_etiket=ters))
            del model
            if cihaz == "cuda":
                torch.cuda.empty_cache()
        vt = pd.DataFrame(vrows)
        vt.to_csv(cikti / "saglamlik_valid_termal.csv", index=False, encoding="utf-8-sig")
        rapor += ["## 5. Valid'deki termal Structural görüntüleri ve model seçimi\n",
                  "Seçilen checkpoint (best.pt) Valid'de yeniden çalıştırıldı. `termal_tersi_olsaydi_val_f1`: termal "
                  "görüntülerin tahmini tersine çevrilseydi doğrulama makro-F1'i.\n", md_tablo(vt), ""]

    (cikti / "saglamlik.md").write_text("\n".join(rapor), encoding="utf-8")
    print("\n".join(rapor))


if __name__ == "__main__":
    main()
