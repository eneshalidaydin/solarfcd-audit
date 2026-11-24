"""Adım 1 - Veri denetimi ve sızıntı analizi.

Ne yapar:
  1. Klasör ağacını gezer; her görüntü için bölme (train/val/test) ve sınıfı klasör adlarından çıkarır.
  2. Kökteki CSV/XLSX meta dosyalarının sütunlarını gösterir; dosya adıyla eşleşirse manifeste ekler.
  3. Her görüntü için çevirmeye dayanıklı algısal özet (pHash) hesaplar.
  4. (--derin) Önceden eğitilmiş bir modelin özniteliklerini çıkarıp kosinüs benzerliğiyle yakın kopyaları bulur.
     Bu yöntem döndürülmüş/gürültü eklenmiş artırma kopyalarını da yakalar.
  5. Bölmeler arası yakın kopyaları (test-eğitim, doğrulama-eğitim) raporlar, örnek çiftleri görselleştirir.
  6. Yakın kopyaları aynı gruba toplayıp grup bazlı 5 katlı bölme (grouped_fold) üretir: sızıntısız protokol.

Kullanım:
    python 01_veri_denetimi.py --kok "D:\\solarfcd\\data\\SolarFCD-Final" --cikti denetim_final --derin
Çıktılar (--cikti klasöründe):
    manifest.csv, sayim_bolme_sinif.csv, capraz_yakin_kopyalar.csv, yakin_kopya_ornekleri.png,
    meta_dosyalari.txt, ozet.json
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ortak import infer_label, infer_split, list_images, load_rgb, to_bool, utf8_konsol

POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ---------------------------------------------------------------- görüntü bilgisi + pHash
def goruntu_bilgisi(path):
    from PIL import Image, ImageOps
    import imagehash
    try:
        with Image.open(path) as im:
            w, h = im.size
            mode, fmt = im.mode, im.format
            if fmt == "JPEG":
                im.draft("L", (256, 256))
            g = im.convert("L")
        varyantlar = [g, ImageOps.mirror(g), ImageOps.flip(g), ImageOps.flip(ImageOps.mirror(g))]
        hashes = [int(str(imagehash.phash(v)), 16) for v in varyantlar]
        return dict(width=w, height=h, mode=mode, fmt=fmt, phash=min(hashes), ok=True, hata="")
    except Exception as e:  # bozuk dosya
        return dict(width=0, height=0, mode="", fmt="", phash=0, ok=False, hata=str(e)[:200])


def hamming_kenarlari(hashes, esik, parca=512):
    """Tüm çiftler arasında Hamming uzaklığı <= esik olanları döndürür: (i, j, uzaklik)."""
    H = np.asarray(hashes, dtype=np.uint64)
    n = len(H)
    kenarlar = []
    for i in range(0, n, parca):
        x = np.bitwise_xor(H[i:i + parca, None], H[None, :])
        d = POP8[x.view(np.uint8)].reshape(x.shape[0], n, 8).sum(-1)
        ii, jj = np.nonzero(d <= esik)
        gi = ii + i
        m = jj > gi
        for a, b, dd in zip(gi[m], jj[m], d[ii[m], jj[m]]):
            kenarlar.append((int(a), int(b), int(dd)))
    return kenarlar


# ---------------------------------------------------------------- derin öznitelik benzerliği
class YolDS:
    """Windows'ta DataLoader işçileri için modül düzeyinde (pickle edilebilir) veri kümesi."""

    def __init__(self, yollar, tf, img):
        self.yollar, self.tf, self.img = yollar, tf, img

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        import torch
        try:
            return self.tf(load_rgb(self.yollar[i], self.img)), i
        except Exception:
            return torch.zeros(3, self.img, self.img), i


def derin_oznitelikler(yollar, model_adi, batch, img, onegitimli=True, workers=4):
    import torch
    import timm
    from torch.utils.data import DataLoader
    from torchvision import transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(model_adi, pretrained=onegitimli, num_classes=0).to(device).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    tf = transforms.Compose([transforms.Resize((img, img)), transforms.ToTensor(),
                             transforms.Normalize(cfg["mean"], cfg["std"])])
    dl = DataLoader(YolDS(yollar, tf, img), batch_size=batch, shuffle=False, num_workers=workers,
                    pin_memory=(device == "cuda"))
    cikti = np.zeros((len(yollar), model.num_features), dtype=np.float32)
    with torch.no_grad():
        for x, idx in dl:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                f = model(x) + model(torch.flip(x, dims=[3])) + model(torch.flip(x, dims=[2]))
            f = torch.nn.functional.normalize(f.float(), dim=1)
            cikti[idx.numpy()] = f.cpu().numpy()
    return cikti


def kosinus_kenarlari(E, esik, k=10, parca=1024):
    """Karşılıklı en yakın komşu (mutual kNN) + eşik: i, j'nin ilk k komşusunda VE j, i'ninkinde ise
    ve benzerlik >= eşik ise kenar ekler. Birbirine benzeyen farklı panellerin zincirleme
    tek dev gruba bağlanmasını engeller."""
    n = len(E)
    k = max(1, min(k, n - 1))
    komsu = np.zeros((n, k), dtype=np.int64)
    benz = np.zeros((n, k), dtype=np.float32)
    for i in range(0, n, parca):
        s = E[i:i + parca] @ E.T
        s[np.arange(s.shape[0]), np.arange(i, i + s.shape[0])] = -1.0  # kendisi hariç
        idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
        komsu[i:i + s.shape[0]] = idx
        benz[i:i + s.shape[0]] = np.take_along_axis(s, idx, axis=1)
    komsu_kume = [set(r) for r in komsu.tolist()]
    kenarlar = []
    for a in range(n):
        for b, v in zip(komsu[a], benz[a]):
            b = int(b)
            if b > a and v >= esik and a in komsu_kume[b]:
                kenarlar.append((a, b, float(v)))
    return kenarlar


def gruplar(n, kenarlar):
    dsu = DSU(n)
    for a, b, _ in kenarlar:
        dsu.union(a, b)
    _, grup = np.unique([dsu.find(i) for i in range(n)], return_inverse=True)
    return grup


# ---------------------------------------------------------------- birleşim-bul (gruplama)
class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# ---------------------------------------------------------------- meta dosyaları
def meta_dosyalarini_incele(kok, df, rapor_satirlari):
    """CSV/XLSX dosyalarını listeler; dosya adı sütunu bulunursa manifestle birleştirir."""
    kok = Path(kok)
    dosyalar = [p for p in kok.rglob("*") if p.suffix.lower() in (".csv", ".xlsx", ".xls")]
    basename_say = df["dosya"].value_counts()
    tekil = set(basename_say[basename_say == 1].index)
    for f in dosyalar:
        try:
            if f.suffix.lower() == ".csv":
                try:
                    m = pd.read_csv(f, sep=None, engine="python", encoding="utf-8")
                except UnicodeDecodeError:
                    m = pd.read_csv(f, sep=None, engine="python", encoding="latin-1")
            else:
                m = pd.read_excel(f)
        except Exception as e:
            rapor_satirlari.append(f"\n### {f}\nOkunamadı: {e}")
            continue
        rapor_satirlari.append(f"\n### {f.relative_to(kok)}  ({m.shape[0]} satır x {m.shape[1]} sütun)")
        rapor_satirlari.append("Sütunlar: " + ", ".join(map(str, m.columns)))
        rapor_satirlari.append(m.head(8).to_string())
        # dosya adı içeren sütunu bul
        en_iyi, oran_en = None, 0.0
        for c in m.columns:
            vals = m[c].dropna().astype(str)
            if len(vals) == 0:
                continue
            adlar = vals.map(lambda v: Path(v.replace("\\", "/")).name)
            oran = adlar.isin(set(df["dosya"])).mean()
            if oran > oran_en:
                en_iyi, oran_en = c, oran
        if en_iyi is None or oran_en < 0.5:
            rapor_satirlari.append("-> Dosya adı sütunu bulunamadı; manifeste eklenmedi.")
            continue
        m = m.copy()
        m["_dosya"] = m[en_iyi].astype(str).map(lambda v: Path(v.replace("\\", "/")).name)
        m = m[m["_dosya"].isin(tekil)].drop_duplicates("_dosya")
        on_ek = "meta_" + f.stem[:20] + "_"
        ekle = m.drop(columns=[en_iyi]).rename(columns=lambda c: c if c == "_dosya" else on_ek + str(c))
        df = df.merge(ekle, how="left", left_on="dosya", right_on="_dosya").drop(columns=["_dosya"])
        rapor_satirlari.append(f"-> '{en_iyi}' sütunu ile eşleşti (%{100 * oran_en:.0f}); "
                               f"{len(ekle.columns) - 1} sütun '{on_ek}*' önekiyle eklendi.")
    if not dosyalar:
        rapor_satirlari.append("Kök klasörde CSV/XLSX meta dosyası bulunamadı.")
    return df


# ---------------------------------------------------------------- görselleştirme
def ciftleri_ciz(df, ciftler, yol, en_fazla=16):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ciftler = ciftler[:en_fazla]
    if not ciftler:
        return
    n = len(ciftler)
    fig, axes = plt.subplots(n, 2, figsize=(6, 2.6 * n))
    if n == 1:
        axes = np.array([axes])
    for r, c in enumerate(ciftler):
        for k, idx in enumerate((c["i"], c["j"])):
            ax = axes[r, k]
            try:
                ax.imshow(load_rgb(df.at[idx, "path"], 256))
            except Exception:
                pass
            baslik = f"{df.at[idx, 'split']} | {df.at[idx, 'label']}\n{df.at[idx, 'dosya'][:28]}"
            if k == 0:
                baslik = f"[{c['olcut']}]\n" + baslik
            ax.set_title(baslik, fontsize=7)
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(yol, dpi=90)
    plt.close(fig)


# ---------------------------------------------------------------- ana akış
def main():
    utf8_konsol()
    ap = argparse.ArgumentParser()
    ap.add_argument("--kok", required=True, help="Veri setinin açıldığı klasör")
    ap.add_argument("--cikti", default="denetim", help="Çıktı klasörü")
    ap.add_argument("--phash-esik", type=int, default=4, help="Yakın kopya için Hamming eşiği (0-64)")
    ap.add_argument("--derin", action="store_true", help="Derin öznitelik benzerliğiyle de yakın kopya ara (GPU önerilir)")
    ap.add_argument("--derin-model", default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--kosinus-esik", type=float, default=0.97)
    ap.add_argument("--derin-onegitimsiz", action="store_true", help=argparse.SUPPRESS)  # yalnız test için
    ap.add_argument("--kat", type=int, default=5, help="Grup bazlı kat sayısı")
    ap.add_argument("--isci", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    args = ap.parse_args()

    t0 = time.time()
    cikti = Path(args.cikti)
    cikti.mkdir(parents=True, exist_ok=True)
    kok = Path(args.kok)

    yollar = list_images(kok)
    if not yollar:
        raise SystemExit(f"'{kok}' altında görüntü bulunamadı.")
    print(f"{len(yollar)} görüntü bulundu. Bilgiler ve pHash hesaplanıyor...")

    satirlar = []
    for p in yollar:
        parts = list(p.relative_to(kok).parts[:-1])
        label, sinif_eslesti = infer_label(parts)
        satirlar.append(dict(path=str(p.resolve()), rel_path=p.relative_to(kok).as_posix(), dosya=p.name,
                             split=infer_split(parts), label=label, sinif_klasorden=sinif_eslesti,
                             kaynak=(parts[0] if parts else "")))
    df = pd.DataFrame(satirlar)

    with ProcessPoolExecutor(max_workers=args.isci) as ex:
        bilgiler = list(ex.map(goruntu_bilgisi, df["path"].tolist(), chunksize=32))
    df = pd.concat([df, pd.DataFrame(bilgiler)], axis=1)
    bozuk = int((~df["ok"]).sum())

    # klasör yapısı özeti
    agac = (df["rel_path"].map(lambda r: "/".join(r.split("/")[:-1])).value_counts().sort_index())
    rapor = ["=== KLASÖR YAPISI (klasör: görüntü sayısı) ==="] + [f"{k or '.'}: {v}" for k, v in agac.items()]
    rapor.append("\n=== META DOSYALARI ===")
    df = meta_dosyalarini_incele(kok, df, rapor)

    # yakın kopyalar
    gecerli = df.index[df["ok"]].to_numpy()
    print("Yakın kopyalar aranıyor (pHash)...")
    kenar_ph = [(gecerli[a], gecerli[b], d) for a, b, d in
                hamming_kenarlari(df.loc[gecerli, "phash"].tolist(), args.phash_esik)]
    kenar_derin = []
    if args.derin:
        print("Derin öznitelikler çıkarılıyor...")
        E = derin_oznitelikler(df.loc[gecerli, "path"].tolist(), args.derin_model, 64, 224,
                               onegitimli=not args.derin_onegitimsiz, workers=args.isci)
        kenar_derin = [(gecerli[a], gecerli[b], s) for a, b, s in kosinus_kenarlari(E, args.kosinus_esik)]

    df["group_id_phash"] = gruplar(len(df), kenar_ph)          # yalnız pHash (daha tutucu)
    df["group_id"] = gruplar(len(df), kenar_ph + kenar_derin)  # pHash + derin (varsayılan)
    df["group_size"] = df.groupby("group_id")["group_id"].transform("size")
    uyarilar = []
    sinir = max(50, int(0.02 * len(df)))
    if df["group_size"].max() > sinir:
        uyarilar.append(f"En büyük grup {int(df['group_size'].max())} görüntü (> {sinir}). Derin eşik gevşek olabilir: "
                        f"--kosinus-esik değerini yükselt (ör. 0.98) ya da eğitimde --group-col group_id_phash kullan.")

    # bölmeler arası çiftler
    capraz = []
    for kaynak_ad, kenarlar in (("phash", kenar_ph), ("derin", kenar_derin)):
        for a, b, v in kenarlar:
            sa, sb = df.at[a, "split"], df.at[b, "split"]
            if sa and sb and sa != sb:
                capraz.append(dict(i=a, j=b, yontem=kaynak_ad, deger=v, split_i=sa, split_j=sb,
                                   label_i=df.at[a, "label"], label_j=df.at[b, "label"],
                                   yol_i=df.at[a, "rel_path"], yol_j=df.at[b, "rel_path"],
                                   olcut=(f"pHash uzaklık={v}" if kaynak_ad == "phash" else f"kosinüs={v:.3f}")))
    capraz_df = pd.DataFrame(capraz)
    if len(capraz_df):
        # önce pHash (küçük uzaklık = daha kesin kopya), sonra derin (yüksek benzerlik önce)
        capraz_df["_anahtar"] = np.where(capraz_df["yontem"] == "phash",
                                         capraz_df["deger"] - 100.0, -capraz_df["deger"])
        capraz_df = capraz_df.sort_values("_anahtar").drop(columns=["_anahtar"])
        capraz_df.drop(columns=["i", "j", "olcut"]).to_csv(cikti / "capraz_yakin_kopyalar.csv",
                                                           index=False, encoding="utf-8-sig")
        ciftleri_ciz(df, capraz_df.to_dict("records"), cikti / "yakin_kopya_ornekleri.png")

    def sizan(hedef):
        hedef_idx = set()
        for c in capraz:
            if {c["split_i"], c["split_j"]} == {hedef, "train"}:
                hedef_idx.add(c["i"] if c["split_i"] == hedef else c["j"])
        return len(hedef_idx)

    # grup bazlı katlar (sızıntısız protokol)
    from sklearn.model_selection import StratifiedGroupKFold
    sub = df[df["ok"]]
    for sutun, grup_sutunu in (("grouped_fold", "group_id"), ("grouped_fold_phash", "group_id_phash")):
        df[sutun] = -1
        kat = min(args.kat, int(sub[grup_sutunu].nunique()))
        if kat < 2:
            uyarilar.append(f"{sutun}: grup sayısı yetersiz, katlar oluşturulamadı.")
            continue
        if kat < args.kat:
            uyarilar.append(f"{sutun}: grup sayısı az olduğu için {kat} kat kullanıldı.")
        sgkf = StratifiedGroupKFold(n_splits=kat, shuffle=True, random_state=42)
        for k, (_, te) in enumerate(sgkf.split(sub, sub["label"], sub[grup_sutunu])):
            df.loc[sub.index[te], sutun] = k

    df["phash"] = df["phash"].map(lambda v: f"{int(v):016x}")
    df.to_csv(cikti / "manifest.csv", index=False, encoding="utf-8-sig")

    sayim = pd.crosstab(df["label"], df["split"].replace("", "(yok)"), margins=True, margins_name="Toplam")
    sayim.to_csv(cikti / "sayim_bolme_sinif.csv", encoding="utf-8-sig")
    (cikti / "meta_dosyalari.txt").write_text("\n".join(rapor), encoding="utf-8")

    ozet = dict(
        kok=str(kok), goruntu=len(df), bozuk=bozuk,
        bolme_bulunamayan=int((df["split"] == "").sum()),
        sinif_klasorden_bulunamayan=int((~df["sinif_klasorden"]).sum()),
        boyut_ornekleri=df.loc[df["ok"], ["width", "height"]].describe().round(0).to_dict(),
        mod_dagilimi=df["mode"].value_counts().to_dict(),
        phash_kenar=len(kenar_ph), derin_kenar=len(kenar_derin),
        grup_sayisi=int(df["group_id"].nunique()), en_buyuk_grup=int(df["group_size"].max()),
        tekil_olmayan_goruntu=int((df["group_size"] > 1).sum()),
        test_egitimle_yakin_kopyasi_olan=sizan("test"),
        dogrulama_egitimle_yakin_kopyasi_olan=sizan("val"),
        grup_boyutu_dagilimi=df["group_size"].value_counts().sort_index().to_dict(),
        uyarilar=uyarilar,
        sure_dk=round((time.time() - t0) / 60, 1),
    )
    (cikti / "ozet.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "\n".join(rapor[: min(len(rapor), 60)]))
    print("\n=== BÖLME x SINIF SAYIMI ===")
    print(sayim.to_string())
    print("\n=== ÖZET ===")
    for k, v in ozet.items():
        if k != "boyut_ornekleri":
            print(f"{k}: {v}")
    print(f"\nÇıktılar: {cikti.resolve()}")


if __name__ == "__main__":
    main()
