"""Adım 13 - İnsan kontrolü ile yapay zekâ kararlarının uyumu (kullanıcı formu doldurdu, 20 Kasım).
Girdi: insan_kontrol/insan_kararlari.csv (İndirilenler'den kopya; tur,id,karar,not) + kontrol_formu.csv + efsf_formu.csv.
YZ kararı eşlemesi: same_scene/ayni_sahne -> ayni_sahne, ayni_foto -> ayni_foto, unsure/belirsiz -> belirsiz,
ilgisiz -> ilgisiz. İkili: ilişkili (foto|sahne) / ilgisiz; insanın emin_degil'i ikili ölçümlerden çıkarılır.
GA: Wilson %95. Çıktı: insan_kontrol/sonuc/ (cift_karsilastirma.csv, efsf_karsilastirma.csv, ozet.txt)."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

IK = Path("insan_kontrol")
OUT = IK / "sonuc"
OUT.mkdir(exist_ok=True)
H = pd.read_csv(IK / "insan_kararlari.csv", encoding="utf-8-sig", dtype=str).fillna("")
K = pd.read_csv(IK / "kontrol_formu.csv", encoding="utf-8-sig")
E = pd.read_csv(IK / "efsf_formu.csv", encoding="utf-8-sig")
Hc = H[H["tur"] == "cift"].set_index("id")
He = H[H["tur"] == "efsf"].set_index("id")
ES = {"same_scene": "ayni_sahne", "same_photo": "ayni_foto", "unsure": "belirsiz"}
K["yz"] = K["yz_karari"].str.split(" ").str[0].replace(ES)
K["yz_kesinlik"] = K["yz_karari"].str.extract(r"\((\w+)\)")[0].fillna("hakem")
K["insan"] = K["no"].map(Hc["karar"]).fillna("")
K["insan_not"] = K["no"].map(Hc["not"]).fillna("")
K.drop(columns=["insan_karari", "not"]).to_csv(OUT / "cift_karsilastirma.csv", index=False, encoding="utf-8-sig")
iki = {"ayni_foto": "iliskili", "ayni_sahne": "iliskili", "ilgisiz": "ilgisiz"}


def wilson(k, n):
    if n == 0:
        return "-"
    z, ph = 1.959964, k / n
    m, w = (ph + z * z / (2 * n)) / (1 + z * z / n), z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    lo, hi = m - w, m + w
    return f"{k}/{n} = {k / n:.1%} (GA {lo:.1%}-{hi:.1%})"


s = [f"Doldurulan: çift {(K['insan'] != '').sum()}/{len(K)}, EF/SF {(He['karar'] != '').sum()}/{len(E)}", ""]
for b in ["A", "B"]:
    s += [f"== Bölüm {b}: YZ kararı (satır) × insan kararı (sütun) ==",
          pd.crosstab(K.loc[K.bolum == b, "yz_karari"], K.loc[K.bolum == b, "insan"].replace("", "(boş)"),
                      margins=True).to_string(), ""]
B = K[K.bolum == "B"]
neg = B[B.yz == "ilgisiz"]
neg_d = neg[neg.insan.isin(iki)]
s.append("Kaçırılan bağ (YZ 'ilgisiz-kesin' dediği, insanın ilişkili dediği): "
         + wilson(int((neg_d.insan != "ilgisiz").sum()), len(neg_d)))
poz = B[B.yz.isin(["ayni_foto", "ayni_sahne"])]
poz_d = poz[poz.insan.isin(iki)]
s.append("YZ ilişkili kararlarının insanca doğrulanması (kesinlik): "
         + wilson(int((poz_d.insan != "ilgisiz").sum()), len(poz_d)))
d = B[B.yz.isin(iki) & B.insan.isin(iki)]
if d["insan"].map(iki).nunique() > 1 and d["yz"].map(iki).nunique() > 1:
    s.append(f"İkili Cohen kappa (B, emin_degil ve belirsiz hariç, n={len(d)}): "
             f"{cohen_kappa_score(d.yz.map(iki), d.insan.map(iki)):.3f}")
d3 = B[B.yz.isin(iki) & B.insan.isin(iki)]
s.append("Üç sınıf tam uyum (foto/sahne/ilgisiz, B): "
         + wilson(int((d3.yz == d3.insan).sum()), len(d3)))
bel = K[K.yz == "belirsiz"]
s.append(f"YZ 'belirsiz/unsure' çiftleri (A+B, n={len(bel)}) için insan: "
         + ", ".join(f"{k} {v}" for k, v in bel.insan.replace("", "(boş)").value_counts().items()))
A = K[K.bolum == "A"]
A_d = A[A.insan.isin(iki)]
s.append("Bölüm A (r4 denetiminin hakemli sızıntı/belirsiz çiftleri) insanın ilişkili bulduğu: "
         + wilson(int((A_d.insan != "ilgisiz").sum()), len(A_d)))
E["uzman"] = E["dosya"].map(He["karar"]).fillna("")
E["uzman_not"] = E["dosya"].map(He["not"]).fillna("")
E["uyum"] = np.where(E.uzman == E.mevcut_etiket, "ayni", np.where(E.uzman.isin(["", "emin_degil"]), "-", "FARKLI"))
E.drop(columns=["uzman_etiketi", "not_"]).to_csv(OUT / "efsf_karsilastirma.csv", index=False, encoding="utf-8-sig")
s += ["", "== EF/SF ==", E[["dosya", "mevcut_etiket", "kat_v1", "uzman", "uyum", "uzman_not"]].to_string(index=False)]
n_not = (K.insan_not != "").sum()
if n_not:
    s += ["", "== İnsan notları (çiftler) ==", K.loc[K.insan_not != "", ["no", "yz_karari", "insan", "insan_not"]]
          .to_string(index=False)]
(OUT / "ozet.txt").write_text("\n".join(s), encoding="utf-8")
print("\n".join(s))
