"""Adım 23 - 3 uzmanın kör etiketlerini birleştirir (önceden sabit kural, NOTLAR 21 Kasım): ≥2/3 aynı sınıf -> yeni
etiket; aksi hâlde belirsiz (temiz analizde dışarıda). Fleiss kappa, uzman-orijinal uyumu (aday/kontrol ayrı).
Çıktı: insan_kontrol/etiket/sonuc.csv, ozet.txt; asama2_temiz/manifest_asama2.csv (asama2 kopyası: düzeltilen
etiketler yazılır, belirsizlerde kat_v1 = -1; label_orijinal sütunu eklenir). Ham veri değişmez."""
from pathlib import Path

import numpy as np
import pandas as pd

E = Path("insan_kontrol/etiket")
A = pd.read_csv(E / "adaylar.csv", encoding="utf-8-sig")
U = pd.concat([pd.read_csv(E / "yanitlar" / f"etiket_uzman{u}.csv", encoding="utf-8-sig", dtype=str) for u in (1, 2, 3)])
U["karar"] = U["karar"].fillna("bos")
W = U.pivot(index="no", columns="uzman", values="karar")
W.columns = [f"uzman{c}" for c in W.columns]
A = A.merge(W, left_on="no", right_index=True)
SEC = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction", "karisik", "emin_degil"]
uc = A[["uzman1", "uzman2", "uzman3"]]
M = np.stack([(uc == c).sum(1).to_numpy() for c in SEC], 1)


def fleiss(M):
    n = M.sum(1)[0]
    p = M.sum(0) / M.sum()
    P = (M * (M - 1)).sum(1) / (n * (n - 1))
    return (P.mean() - (p ** 2).sum()) / (1 - (p ** 2).sum())


def cogunluk(r):
    v = pd.Series(r).value_counts()
    return v.index[0] if v.iloc[0] >= 2 and v.index[0] in SEC[:4] else "belirsiz"


A["uzlasi"] = [cogunluk(list(r)) for r in uc.itertuples(index=False)]
A["oybirligi"] = uc.nunique(axis=1) == 1
A["durum"] = np.where(A.uzlasi == "belirsiz", "belirsiz", np.where(A.uzlasi == A.label, "ayni", "degisti"))
A.to_csv(E / "sonuc.csv", index=False, encoding="utf-8-sig")
s = [f"Görüntü {len(A)} (aday {A.aday.sum()}, kontrol {(~A.aday).sum()}); boş yanıt: {(uc == 'bos').sum().sum()}",
     f"Fleiss kappa (6 seçenek): {fleiss(M):.3f}; oybirliği oranı {A.oybirligi.mean():.3f}"]
for u in ["uzman1", "uzman2", "uzman3"]:
    s.append(f"{u}: orijinal etiketle uyum aday {(A[A.aday][u] == A[A.aday].label).mean():.3f}, kontrol "
             f"{(A[~A.aday][u] == A[~A.aday].label).mean():.3f}")
for ad, g in [("aday", A[A.aday]), ("kontrol", A[~A.aday])]:
    s.append(f"{ad}: " + ", ".join(f"{k} {v}" for k, v in g.durum.value_counts().items()))
b = A[A.aday & (A.uzlasi != "belirsiz")]
s += ["", "Orijinal (satır) -> uzlaşı (sütun):", pd.crosstab(A.label, A.uzlasi).to_string(), "",
      f"Adaylarda model tahmini = uzlaşı (belirsiz hariç): {(b.model_tahmini == b.uzlasi).mean():.3f}"]
m = pd.read_csv("asama2/manifest_asama2.csv")
yeni = A.set_index("path")["uzlasi"]
deg = m.path.isin(yeni.index)
m["label_orijinal"] = m["label"]
y2 = m.loc[deg, "path"].map(yeni)
m.loc[y2[y2.isin(SEC[:4])].index, "label"] = y2[y2.isin(SEC[:4])]
m.loc[y2[y2 == "belirsiz"].index, "kat_v1"] = -1
Path("asama2_temiz").mkdir(exist_ok=True)
m.to_csv("asama2_temiz/manifest_asama2.csv", index=False)
t = m[m.kat_v1 >= 0]
s.append(f"\nTemiz manifest: {len(t)} değerlendirme görüntüsü; değişen etiket {(t.label != t.label_orijinal).sum()}; "
         f"sınıflar {t.label.value_counts().to_dict()}")
(E / "ozet.txt").write_text("\n".join(s), encoding="utf-8")
print("\n".join(s))
