"""Adım 22 - D: garantili seçici sınıflandırma + konformal tahmin kümeleri (kullanıcı kararı 21 Kasım).
Girdi: havuzlanmış OOF olasılıkları (topluluk; --uyeler). Her tekrarda sahne_grup düzeyinde %50 kalibrasyon / %50 test
(1.000 tekrar). (1) Seçici risk kontrolü: güven eşiği, kalibrasyonda hatanın Clopper-Pearson %90 üst sınırı ≤ hedef
risk (%5) olacak en geniş kapsamı veren eşik (Geifman & El-Yaniv SGR'nin tek eşik hâli). Testte kapsam, doğruluk ve
garanti ihlali oranı raporlanır. (2) LAC konformal tahmin kümeleri (%95 kapsama): test kapsaması ve ortalama küme boyu.
Etiket sütunu --etiket ile değiştirilebilir (etiket denetimi sonrası). Çıktı: asama2_sonuclar/secici.txt"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta

SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
PC = [f"p_{c}" for c in SIN]
ap = argparse.ArgumentParser()
ap.add_argument("--uyeler", default="convnext_tiny_384,swin_tiny_patch4_window7_224_224,vit_base_patch14_dinov2_392")
ap.add_argument("--risk", type=float, default=0.05)
ap.add_argument("--delta", type=float, default=0.10)
ap.add_argument("--etiket-dosyasi", default="")
ap.add_argument("--kok", default="runs_asama2")
ap.add_argument("--son", default="kat_v1_asama2_none")
ap.add_argument("--kayit", default="", help="bölme bazında sonuçların CSV yolu (isteğe bağlı)")
ap.add_argument("--cikti", default="asama2_sonuclar/secici.txt")
a = ap.parse_args()
D = [pd.concat([pd.read_csv(f"{a.kok}/grouped{k}_{u}_s{s}_{a.son}/test_predictions.csv")
                for k in range(5)]).set_index("path").sort_index() for u in a.uyeler.split(",") for s in (42, 43, 44)]
P = np.mean([d[PC].to_numpy() for d in D], 0)
lab = D[0]["label"]
if a.etiket_dosyasi:
    e = pd.read_csv(a.etiket_dosyasi).set_index("path")["etiket"]
    lab = lab.index.map(lambda p: e.get(p, lab[p])).to_series(index=lab.index)
tut = lab.isin(SIN).to_numpy()
P, y = P[tut], lab[tut].map(SIN.index).to_numpy()
g = lab.index[tut].map(pd.read_csv("asama2/manifest_asama2.csv").set_index("path")["sahne_grup"]).to_numpy()
gu = np.unique(g)
conf, pred = P.max(1), P.argmax(1)
dogru = pred == y
rng = np.random.default_rng(0)


def ust(k, n, d):
    return 1.0 if n == 0 else (1.0 if k == n else beta.ppf(1 - d, k + 1, n - k))


R = []
for _ in range(1000):
    cal = np.isin(g, rng.choice(gu, len(gu) // 2, replace=False))
    te = ~cal
    o = np.argsort(-conf[cal])
    hata = np.cumsum(~dogru[cal][o])
    n = np.arange(1, cal.sum() + 1)
    # Learn-then-Test, Bonferroni: 11 aday kapsam (%50-%100, adım %5), her biri δ/11 ile sınanır; geçenlerin en genişi.
    # (Sabit sıralı test en emin uçtaki birkaç kendinden emin hatada hemen durduğu için kullanılmadı.)
    son = None
    for c in np.arange(0.50, 1.0001, 0.05):
        i = int(c * len(n)) - 1
        if ust(hata[i], n[i], a.delta / 11) <= a.risk:
            son = i
    esik = conf[cal][o][son] if son is not None else 1.01
    sec = te & (conf >= esik)
    # LAC konformal küme (skor 1 - p_y)
    sk = 1 - P[np.arange(len(y)), y]
    q = np.quantile(sk[cal], min(1, np.ceil((cal.sum() + 1) * 0.95) / cal.sum()), method="higher")
    kume = (1 - P) <= q
    R.append(dict(kapsam=sec.sum() / te.sum(), acc=dogru[sec].mean() if sec.any() else np.nan,
                  ihlal=(1 - dogru[sec].mean()) > a.risk if sec.any() else False, tam_acc=dogru[te].mean(),
                  aps_kapsama=kume[te, y[te]].mean(), aps_boy=kume[te].sum(1).mean(),
                  aps_tekil=(kume[te].sum(1) == 1).mean()))
R = pd.DataFrame(R)
if a.kayit:
    R.to_csv(a.kayit, index=False)
s = [f"Üyeler: {a.uyeler} (3 tohum); n={len(y)}; hedef risk {a.risk:.0%}, δ={a.delta}",
     f"Tam kapsam doğruluk: {R.tam_acc.mean():.4f}",
     f"Seçici: kapsam {R.kapsam.mean():.3f} (±{R.kapsam.std():.3f}), seçilenlerde doğruluk {R.acc.mean():.4f} "
     f"(%5-%95: {R.acc.quantile(.05):.3f}-{R.acc.quantile(.95):.3f}), garanti ihlali oranı {R.ihlal.mean():.3f}",
     f"LAC %95: test kapsaması {R.aps_kapsama.mean():.4f}, ort. küme boyu {R.aps_boy.mean():.3f}, "
     f"tek etiketli küme oranı {R.aps_tekil.mean():.3f}"]
Path(a.cikti).write_text("\n".join(s), encoding="utf-8")
print("\n".join(s))
