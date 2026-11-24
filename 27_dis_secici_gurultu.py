"""Adım 27 - (1) Dış sette seçici sınıflandırma/konformal güvence aktarılıyor mu (kullanıcı onayı 22 Kasım)? Topluluk-3
yüksek (ConvNeXt-224, Swin-224, DINOv2-392; orijinal etiket modelleri, 25'teki dış tahminler). Eşik ve LAC nicel değeri
iç OOF'ta (2.095, orijinal etiket) LTT/Bonferroni (δ=0,1, risk %5) ile belirlenip dış sete doğrudan uygulanır. Yeniden
kalibrasyon: dış setten n_cal = 50/100/200/400 rastgele görüntüyle aynı yöntem, kalan görüntüde test (1.000 tekrar).
Izgara: iç ve ilk analiz %50-%100 (11 düzey, önceden sabit); dışta ek olarak %5-%100 (20 düzey, δ/20), çünkü dış doğruluk
%50 kapsamda bile %95'in altında. Yalnız 4 sınıf etiketi bilinen dış görüntüler. (2) Etiket gürültüsü tahmini: aday değişimleri (tümü denetlendi) + kontrol
oranı × aday olmayan nüfus; Wilson %95 GA (kontrol oranından). Çıktı: asama2_sonuclar/dis_secici.txt"""
import numpy as np
import pandas as pd
from scipy.stats import beta, norm

SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
PC = [f"p_{c}" for c in SIN]
RISK, DEL = 0.05, 0.10
UY = ["convnext_tiny_224", "swin_tiny_patch4_window7_224_224", "vit_base_patch14_dinov2_392"]
D = [pd.concat([pd.read_csv(f"runs_asama2/grouped{k}_{u}_s{s}_kat_v1_asama2_none/test_predictions.csv")
                for k in range(5)]).set_index("path").sort_index() for u in UY for s in (42, 43, 44)]
Pi = np.mean([d[PC].to_numpy() for d in D], 0)
yi = D[0].label.map(SIN.index).to_numpy()


def ust(k, n, d):
    return 1.0 if n == 0 or k == n else beta.ppf(1 - d, k + 1, n - k)


def ltt(P, y, alt=0.50):
    conf = P.max(1)
    o = np.argsort(-conf)
    hata = np.cumsum(P.argmax(1)[o] != y[o])
    esik = 1.01
    G = np.arange(alt, 1.0001, 0.05)
    for c in G:
        i = max(int(c * len(y)) - 1, 0)
        if ust(hata[i], i + 1, DEL / len(G)) <= RISK:
            esik = conf[o][i]
    sk = 1 - P[np.arange(len(y)), y]
    q = np.quantile(sk, min(1, np.ceil((len(y) + 1) * 0.95) / len(y)), method="higher")
    return esik, q


def uygula(P, y, esik, q):
    sec = P.max(1) >= esik
    dg = P.argmax(1) == y
    k = (1 - P) <= q
    return dict(kapsam=sec.mean(), acc=dg[sec].mean() if sec.any() else np.nan,
                ihlal=(1 - dg[sec].mean()) > RISK if sec.any() else False, tam=dg.mean(),
                lac=k[np.arange(len(y)), y].mean(), boy=k.sum(1).mean())


esik, q = ltt(Pi, yi)
s = [f"İç kalibrasyon (2.095): eşik {esik:.4f}, LAC q {q:.4f}; iç uygulama: {uygula(Pi, yi, esik, q)}", ""]
rng = np.random.default_rng(0)
for veri in ["roboflow", "melbourne"]:
    T = pd.read_csv(f"dis_dogrulama/{veri}/ortusme.csv", encoding="utf-8-sig")
    T = T[~T.ortusen].reset_index(drop=True)
    y = T.label.map(lambda v: SIN.index(v) if v in SIN else -1).to_numpy()
    P = np.mean([np.load(f"dis_dogrulama/{veri}/tahminler_{a}.npy") for a in ("ConvNeXt", "Swin", "DINOv2-392")], 0)
    P, y = P[y >= 0], y[y >= 0]
    r = uygula(P, y, esik, q)
    s.append(f"{veri} (n={len(y)}) iç eşik doğrudan: tam acc {r['tam']:.3f} | seçici kapsam {r['kapsam']:.3f}, "
             f"seçilen acc {r['acc']:.3f}, risk ≤%5 sağlandı mı: {not r['ihlal']} | LAC kapsama {r['lac']:.3f} (hedef 0,95), boy {r['boy']:.2f}")
    for nc, alt in [(n, a) for a in (0.50, 0.05) for n in (50, 100, 200, 400)]:
        if nc >= len(y) - 30:
            continue
        R = []
        for _ in range(1000):
            i = rng.permutation(len(y))
            c, t = i[:nc], i[nc:]
            R.append(uygula(P[t], y[t], *ltt(P[c], y[c], alt)))
        R = pd.DataFrame(R)
        s.append(f"   yeniden kalibrasyon (kapsam ızgarası {alt:.2f}-1) n_cal={nc}: kapsam {R.kapsam.mean():.3f} (±{R.kapsam.std():.3f}), seçilen acc "
                 f"{R.acc.mean():.3f}, ihlal oranı {R.ihlal.mean():.3f} | LAC kapsama {R.lac.mean():.3f}, boy {R.boy.mean():.2f}")
    s.append("")
A = pd.read_csv("insan_kontrol/etiket/sonuc.csv", encoding="utf-8-sig")
K = A[~A.aday]
N, na = 2095, int(A.aday.sum())
for ad, olay in [("değişen", ["degisti"]), ("değişen+belirsiz", ["degisti", "belirsiz"])]:
    k, n = int(K.durum.isin(olay).sum()), len(K)
    p, z = k / n, norm.ppf(0.975)
    m = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    ca = int(A[A.aday].durum.isin(olay).sum())
    f = lambda r: (ca + r * (N - na)) / N
    s.append(f"Etiket gürültüsü ({ad}): kontrol {k}/{n} = {p:.3f} [Wilson {m - h:.3f}-{m + h:.3f}]; tüm veri tahmini "
             f"{f(p):.3f} [{f(m - h):.3f}-{f(m + h):.3f}] (aday {ca}/{na} + kontrol oranı × {N - na})")
open("asama2_sonuclar/dis_secici.txt", "w", encoding="utf-8").write("\n".join(s))
