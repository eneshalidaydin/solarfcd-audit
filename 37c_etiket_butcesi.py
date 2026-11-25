"""Adım 37c - Yeni bir sahada %5'lik hata sınırını yeniden sertifikalamak için gereken etiketli hedef görüntü sayısı
(etiket bütçesi) şekli. Yeni eğitim yok, ham veriye dokunulmaz, yalnız SciPy ile kesin hesap yapılır.

Yöntem (Ek Yöntemler, Denk. S1): Learn-then-Test, 11 aday kapsam düzeyi c = 0,50, 0,55, ..., 1,00, Bonferroni düzeyi
δ/11 (δ = 0,1). Kapsam c, kabul edilen n_kabul kalibrasyon görüntüsündeki e hata için tek yönlü Clopper–Pearson üst
sınırı U = Beta^-1(1 - δ/11; e + 1, n_kabul - e) ≤ α = 0,05 olduğunda sertifikalanır.

Tek ve tutarlı kurallar (çizilen eğriler)
  * n_kabul = ⌊c·n⌋, yani en güvenli ⌊c·n⌋ görüntü kabul edilir (r1_a_ic_alan.py ve k_recal.py'deki int(c * n) ile aynı).
  * e = ⌊r·n_kabul + 1/2⌋, yani kabul edilenlerdeki gerçek hata oranı r iken beklenen hata sayısı en yakın tam sayıya
    (yarım yukarı) yuvarlanır. Hesap tam sayı aritmetiğiyle yapılır (r = b/10000), kayan nokta hatası yoktur.
  * Gereken n, kendisinden büyük her bütçede de Denk. S1'i sağlayan en küçük n'dir (yuvarlamanın yarattığı testere
    dişi etkisine karşı sağlam tanım). Hesap n_kabul uzayında yapılıp n = ⌈n_kabul/c⌉ ile çevrilir.
Karşılaştırma için tutucu kural e = ⌈r·n_kabul⌉ ve "ilk geçiş" tanımı da yazdırılır, çizilmez.
Kontrol: hatasız model (r = 0) %100 kapsam için 92, %50 kapsam için 184 görüntü ister (metin ve Ek Yöntemler).
RF-Fault çizgisi bilerek konmamıştır. Aktarılan SolarFCD eşiğindeki %3,1'lik hata (259 kabulde 8 hata, Tablo 6) yalnız
%23,4 kapsamda ölçülmüştür. E3'ün RF-Fault hatası ızgaranın en düşük düzeyi olan %50 kapsamda bile α'nın çok üstündedir
(betik bunu k_recal.py ile aynı tahminlerden yazdırır), yani bu kapsamlarda hiçbir bütçe sertifika vermez. %3,1'i eğrilerin
üzerine koymak 735–1.470 etiketin yeteceği izlenimini verirdi.
Çıktı: makale/fig_label_budget.png (300 dpi, basılı boyut yaklaşık 11 × 5,6 cm, en küçük yazı 7,5 pt)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from PIL import Image
from scipy.stats import beta, binom

OUT = Path("makale")
C1, C2, C3, C4, GRI = "#3b6ea5", "#d9822b", "#2e8b57", "#c0392b", "#9e9e9e"
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False})
ALFA, DELTA, KDUZEY = 0.05, 0.1, 11
DB = DELTA / KDUZEY
MMAX = 400_000
M = np.arange(1, MMAX + 1, dtype=np.int64)


def ust(e, m):
    """Denk. S1, analiz betiklerindeki ust() ile aynı (e = m ise 1)."""
    e, m = np.broadcast_arrays(np.asarray(e, dtype=np.int64), np.asarray(m, dtype=np.int64))
    u = np.ones(e.shape)
    ok = (e >= 0) & (e < m)
    u[ok] = beta.ppf(1 - DB, e[ok] + 1, m[ok] - e[ok])
    return u


# e_max(m): Denk. S1'i sağlayan en büyük hata sayısı (yoksa -1). Başlangıç Clopper–Pearson/binom ikiliğinden
# [U(e, m) <= α  <=>  binom.cdf(e, m, α) <= δ/11], sonra doğrudan Denk. S1 ile düzeltilip doğrulanır.
emax = binom.ppf(DB, M, ALFA).astype(np.int64) - 1
for _ in range(3):
    emax[ust(emax + 1, M) <= ALFA] += 1
    emax[(emax >= 0) & (ust(np.maximum(emax, 0), M) > ALFA)] -= 1
assert np.all(ust(emax + 1, M) > ALFA) and np.all((emax < 0) | (ust(np.maximum(emax, 0), M) <= ALFA))
assert np.all(np.diff(emax) >= 0)


def hata_sayisi(pay, payda, m, kural):
    """r = pay/payda için e; 'yakin' = ⌊r·m + 1/2⌋ (çizilen), 'tavan' = ⌈r·m⌉ (karşılaştırma)."""
    return (2 * pay * m + payda) // (2 * payda) if kural == "yakin" else -(-pay * m // payda)


def m_gerek(pay, payda, kural="yakin"):
    """(kalıcı, ilk) n_kabul: sonrasında hep sertifikalanan en küçük n_kabul ve ilk sertifikalanan n_kabul."""
    gec = hata_sayisi(pay, payda, M, kural) <= emax
    if not gec.any():
        return np.nan, np.nan
    kal = np.flatnonzero(~gec)
    kalici = kal[-1] + 2 if len(kal) else 1
    return (kalici if kalici <= MMAX else np.nan), int(np.argmax(gec) + 1)


KAPSAM = {20: ("100%", C1), 15: ("75%", C2), 10: ("50%", C3)}  # c = k/20


def n_gerek(m, k):
    return np.nan if np.isnan(m) else int(-(-int(m) * 20 // k))


# ---------- hesap ----------
B = np.arange(0, 491)  # r = %0,00 ... %4,90, 0,01 puan adım
Y = np.array([m_gerek(b, 10000) for b in B], dtype=float)           # [:, 0] kalıcı (çizilen), [:, 1] ilk geçiş
T = np.array([m_gerek(b, 10000, "tavan") for b in B], dtype=float)
N = {k: np.array([n_gerek(m, k) for m in Y[:, 0]], dtype=float) for k in KAPSAM}

assert N[20][0] == 92 and N[10][0] == 184, (N[20][0], N[10][0])
print("r = 0: n =", {KAPSAM[k][0]: int(N[k][0]) for k in KAPSAM}, "| e_max(92) =", emax[91], "e_max(133) =", emax[132],
      "e_max(132) =", emax[131])
print("400 etiketle (kuraldan bağımsız): %100 kapsamda e_max(400) =", emax[399], f"(r ≤ {emax[399] / 400:.2%}),",
      "%50 kapsamda e_max(200) =", emax[199], f"(r ≤ {emax[199] / 200:.2%})")
for k in KAPSAM:
    i = np.flatnonzero(N[k] <= 400)
    print(f"  {KAPSAM[k][0]} kapsam: çizilen n ≤ 400 olan en büyük r = {B[i].max() / 100:.2f}%")
rf = [m_gerek(8, 259, kr) for kr in ("yakin", "tavan")]
print("Yalnız bilgi (şekilde yok): r = 8/259 = %.2f%% için n (çizilen; tavan kalıcı):" % (800 / 259),
      {KAPSAM[k][0]: (n_gerek(rf[0][0], k), n_gerek(rf[1][0], k)) for k in KAPSAM})
# RF-Fault'ta E3'ün kapsam düzeylerindeki hatası (k_recal.py ile aynı dosyalar, yalnız okunur): %3,1 neden çizilmedi
SIN =["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
RB = Path("dis_dogrulama/roboflow")
TT = pd.read_csv(RB / "ortusme.csv", encoding="utf-8-sig")
TT = TT[~TT.ortusen].reset_index(drop=True)
yy = TT.label.map(lambda v: SIN.index(v) if v in SIN else -1).to_numpy()
PP = np.mean([np.load(RB / f"tahminler_{a}.npy") for a in ("ConvNeXt", "Swin", "DINOv2-392")], 0)[yy >= 0]
yy = yy[yy >= 0]
sira = np.argsort(-PP.max(1))
hata = np.cumsum(PP.argmax(1)[sira] != yy[sira])
esik = PP.max(1) >= 0.9072
print(f"RF-Fault E3 ({len(yy)} görüntü): aktarılan eşik 0,9072 -> {esik.sum()} kabul, "
      f"{(PP.argmax(1)[esik] != yy[esik]).sum()} hata (%{esik.mean() * 100:.1f} kapsam)")
for k in range(10, 21):
    mk = int(k / 20 * len(yy))
    print(f"  kapsam %{k * 5}: {mk} kabul, {hata[mk - 1]} hata = %{hata[mk - 1] / mk * 100:.1f}")
assert all(hata[int(k / 20 * len(yy)) - 1] / int(k / 20 * len(yy)) > ALFA for k in range(10, 21))
for b in (25, 50, 100, 150, 200, 300, 400, 450):
    print(f"r = {b / 100:.2f}%: çizilen {[n_gerek(Y[b, 0], k) for k in KAPSAM]}, ilk geçiş "
          f"{[n_gerek(Y[b, 1], k) for k in KAPSAM]}, tavan kalıcı {[n_gerek(T[b, 0], k) for k in KAPSAM]}")
print(f"Kalıcı/ilk geçiş en büyük göreli fark (r > 0): {np.nanmax(Y[1:, 0] / Y[1:, 1] - 1):.1%}; "
      f"tavan/yakın kalıcı: {np.nanmax(T[1:, 0] / Y[1:, 0] - 1):.1%}")
pb = [binom.cdf(emax[int(m) - 1], int(m), b / 10000) for b, m in zip(B[1:], Y[1:, 0]) if not np.isnan(m)]
print(f"Binom(n_kabul, r) hata modelinde çizilen n'de sertifika olasılığı: en az {min(pb):.2f}, "
      f"ortanca {np.median(pb):.2f}, en çok {max(pb):.2f}")
for k in KAPSAM:  # çizilen eğrinin r'ye göre azalmadığını doğrula
    v = N[k][~np.isnan(N[k])]
    assert np.all(np.diff(v) >= 0)

# ---------- şekil ----------
fig, ax = plt.subplots(figsize=(4.45, 2.15))
ax.axhspan(50, 400, color=GRI, alpha=0.22, lw=0, zorder=0)
ax.text(4.93, 141, "Target labels\ntested (50–400)", ha="right", va="center", fontsize=7.5, color="#555555",
        linespacing=1.1)
for k, (et, renk) in KAPSAM.items():
    ax.plot(B / 100, N[k], color=renk, lw=1.4, label=et, zorder=3)
for k, deger in ((20, 92), (10, 184)):  # hatasız model; değerler y ekseninde renkli yazılır
    ax.plot(0, deger, "o", ms=4.5, color=KAPSAM[k][1], zorder=5, clip_on=False)
ax.axvline(5, color=GRI, ls="--", lw=0.8, zorder=1)
ax.text(4.93, 1000, "α = 5%", ha="right", va="center", fontsize=7.5, color="#666666")
ax.set_yscale("log")
ax.set_ylim(40, 3e4)
ax.set_xlim(0, 5.0)
ax.yaxis.set_major_locator(FixedLocator([50, 92, 184, 400, 1000, 2000, 5000, 10000, 20000]))
ax.yaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v)):,}"))
ax.set_xticks(range(6))
ax.tick_params(labelsize=7.5, length=2.5)
fig.canvas.draw()
for et in ax.get_yticklabels():
    renk = {"92": C1, "184": C3}.get(et.get_text())
    if renk:
        et.set_color(renk)
        et.set_fontweight("bold")
ax.set_xlabel("Error rate on accepted images (%)", fontsize=8)
ax.set_ylabel("Target labels needed", fontsize=8)
ax.legend(title="Certified coverage", frameon=False, fontsize=7.5, title_fontsize=7.5, loc="upper left",
          bbox_to_anchor=(0.03, 1.0), handlelength=1.4, labelspacing=0.25, borderaxespad=0.2)
fig.savefig(OUT / "fig_label_budget.png", dpi=300, bbox_inches="tight")

def pdf_kaydet(ad, fig=None):
    """Dergiye ayrı yüklenecek vektörel sürüm (SEKIL_PDF=1 iken), makale/yuksek/<ad>.pdf; PNG çıktısı değişmez."""
    import os
    if not os.environ.get("SEKIL_PDF"):
        return
    (OUT / "yuksek").mkdir(exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42}):
        (fig or plt.gcf()).savefig(OUT / "yuksek" / (Path(ad).stem + ".pdf"), bbox_inches="tight", dpi=600)
pdf_kaydet("fig_label_budget.png", fig)
plt.close(fig)
w, h = Image.open(OUT / "fig_label_budget.png").size
print(f"fig_label_budget.png {w} × {h} px = {w / 300 * 2.54:.2f} × {h / 300 * 2.54:.2f} cm at 300 dpi")
