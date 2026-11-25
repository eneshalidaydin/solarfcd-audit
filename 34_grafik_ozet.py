"""Adım 34 - Grafik özet (Elsevier: en az 531 × 1328 piksel, y × g; TIFF/PDF). Yalnız makaledeki sayılar; üretken YZ görseli yok.
Gönderim öncesi iç değerlendirme (24 Kasım): dış veri dört model noktasıyla, az örnek sonucu gruplu tekrardan (28b) okunur, sağ panelde
%82-91/%57 yerine işletme-bakım çıkarımları (kullanıcı kararı UD3). Çıktı: makale/graphical_abstract.{png,tif,pdf}
Son denetim (25 Kasım): kılavuz grafik özetin 13 cm genişlikte okunmasını istiyor. Tuval 16,9 x 6,7 cm (600 dpi), en küçük
yazı 9 pt, yani 13 cm genişlikte yaklaşık 6,9 pt. Sayılar ve UD3 ifadeleri aynı, yalnız alt bilgideki veri seti satırı çıktı."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
C1, C2, C3, C4, G = "#3b6ea5", "#d9822b", "#2e8b57", "#c0392b", "#9e9e9e"
R = pd.read_csv("asama2_sonuclar/az_ornek_gruplu.csv")
az = R[(R.yontem == "ince_ayar_solarfcd") & (R.butce == sorted(R[R.butce > 0].butce.unique())[1])]  # yaklaşık 110 görüntü
az_m, az_lo, az_hi, az_n = az.acc.mean(), az.acc.min(), az.acc.max(), az.n_egitim.mean()
fig = plt.figure(figsize=(6.64, 2.655), dpi=600)  # 16.9 x 6.7 cm: 9 pt text = 6.9 pt at a width of 13 cm

# Sol: doğruluk merdiveni
fig.text(0.012, 0.975, "What does \"97.6% accuracy\"\nmean for PV inspection?", fontsize=10.5, fontweight="bold", va="top", linespacing=1.15)
ax = fig.add_axes([0.235, 0.27, 0.205, 0.53])
lab = ["Official split\n(duplicates kept)", "Leak-controlled\ngrouped CV", "External web data\n(overlap-screened)",
       f"New sessions +\n~{round(az_n, -1):.0f} labelled images"]
dis = {"ConvNeXt-T": 0.591, "Swin-T": 0.579, "DINOv2": 0.707, "Ensemble": 0.621}
y = list(range(len(lab)))[::-1]
ax.barh(y[:2], [0.976, 0.900], color=[G, C1], height=0.62)
ax.barh([y[2]], [max(dis.values())], color=C4, height=0.62, alpha=0.35)
ax.scatter(list(dis.values()), [y[2]] * 4, color=C4, zorder=3, s=7)
ax.barh([y[3]], [az_m], color=C3, height=0.62)
ax.errorbar([az_m], [y[3]], xerr=[[az_m - az_lo], [az_hi - az_m]], fmt="none", ecolor="black", capsize=2.5, lw=0.9)
for yi, v, t in zip(y, [0.976, 0.900, 0.707, az_hi], ["97.6%", "90.0%", "58–71%", f"{100 * az_m:.0f}%"]):
    ax.text(v + (0.022 if t == "58–71%" else 0.012), yi, t, va="center", fontsize=10, fontweight="bold")
ax.set_yticks(y, lab, fontsize=9)
ax.set_xlim(0.4, 1.0)
ax.set_xticks([0.4, 0.6, 0.8, 1.0])
ax.set_xlabel("Classification accuracy", fontsize=9, labelpad=2)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.012, 0.025, f"Whisker: range over {R.tekrar.nunique()} session-grouped splits", fontsize=9, color="#444444")


def kutu(x, y0, w, h, baslik, metin, ec, tl=1, ls=1.2):
    fig.patches.append(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.006", transform=fig.transFigure, fc="white", ec=ec, lw=1.0))
    fig.text(x + 0.010, y0 + h - 0.035, baslik, fontsize=10, fontweight="bold", color=ec, va="top", linespacing=1.1)
    fig.text(x + 0.010, y0 + h - 0.055 - 0.07 * tl, metin, fontsize=9, va="top", linespacing=ls)


# Orta: bulgular
kutu(0.535, 0.535, 0.2, 0.43, "Leakage", "31.5% of files\nare copies\n42.7% of test\nimages duplicated\nin training", C4)
kutu(0.535, 0.035, 0.2, 0.43, "Label ambiguity", "3 blinded PV\nexperts\nFleiss κ = 0.12\n84% of errors:\nhealthy ↔ soiled", C2)
# Sağ: işletme-bakım çıkarımları
kutu(0.758, 0.035, 0.232, 0.93, "Implications\nfor O&M",
     "• Use RGB classifiers\n  to triage, not to\n  diagnose\n• Evaluate with scene-\n  or site-grouped\n  splits\n• Audit labels on a\n  random sample\n"
     "• Screen external test\n  data for overlap\n• Re-certify error\n  bounds on target\n  data", C3, tl=2, ls=1.13)
for ext in ("png", "tif", "pdf"):
    fig.savefig(f"makale/graphical_abstract.{ext}", dpi=600, **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}))
print("ok", round(az_m, 4), round(az_lo, 4), round(az_hi, 4), round(az_n, 1))
