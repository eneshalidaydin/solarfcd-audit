"""Adım 34 - Grafik özet (Elsevier: en az 531 × 1328 piksel, y × g; TIFF/PDF). Yalnız makaledeki sayılar; üretken YZ görseli yok.
Çıktı: makale/graphical_abstract.{png,tif,pdf}"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
C1, C2, C3, C4, G = "#3b6ea5", "#d9822b", "#2e8b57", "#c0392b", "#9e9e9e"
fig = plt.figure(figsize=(13.28, 5.31), dpi=300)

# Sol: doğruluk merdiveni
ax = fig.add_axes([0.165, 0.18, 0.39, 0.66])
lab = ["Official split\n(duplicates kept)", "Leak-controlled\ngrouped CV", "New site\n(independent data)", "New site\n+ ~110 labelled images"]
val = [0.976, 0.900, 0.707, 0.918]
col = [G, C1, C4, C3]
y = list(range(len(lab)))[::-1]
ax.barh(y, val, color=col, height=0.62)
ax.plot([0.579, 0.707], [y[2], y[2]], color="white", lw=0)
ax.errorbar([0.643], [y[2]], xerr=[[0.064], [0.064]], fmt="none", ecolor="black", capsize=4, lw=1.2)
for yi, v, t in zip(y, val, ["97.6%", "90.0%", "58–71%", "91.8%"]):
    ax.text(v + 0.012, yi, t, va="center", fontsize=13, fontweight="bold")
ax.set_yticks(y, lab, fontsize=11)
ax.set_xlim(0.4, 1.08)
ax.set_xlabel("Classification accuracy", fontsize=11)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.set_title("What does \"97% accuracy\" mean for PV inspection?", loc="right", fontsize=13, fontweight="bold", pad=10)

# Orta: bulgular
def kutu(x, y0, w, h, baslik, metin, ec):
    fig.patches.append(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.008", transform=fig.transFigure, fc="white", ec=ec, lw=1.6))
    fig.text(x + 0.012, y0 + h - 0.05, baslik, fontsize=11.5, fontweight="bold", color=ec, va="top")
    fig.text(x + 0.012, y0 + h - 0.13, metin, fontsize=10.5, va="top", linespacing=1.35)

kutu(0.605, 0.56, 0.17, 0.34, "Leakage", "31.5% of files are copies\n42.7% of test images\nduplicated in training", C4)
kutu(0.605, 0.12, 0.17, 0.36, "Label ambiguity", "3 blinded PV experts\nFleiss κ = 0.12\n84% of errors:\nhealthy ↔ soiled", C2)

# Sağ: risk denetimli ön eleme
ax2 = fig.add_axes([0.81, 0.14, 0.16, 0.70])
ax2.bar([0], [0.906], color=C3, width=0.6)
ax2.bar([0], [0.094], bottom=[0.906], color="#e0e0e0", width=0.6)
ax2.text(0, 0.45, "91%\nautomatic\n97% accurate\n(certified\n≤5% error)", ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
ax2.text(0, 0.953, "9% to inspector", ha="center", va="center", fontsize=9.5)
ax2.set_xlim(-0.5, 0.5); ax2.set_ylim(0, 1); ax2.axis("off")
ax2.set_title("Risk-controlled triage\n(after label audit)", fontsize=11.5, fontweight="bold", color=C3)

fig.text(0.165, 0.025, "SolarFCD benchmark audit · 2,095 de-duplicated RGB images · scene-grouped CV · overlap-screened external test",
         fontsize=9.5, color="#555555")
for ext in ("png", "tif", "pdf"):
    fig.savefig(f"makale/graphical_abstract.{ext}", dpi=300, **({"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}))
print("ok")
