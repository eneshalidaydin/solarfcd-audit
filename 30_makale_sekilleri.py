"""Adım 30 - Makale şekilleri (yalnız mevcut çıktılardan; yeni eğitim yok). Çıktı: makale/fig*.png"""
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
PC = [f"p_{c}" for c in SIN]
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
# Fig 1: değerlendirme ayarına göre doğruluk (aynı aile modeller)
lab = ["Official split\n(duplicates kept)", "Random 5-fold\n(scene leakage)", "Scene-grouped\n5-fold", "RF-Fault\n(external)", "RF-Soiling\n(external, SO only)"]
dino = [0.9760, 0.9103, 0.8974, 0.7067, 0.2847]
fig, ax = plt.subplots(figsize=(7.2, 3.4))
b = ax.bar(lab, dino, color=["#9e9e9e", "#bdbdbd", "#3b6ea5", "#d9822b", "#c0392b"])
for r, v in zip(b, dino):
    ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center")
ax.set_ylim(0, 1.08); ax.set_ylabel("Accuracy (DINOv2 ViT-B/14, 224 px)")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("makale/fig1_evaluation_settings.png", dpi=300); plt.close()
# Fig 2: doğruluk-kapsam eğrisi (E3; orijinal ve temiz)
def oof(kok, son):
    D = [pd.concat([pd.read_csv(f"{kok}/grouped{k}_{u}_s{s}_{son}/test_predictions.csv") for k in range(5)]).set_index("path").sort_index()
         for u in ["convnext_tiny_224", "swin_tiny_patch4_window7_224_224", "vit_base_patch14_dinov2_392"] for s in (42, 43, 44)]
    return np.mean([d[PC].to_numpy() for d in D], 0), D[0].label.map(SIN.index).to_numpy()
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for (kok, son, ad, c) in [("runs_asama2", "kat_v1_asama2_none", "Original labels (n = 2,095)", "#3b6ea5"),
                          ("runs_temiz", "kat_v1_asama2_temiz_none", "Expert-audited labels (n = 2,049)", "#2e8b57")]:
    P, y = oof(kok, son)
    o = np.argsort(-P.max(1)); d = (P.argmax(1) == y)[o]
    cov = np.arange(1, len(y) + 1) / len(y); acc = np.cumsum(d) / np.arange(1, len(y) + 1)
    ax.plot(cov[20:], acc[20:], label=ad, color=c)
ax.axhline(0.95, ls="--", color="grey", lw=0.8); ax.text(0.02, 0.952, "95%", color="grey")
ax.set_xlabel("Coverage (fraction of images auto-classified)"); ax.set_ylabel("Accuracy on accepted images")
ax.set_ylim(0.88, 1.001); ax.legend(frameon=False, loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("makale/fig2_accuracy_coverage.png", dpi=300); plt.close()
# Fig 3: az örnekle uyarlama
R = pd.read_csv("asama2_sonuclar/az_ornek.csv")
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for bas, ad, c in [("solarfcd", "Initialised from SolarFCD model", "#3b6ea5"), ("imagenet", "Initialised from ImageNet only", "#d9822b")]:
    g = R[R.baslangic == bas].groupby("oran").agg(n=("n_egitim", "mean"), m=("acc", "mean"), s=("acc", "std")).fillna(0)
    ax.errorbar(g.n, g.m, yerr=g.s, marker="o", capsize=3, label=ad, color=c)
ax.set_xlabel("Labelled target-domain training images"); ax.set_ylabel("Accuracy on fixed RF-Fault test set (n = 552)")
ax.set_ylim(0.55, 1.0); ax.legend(frameon=False, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("makale/fig3_few_shot.png", dpi=300); plt.close()
shutil.copy("asama2_sonuclar/gradcam.png", "makale/fig4_gradcam.png")
print("ok")
