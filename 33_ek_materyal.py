"""Adım 33 - Makale ek materyali (Tablo S1-S6) mevcut sonuç dosyalarından üretilir -> makale/supplementary.md"""
import re
import pandas as pd

def md(df):
    cols = list(df.columns)
    return "\n".join(["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)])

s = ["---", 'title: "Supplementary material"', "---", "",
     "**How reliable is image-based photovoltaic fault detection for operation and maintenance? Leakage, label ambiguity and domain shift in a public benchmark**", "",
     "Enes Halit Aydın", ""]
# S1
a = pd.read_csv("asama2_sonuclar/az_ornek.csv")
g = a.groupby(["oran", "baslangic"]).agg(n=("n_egitim", "mean"), acc=("acc", "mean"), acc_sd=("acc", "std"), f1=("f1", "mean"), f1_sd=("f1", "std")).reset_index()
g["Target fraction"] = (g.oran * 100).round().astype(int).astype(str) + "%"
g["Initialisation"] = g.baslangic.map({"solarfcd": "SolarFCD", "imagenet": "ImageNet"})
g["Training images"] = g.n.round().astype(int)
f = lambda m, sd: f"{m:.3f}" + ("" if pd.isna(sd) else f" ± {sd:.3f}")
g["Accuracy"] = [f(m, sd) for m, sd in zip(g.acc, g.acc_sd)]
g["Macro-F1 (H, SF, SO)"] = [f(m, sd) for m, sd in zip(g.f1, g.f1_sd)]
s += ["# Table S1. Few-shot adaptation on RF-Fault", "",
      "Fixed group-level test set of 552 images (613 groups in 1,108 images); mean ± SD over three samples per budget.", "",
      md(g[["Target fraction", "Initialisation", "Training images", "Accuracy", "Macro-F1 (H, SF, SO)"]]), ""]
# S2
rows = []
for et, ad in [("orij", "Original (2,095)"), ("orijbel", "Ambiguous removed (2,049)"), ("temiz", "Audited (2,049)")]:
    for r in ("0.02", "0.05", "0.10"):
        t = open(f"asama2_sonuclar/secici_ek/{et}_r{r}.txt", encoding="utf-8").read()
        m = re.search(r"kapsam ([\d.]+) \(±([\d.]+)\), seçilenlerde doğruluk ([\d.na]+).*?ihlali oranı ([\d.]+)", t)
        rows.append([ad, f"{float(r):.0%}", f"{float(m.group(1)):.3f} ± {float(m.group(2)):.3f}", m.group(3)[:6], m.group(4)])
s += ["# Table S2. Sensitivity of the LTT selective classifier to the target error α", "",
      "Ensemble E3; δ = 0.1; 1,000 scene-level 50/50 calibration/test splits. 'nan': no image accepted in any split.", "",
      md(pd.DataFrame(rows, columns=["Labels", "α", "Coverage", "Accuracy on accepted", "Violation rate"])), ""]
# S3
t = open("asama2_sonuclar/ek_analiz.txt", encoding="utf-8").read()
lr = re.findall(r"^(ConvNeXt|DINOv2|Swin)\s+F1 ([\d.]+) vs ([\d.]+)\s+fark ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]\s+acc ([\d.]+)\s+SF ([\d.]+)", t, re.M)
sel = {"ConvNeXt": "×0.5 (5 × 10⁻⁵)", "Swin": "×3 (1.5 × 10⁻⁴)", "DINOv2": "×0.5 (1 × 10⁻⁵)"}
s += ["# Table S3. Learning-rate tuning by inner grouped validation", "",
      "Candidate multipliers ×0.5, ×1, ×3 of the base learning rate, selected by mean last-epoch macro-F1 on an inner scene-grouped validation split (12.5%); then retrained without validation; seed 42, pooled OOF.", "",
      md(pd.DataFrame([[m, sel[m], b, a_, f"{d} [{lo}, {hi}]", acc, sf] for m, a_, b, d, lo, hi, acc, sf in lr],
                      columns=["Model", "Selected learning rate", "Macro-F1, base", "Macro-F1, tuned", "Difference [95% CI]", "Accuracy, tuned", "SF recall, tuned"])), ""]
# S4
c = open("asama2_sonuclar/cozunurluk.txt", encoding="utf-8").read()
rr = re.findall(r"^(ConvNeXt-224|ConvNeXt-384|DINOv2-224|DINOv2-392)\s+acc ([\d.]+)\s+F1 ([\d.]+)\s+EF ([\d.]+) H ([\d.]+) SF ([\d.]+) SO ([\d.]+)", c, re.M)
s += ["# Table S4. Effect of input resolution (P3, three seeds, pooled OOF)", "",
      md(pd.DataFrame(rr, columns=["Model", "Accuracy", "Macro-F1", "EF recall", "H recall", "SF recall", "SO recall"])), "",
      "Paired macro-F1 differences (cluster bootstrap): ConvNeXt-384 − ConvNeXt-224 = −0.013 [−0.043, +0.015]; DINOv2-392 − DINOv2-224 = +0.025 [+0.003, +0.049].", ""]
# S5
r5 = pd.read_csv("asama2_sonuclar/resnet_duyarlilik.csv")
r5["Variant"] = r5.varyant.map({"dA": "A: inner grouped validation, F1-based selection", "dB": "B: 60 epochs", "dC": "C: learning rate 3 × 10⁻⁴"})
for k in ["acc", "f1", "rec_El", "rec_He", "rec_St", "rec_Su"]:
    r5[k] = r5[k].map(lambda v: f"{v:.3f}")
s += ["# Table S5. ResNetV2-101 sensitivity (seed 42, P3, pooled OOF)", "",
      "Main protocol (three seeds): accuracy 0.810, macro-F1 0.435.", "",
      md(r5[["Variant", "acc", "f1", "rec_El", "rec_He", "rec_St", "rec_Su"]].rename(columns={"acc": "Accuracy", "f1": "Macro-F1", "rec_El": "EF recall", "rec_He": "H recall", "rec_St": "SF recall", "rec_Su": "SO recall"})), ""]
# S6
t6 = pd.read_csv("asama2_sonuclar/tohum_bazinda.csv")
ad = {"convnext_tiny": "ConvNeXt-T", "convnextv2_tiny": "ConvNeXt V2-T", "resnetv2_101": "ResNetV2-101", "swin_tiny_patch4_window7": "Swin-T",
      "tf_efficientnetv2_s": "EfficientNetV2-S", "vit_base_patch14_dinov2": "DINOv2 (224 px)"}
t6["Model"] = t6.model.map(ad).fillna(t6.model)
for k in ["acc", "f1", "f1_lo", "f1_hi", "rec_El", "rec_He", "rec_St", "rec_Su"]:
    t6[k] = t6[k].map(lambda v: f"{v:.3f}")
t6["Macro-F1 [95% CI]"] = t6.f1 + " [" + t6.f1_lo + ", " + t6.f1_hi + "]"
s += ["# Table S6. Seed-level results under P3 (pooled OOF, 2,095 images)", "",
      md(t6[["Model", "tohum", "acc", "Macro-F1 [95% CI]", "rec_El", "rec_He", "rec_St", "rec_Su"]].rename(columns={"tohum": "Seed", "acc": "Accuracy", "rec_El": "EF recall", "rec_He": "H recall", "rec_St": "SF recall", "rec_Su": "SO recall"})), ""]
open("makale/supplementary.md", "w", encoding="utf-8").write("\n".join(s))
print("ok", len(s))
