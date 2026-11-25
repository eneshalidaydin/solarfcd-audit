"""İç değerlendirme R1 kontrolü - F: RF-Fault kutu birlesimleri x eslenen etiket (tam tablo, ek tablo icin); ozet kelime ve
one cikan karakter sayilari. Yalniz okur."""
import re
from pathlib import Path

import pandas as pd

H = Path(__file__).parent
DEP = Path(r"<DEPOSIT_DIR>")
AD = {"0": "Bird Drop", "1": "Defective", "2": "Dust", "3": "Dusty", "4": "Non Defective", "5": "Physical Damage", "6": "Snow"}
O = pd.read_csv(DEP / "external_overlap_rf_fault.csv", dtype={"source_label_or_box_classes": str})
O["removed"] = O.removed_as_overlapping.astype(str) == "True"
O["combo"] = O.source_label_or_box_classes.fillna("").map(lambda s: " + ".join(AD[x] for x in s.split(",") if x))
out = []
for nm, d in [("all 2,763", O), ("retained 1,478", O[~O.removed])]:
    t = d.groupby(["mapped_label", "combo"]).size().rename("n").reset_index().sort_values(["mapped_label", "n"], ascending=[True, False])
    out.append(f"== {nm}\n" + t.to_string(index=False))
    out.append(d.mapped_label.value_counts().to_string())
txt = "\n".join(out)
print(txt)

ABS_OLD = """Deep-learning models for the visual inspection of photovoltaic (PV) modules are often reported to exceed 95% accuracy on public image datasets, yet these figures are rarely checked for data leakage, label reliability or transfer to new data. This study examines all three issues using the red–green–blue (RGB) images of SolarFCD, a recent four-class benchmark, in order to assess how reliable image-based fault detection is for operation and maintenance (O&M).
A combined duplicate and scene-level audit revealed substantial overlap between the official training and test splits, and under leak-controlled, scene-grouped evaluation the accuracy of the best model fell from 97.6% to 90.0%. Most remaining errors lay at the boundary between healthy and soiled modules, on which three blinded PV experts also agreed poorly in an independent relabelling. Building on these findings, a selective classifier with a certified 5% error bound classified 82–91% of the images automatically at about 97% accuracy once ambiguous labels were removed or audited. On an independent dataset with differently defined labels, the accuracy was lower, but fine-tuning with about 110 labelled images from that dataset raised it to 91.8%.
Taken together, the results show that headline accuracies on SolarFCD, and probably on other web-sourced PV image datasets, reflect evaluation design and label quality as much as model capability. Image-based classification is therefore best used for fault triage with a predictable inspection workload, and the study proposes a reporting checklist that makes these effects visible in future PV image benchmarks."""
ABS_NEW = """Deep-learning models for the visual inspection of photovoltaic (PV) modules are often reported to exceed 95% accuracy on public image datasets, yet these figures are rarely checked for data leakage, label reliability or transfer to new data. This study examines all three issues using the web-sourced red–green–blue (RGB) photographs of SolarFCD, a recent four-class benchmark, in order to assess what such benchmarks imply for image-based triage in operation and maintenance (O&M).
A combined duplicate and scene-level audit revealed substantial overlap between the official training and test splits, and under leak-controlled, scene-grouped evaluation on a harder, multi-source test population the accuracy of the best model fell from 97.6% to 90.0%. Most remaining errors lay at the boundary between healthy and soiled modules, on which three blinded PV experts also agreed poorly in an independent relabelling. A selective classifier with a certified 5% error bound classified on average 57% of the images automatically with the released labels and 82–91% once ambiguous labels were removed or audited. On an overlap-screened external dataset with differently defined labels, the accuracy fell to 58–71% and the certified error bound did not transfer, whereas fine-tuning with about 110 labelled images from that dataset raised the accuracy to about 92%.
Taken together, the results show that headline accuracies on SolarFCD, and probably on other web-sourced PV image datasets, reflect evaluation design and label quality as much as model capability. RGB image classification is therefore best used to triage images ahead of on-site inspection and electrical verification, and the study proposes a reporting checklist that makes these effects visible in future PV image benchmarks."""
HL = ["Leak-controlled, multi-source evaluation cut accuracy from 97.6% to 90.0%",
      "Without a severity scale, three blinded PV experts agreed poorly (κ = 0.12)",
      "A certified 5% error bound automated 57% of images, or 82–91% after label audit",
      "Nearly all images (98.5%) of a public Kaggle PV dataset overlapped with SolarFCD",
      "On overlap-screened external web data, accuracy fell to 58–71%",
      "About 110 labels from an external web dataset raised accuracy to about 92%"]
s = [f"abstract words (whitespace split): old {len(ABS_OLD.split())}, new {len(ABS_NEW.split())}; "
     f"counting en-dash compounds as separate words: old {len(re.split(r'[\s–]+', ABS_OLD))}, new {len(re.split(r'[\s–]+', ABS_NEW))}; "
     f"paragraphs new {ABS_NEW.count(chr(10)) + 1}"]
s += [f"highlight {len(h):3d} chars: {h}" for h in HL]
print("\n".join(s))
(H / "r1_f_sonuc.txt").write_text(txt + "\n\n" + "\n".join(s), encoding="utf-8")
