"""İç değerlendirme R1 kontrolü - C: RF-Fault Imgdirty adlari (tum satirlar) ve IMG_#### oturumlari. Yalniz okur."""
import re
from pathlib import Path
import pandas as pd
DEP = Path(r"<DEPOSIT_DIR>")
O = pd.read_csv(DEP / "external_overlap_rf_fault.csv", dtype={"source_label_or_box_classes": str})
O["name"] = O.rel_path.map(lambda p: Path(p).name)
O["stem"] = O.name.map(lambda n: re.split(r"[._]rf[._]|_jpg\.rf", n)[0])
O["removed"] = O.removed_as_overlapping.astype(str) == "True"
im = O[O.stem.str.startswith("Imgdirty_")]
print("Imgdirty_* in all RF-Fault:", len(im), "removed", int(im.removed.sum()), "retained", int((~im.removed).sum()))
print(" with Defective box:", int(im.source_label_or_box_classes.fillna("").str.split(",").map(lambda c: "1" in c).sum()))
print(" box combos:", im.source_label_or_box_classes.value_counts().head(6).to_dict())
print(" mapped:", im.mapped_label.value_counts().to_dict())
R = O[~O.removed]
K4 = R[R.mapped_label.isin(["Healthy", "Structural_Fault", "Surface_Obstruction"])]
cam = K4[K4.stem.str.match(r"^IMG_\d{4}(?!\d)")]
cn = cam.stem.str.extract(r"^IMG_(\d{4})")[0]
print("IMG_#### (4 digits, not dates) in four-class retained:", len(cam), "distinct numbers:", cn.nunique(),
      "labels:", cam.mapped_label.value_counts().to_dict())
print(" range of numbers:", cn.astype(int).min(), cn.astype(int).max())
