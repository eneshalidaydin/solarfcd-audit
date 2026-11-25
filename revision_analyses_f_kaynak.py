"""R2.3: training-free check of source cues with frozen ImageNet ConvNeXt-T features and 1-NN lookup (no model is
trained or fine-tuned). Inputs: SolarFCD evaluation images (read-only), deposit manifest, E3 OOF predictions."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageOps
from torchvision import transforms

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(16)
OUT = Path(__file__).parent
ROOT = Path("<SOLARFCD_DIR>")
M = pd.read_csv("<DEPOSIT_DIR>/solarfcd_manifest.csv")
M = M[M.in_evaluation_set].reset_index(drop=True)
print(M.groupby(["source_family", "label"]).size().to_string())
f = OUT / "sfcd_feat_imagenet.npy"
if f.exists():
    Z = np.load(f)
else:
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    m = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=0).eval()
    Z = []
    with torch.no_grad():
        for i in range(0, len(M), 32):
            x = torch.stack([tf(ImageOps.exif_transpose(Image.open(ROOT / p)).convert("RGB").resize((256, 256)))
                             for p in M.rel_path[i:i + 32]])
            Z.append(torch.nn.functional.normalize(m(x), dim=1))
    Z = torch.cat(Z).numpy()
    np.save(f, Z)
fold = M.fold_scene_grouped.to_numpy()
fam = M.source_family.to_numpy()
lab = M.label.to_numpy()


def nn_cv(mask, target):
    """1-NN with scene-grouped folds: each image looks up its nearest neighbour among masked images in other folds."""
    idx = np.flatnonzero(mask)
    ok = []
    for i in idx:
        cand = idx[fold[idx] != fold[i]]
        j = cand[np.argmax(Z[cand] @ Z[i])]
        ok.append(target[j] == target[i])
    ok = np.array(ok)
    t = target[idx]
    bal = np.mean([ok[t == c].mean() for c in np.unique(t)])
    return ok.mean(), bal, len(idx)


src = np.where(np.isin(fam, ["Imgclean", "Imgdirty"]), "K", "PD")
for c in ["Healthy", "Surface_Obstruction"]:
    a, b, n = nn_cv(lab == c, src)
    print(f"Within {c}: 1-NN recovers the source collection (K vs PD) for {a:.3f} of {n} images (balanced {b:.3f})")
hs = np.isin(lab, ["Healthy", "Surface_Obstruction"])
for c in ["Healthy", "Surface_Obstruction"]:
    a, b, n = nn_cv(lab == c, fam)
    print(f"Within {c}: 1-NN recovers the source family for {a:.3f} (balanced {b:.3f}) over {np.unique(fam[lab == c]).tolist()}")
# H vs SO: within-source (grouped 1-NN) and cross-source (K -> PD and PD -> K)
for s in ["K", "PD"]:
    a, b, n = nn_cv(hs & (src == s), lab)
    print(f"H vs SO within {s} (grouped 1-NN): acc {a:.3f}, balanced {b:.3f}, n {n}")
for s, t in [("K", "PD"), ("PD", "K")]:
    tr, te = np.flatnonzero(hs & (src == s)), np.flatnonzero(hs & (src == t))
    j = tr[(Z[te] @ Z[tr].T).argmax(1)]
    ok = lab[j] == lab[te]
    bal = np.mean([ok[lab[te] == c].mean() for c in ["Healthy", "Surface_Obstruction"]])
    print(f"H vs SO, lookup in {s} only, tested on {t}: acc {ok.mean():.3f}, balanced {bal:.3f}, "
          f"recall H {ok[lab[te] == 'Healthy'].mean():.3f}, SO {ok[lab[te] == 'Surface_Obstruction'].mean():.3f}, n {len(te)}")
# E3 accuracy by source for H and SO (pooled OOF), for reference
D = pd.read_csv("<DEPOSIT_DIR>/oof_predictions.csv")
D = D[(D.protocol == "P3_scene_grouped") & D.model.isin(["convnext_tiny", "swin_tiny_patch4_window7", "vit_base_patch14_dinov2"])]
print(D.groupby(["model", "img_size"]).size().to_string())
