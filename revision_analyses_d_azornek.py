"""R2.4: reconstruct the few-shot split of 28_az_ornek_uyarlama.py (same grouping, seeds and sampling) on the CPU and
compute (a) file-name family baselines and (b) frozen-feature 1-NN baselines. No model is trained. Read-only inputs."""
import re
import sys
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageOps
from scipy.sparse.csgraph import connected_components
from torchvision import transforms

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(16)
K = "C:/solarfcd/pilot/"
OUT = Path(__file__).parent
SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
T = pd.read_csv(K + "dis_dogrulama/roboflow/ortusme.csv", encoding="utf-8-sig")
T = T[~T.ortusen & T.label.isin(SIN)].reset_index(drop=True)
y = T.label.map(SIN.index).to_numpy()
ac = lambda p: ImageOps.exif_transpose(Image.open(p)).convert("RGB")
kok = T.path.map(lambda p: re.split(r"[._]rf[._]|_jpg\.rf", Path(p).name)[0])
cache = OUT / "rf_phash.npy"
if cache.exists():
    H = np.load(cache)
else:
    H = np.array([imagehash.phash(ac(p)).hash.flatten() for p in T.path])
    np.save(cache, H)
A = (H[:, None, :] != H[None, :, :]).sum(2) <= 6
A |= (kok.to_numpy()[:, None] == kok.to_numpy()[None, :])
ng, grp = connected_components(A, directed=False)
rng = np.random.default_rng(20261122)
gp = rng.permutation(ng)
test_g, havuz_g = gp[: ng // 2], gp[ng // 2:]
te = np.isin(grp, test_g)
ti = np.where(te)[0]
print("images", len(T), "groups", ng, "test", len(ti))


def fam(s):
    m = re.match(r"^IMG_(\d{8})_", s)
    if m:
        return "IMG_" + m.group(1)
    if re.match(r"^IMG_\d+$", s):
        return "IMG_generic"
    if re.match(r"^\d+_F_", s):
        return "stock_F"
    m = re.match(r"^([A-Za-z][A-Za-z_\-]*?)[_\-]?\d", s)
    if m:
        return m.group(1).lower()
    m = re.match(r"^([A-Za-z][A-Za-z_\-]*)", s)
    return m.group(1).lower() if m else "numeric"


F = kok.map(fam)
T["family"] = F
tab = pd.crosstab(F, T.label)
maj = tab.idxmax(1)
print("families", len(tab), "| in-sample family-majority accuracy", round((T.family.map(maj) == T.label).mean(), 4),
      "| majority-class baseline", round(T.label.value_counts().max() / len(T), 4))
big = tab.assign(n=tab.sum(1)).sort_values("n", ascending=False).head(12)
print(big.to_string())
tab.assign(n=tab.sum(1)).sort_values("n", ascending=False).to_csv(OUT / "rf_families.csv")

# frozen features (no training): ImageNet ConvNeXt-T and the SolarFCD fold-0 ConvNeXt-T used for initialisation
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def feats(init):
    f = OUT / f"rf_feat_{init}.npy"
    if f.exists():
        return np.load(f)
    m = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=(init == "imagenet"), num_classes=4)
    if init == "solarfcd":
        m.load_state_dict(torch.load(K + "runs_asama2/grouped0_convnext_tiny_224_s42_kat_v1_asama2_none/best.pt",
                                     map_location="cpu"))
    m.eval()
    Z = []
    with torch.no_grad():
        for i in range(0, len(T), 32):
            x = torch.stack([tf(ac(p).resize((256, 256))) for p in T.path[i:i + 32]])
            Z.append(torch.nn.functional.normalize(m.forward_head(m.forward_features(x), pre_logits=True), dim=1))
    Z = torch.cat(Z).numpy()
    np.save(f, Z)
    return Z


Z = {b: feats(b) for b in ("imagenet", "solarfcd")}
ref = pd.read_csv(K + "asama2_sonuclar/az_ornek.csv")
R = []
for oran in (0.05, 0.10, 0.20):
    for tek in range(3):
        r2 = np.random.default_rng(100 * tek + 7)
        hg = r2.permutation(havuz_g)
        sec, n = [], 0
        for g_ in hg:
            if n >= oran * len(T):
                break
            sec.append(g_); n += (grp == g_).sum()
        tr = np.where(np.isin(grp, sec))[0]
        n_ref = ref[(ref.oran == oran) & (ref.tekrar == tek)].n_egitim.iloc[0]
        fm = pd.Series(y[tr]).groupby(T.family.to_numpy()[tr]).agg(lambda v: np.bincount(v, minlength=4).argmax())
        glob = np.bincount(y[tr], minlength=4).argmax()
        pf = np.array([fm.get(f_, glob) for f_ in T.family.to_numpy()[ti]])
        seen = np.isin(T.family.to_numpy()[ti], fm.index)
        r = dict(oran=oran, tekrar=tek, n_train=len(tr), n_train_28=int(n_ref),
                 test_family_seen=seen.mean(), family_majority_acc=(pf == y[ti]).mean(),
                 family_majority_acc_seen=(pf[seen] == y[ti][seen]).mean())
        for b in Z:
            s = Z[b][ti] @ Z[b][tr].T
            nn = s.argmax(1)
            r[f"1nn_{b}"] = (y[tr][nn] == y[ti]).mean()
            r[f"nn_cos_median_{b}"] = np.median(s.max(1))
        R.append(r)
R = pd.DataFrame(R)
R.to_csv(OUT / "d_azornek.csv", index=False)
print(R.round(3).to_string())
fs = ref[ref.oran > 0].groupby(["oran", "baslangic"]).acc.mean().unstack()
print(R.groupby("oran")[["n_train", "test_family_seen", "family_majority_acc", "1nn_imagenet", "1nn_solarfcd"]].mean().round(3)
      .join(fs.round(3)).to_string())
