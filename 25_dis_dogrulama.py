"""Adım 25 - Dış doğrulama (kullanıcı kararı 21 Kasım): Kaggle "PV Panel Defect Dataset" (alicjalena, 2025; 1.574 RGB;
<EXTERNAL_DATA_DIR>/pv_panel_defect, salt okunur). Sınıf eşlemesi: Clean->Healthy; Dusty, Bird-drop, Snow-Covered->
Surface_Obstruction; Electrical-damage->Electrical_Fault; Physical-Damage->Structural_Fault.
1) Örtüşme (ihtiyatlı): SolarFCD'nin tüm görüntülerine karşı MD5, pHash (Hamming ≤ 12) ve DINOv2 (ön eğitimli, CLS) kosinüs
   ≥ 0,85. Herhangi birine takılan dış görüntü "örtüşen" sayılır; dış test = örtüşmeyenler.
2) Çıkarım: SolarFCD'de eğitilmiş kat modelleri (5 kat × 3 tohum, aynı mimari olasılık ortalaması; TTA = özdeş + yatay
   + dikey çevirme, 02b ile aynı). Ayrıca tohum 42 rastgele (sızıntılı) ve gruplu (sızıntısız) kat modelleri karşılaştırılır:
   iç CV tahmini ile dış doğruluk arasındaki fark (iyimserlik).
Çıktı: dis_dogrulama/ (ortusme.csv, tahminler_*.csv, ozet.txt)."""
import hashlib
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageOps
from torchvision import transforms

SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
ESL = {"Clean": "Healthy", "Dusty": "Surface_Obstruction", "Bird-drop": "Surface_Obstruction",
       "Snow-Covered": "Surface_Obstruction", "Electrical-damage": "Electrical_Fault", "Physical-Damage": "Structural_Fault"}
import sys
VERI = sys.argv[1] if len(sys.argv) > 1 else "kaggle"
D0 = Path("<EXTERNAL_DATA_DIR>")
OUT = Path("dis_dogrulama") / ("" if VERI == "kaggle" else VERI)
OUT.mkdir(parents=True, exist_ok=True)
# Tespit setleri (YOLO) görüntü düzeyine indirgenir. Roboflow sınıfları: 0 Bird Drop, 1 Defective, 2 Dust, 3 Dusty,
# 4 Non Defective, 5 Physical Damage, 6 Snow. Öncelik: Physical Damage -> SF; Defective -> "Ariza" (EF/SF ayrılamaz,
# yalnız arıza var/yok analizinde); Bird/Dust/Dusty/Snow -> SO; yalnız Non Defective -> Healthy.
# Melbourne (Soiling_DS_v2_14; 0 Dust, 1 bird-droppings): kutu varsa SO, boş etiket -> Healthy.


def yolo(kok, kural):
    r = []
    for p in sorted(kok.rglob("*.jpg")):
        lb = p.parent.parent / "labels" / (p.stem + ".txt")
        c = {int(s.split()[0]) for s in lb.read_text().splitlines() if s.strip()} if lb.exists() else set()
        r.append(dict(path=str(p), ust=",".join(map(str, sorted(c))), label=kural(c)))
    return pd.DataFrame(r)


def rob(c):
    if 5 in c: return "Structural_Fault"
    if 1 in c: return "Ariza"
    if c & {0, 2, 3, 6}: return "Surface_Obstruction"
    return "Healthy" if c == {4} else "yok"


KUR = {"roboflow": lambda: yolo(D0 / "roboflow", rob),
       "melbourne": lambda: yolo(D0 / "melbourne", lambda c: "Surface_Obstruction" if c else "Healthy")}
cihaz = "cuda" if torch.cuda.is_available() else "cpu"


def ac(p):
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")


if VERI == "kaggle":
    X = pd.DataFrame([dict(path=str(p), ust=p.parent.name, label=ESL[p.parent.name]) for p in
                      sorted((D0 / "pv_panel_defect").rglob("*")) if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
else:
    X = KUR[VERI]()
    X = X[X.label != "yok"].reset_index(drop=True)
S = pd.read_csv("asama2/manifest_asama2.csv")
print("dış", len(X), "| SolarFCD", len(S), flush=True)


@torch.no_grad()
def ozellik(yollar):
    m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224).to(cihaz).eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    F = []
    for i in range(0, len(yollar), 64):
        F.append(torch.nn.functional.normalize(m(torch.stack([tf(ac(p)) for p in yollar[i:i + 64]]).to(cihaz)), dim=1).cpu())
    return torch.cat(F)


if not (OUT / "ortusme.csv").exists():
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    X["md5"] = X.path.map(md5)
    X["ph"] = [imagehash.phash(ac(p)) for p in X.path]
    sph = [imagehash.phash(ac(p)) for p in S.path]
    sh = np.array([h.hash.flatten() for h in sph])
    X["ph_min"] = [int((sh != h.hash.flatten()).sum(1).min()) for h in X.ph]
    fx, fs = ozellik(list(X.path)), ozellik(list(S.path))
    sim = fx @ fs.T
    X["cos_max"] = sim.max(1).values.numpy()
    X["en_yakin"] = S.rel_path.to_numpy()[sim.argmax(1).numpy()]
    X["md5_eslesme"] = X.md5.isin(set(S.md5))
    X["ortusen"] = X.md5_eslesme | (X.ph_min <= 12) | (X.cos_max >= 0.85)
    X.drop(columns=["ph"]).to_csv(OUT / "ortusme.csv", index=False, encoding="utf-8-sig")
X = pd.read_csv(OUT / "ortusme.csv", encoding="utf-8-sig")
T = X[~X.ortusen].reset_index(drop=True)
y = T.label.map(lambda v: SIN.index(v) if v in SIN else -1).to_numpy()
bil = y >= 0
ariza_y = np.isin(y, [0, 2]) | (y == -1)


@torch.no_grad()
def cikar(kosu, model_adi, S_):
    m = timm.create_model(model_adi, pretrained=False, num_classes=4,
                          **({"img_size": S_} if any(k in model_adi for k in ("vit", "dinov2")) else {}))
    m.load_state_dict(torch.load(kosu / "best.pt", map_location="cpu"))
    m = m.to(cihaz).eval()
    cfg = timm.data.resolve_data_config({}, model=m)
    tf = transforms.Compose([transforms.Resize((S_, S_)), transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])])
    P = []
    for i in range(0, len(T), 32):
        x = torch.stack([tf(ac(p)) for p in T.path[i:i + 32]]).to(cihaz)
        P.append(torch.stack([m(x).softmax(1), m(x.flip(3)).softmax(1), m(x.flip(2)).softmax(1)]).mean(0).cpu())
    return torch.cat(P).numpy()


def f1(y, p):
    c = np.zeros((4, 4))
    np.add.at(c, (y, p), 1)
    den = c.sum(0) + c.sum(1)
    return np.mean(np.where(den > 0, 2 * np.diag(c) / np.maximum(den, 1e-12), 0))


MOD = {"ConvNeXt": ("convnext_tiny.fb_in22k_ft_in1k", 224, "convnext_tiny_224"),
       "Swin": ("swin_tiny_patch4_window7_224.ms_in22k_ft_in1k", 224, "swin_tiny_patch4_window7_224_224"),
       "DINOv2-392": ("vit_base_patch14_dinov2.lvd142m", 392, "vit_base_patch14_dinov2_392"),
       "DINOv2-224": ("vit_base_patch14_dinov2.lvd142m", 224, "vit_base_patch14_dinov2_224")}
s = [f"Dış set {len(X)}; örtüşen {int(X.ortusen.sum())} (MD5 {int(X.md5_eslesme.sum())}, pHash≤12 {int((X.ph_min <= 12).sum())}, "
     f"kosinüs≥0,85 {int((X.cos_max >= 0.85).sum())}); dış test {len(T)}: {T.label.value_counts().to_dict()}", ""]
olas = {}
for ad, (mdl, S_, kisa) in MOD.items():
    f = OUT / f"tahminler_{ad}.npy"
    if not f.exists():
        np.save(f, np.mean([cikar(Path(f"runs_asama2/grouped{k}_{kisa}_s{t}_kat_v1_asama2_none"), mdl, S_)
                            for t in (42, 43, 44) for k in range(5)], 0))
    olas[ad] = np.load(f)
olas["Topluluk-3 yüksek"] = np.mean([olas["ConvNeXt"], olas["Swin"], olas["DINOv2-392"]], 0)
for ad, P in olas.items():
    p = P.argmax(1)
    s.append(f"{ad:18s} arıza var/yok acc {(np.isin(p, [0, 2]) == ariza_y).mean():.4f} (n={len(y)})")
    p, yy = p[bil], y[bil]
    s.append(f"{ad:18s} dış acc {(p == yy).mean():.4f}  F1 {f1(yy, p):.4f}  " + " ".join(
        f"{c[:2]} {(p[yy == i] == i).mean():.3f}" for i, c in enumerate(SIN)) +
        f"  | H+SO birleşik acc {(np.where(np.isin(p, [1, 3]), 9, p) == np.where(np.isin(yy, [1, 3]), 9, yy)).mean():.4f}")
s += ["", "İyimserlik: iç CV (tohum 42, 2.095) vs dış acc/F1, aynı mimari; sızıntılı (rastgele) vs sızıntısız (gruplu) kat modelleri"]
for ad, (mdl, S_, kisa) in [("ConvNeXt", MOD["ConvNeXt"]), ("Swin", MOD["Swin"]), ("DINOv2-224", MOD["DINOv2-224"])]:
    for tur, kok, son in [("sızıntılı", "runs_rastgele", "kat_rastgele_asama2_rastgele_none"),
                          ("sızıntısız", "runs_asama2", "kat_v1_asama2_none")]:
        f = OUT / f"tahminler_{ad}_{tur}_s42.npy"
        if not f.exists():
            np.save(f, np.mean([cikar(Path(f"{kok}/grouped{k}_{kisa}_s42_{son}"), mdl, S_) for k in range(5)], 0))
        p = np.load(f).argmax(1)[bil]
        ic = pd.concat([pd.read_csv(f"{kok}/grouped{k}_{kisa}_s42_{son}/test_predictions.csv") for k in range(5)])
        yi, pi = ic.label.map(SIN.index).to_numpy(), ic.tahmin.map(SIN.index).to_numpy()
        s.append(f"{ad:10s} {tur:10s} iç acc {(pi == yi).mean():.4f} F1 {f1(yi, pi):.4f} -> dış acc {(p == y[bil]).mean():.4f} "
                 f"F1 {f1(y[bil], p):.4f}  (iyimserlik acc {(pi == yi).mean() - (p == y[bil]).mean():+.4f})")
(OUT / "ozet.txt").write_text("\n".join(s), encoding="utf-8")
print("\n".join(s))
