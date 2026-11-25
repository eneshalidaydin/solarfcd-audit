"""Adım 28b - Az örnekli uyarlamanın oturum/alt koleksiyon gruplu tekrarı (kullanıcı onayı 24 Kasım, iç değerlendirme R1.6/R2.4).
28'den farkı yalnızca bölme: Roboflow görüntüleri (örtüşmeyen 1.108, H/SF/SO) dosya adlarından kurulan oturum ve alt
koleksiyon gruplarına ayrılır. Bağlar: (i) aynı kaynak dosya kökü (Roboflow artırma kopyaları) ya da pHash Hamming ≤ 6,
(ii) aynı adlandırma ailesi (ör. Physical_Damge, roi, stok fotoğraf kimlikleri), (iii) tarih damgalı telefon fotoğrafları
için aynı gün, (iv) IMG_#### kamera dosyaları için ardışık numara dizileri (boşluk ≤ 10). Her tekrar (10 tekrar) grupları
sınıf dengeli olarak test ve uyarlama yarılarına böler (StratifiedGroupKFold, 2 kat). Uyarlama örneği, uyarlama
yarısından bütün kopya grupları çekilerek 1.108'in %5/10/20'sine ulaşılana dek toplanır. Eğitim, başlangıçlar ve TTA
28 ile aynı (ConvNeXt-T, 15 epok, AdamW 1e-4, bs 16). Eğitimsiz tabanlar: 1-NN ve doğrusal yoklama (dondurulmuş
ImageNet ve SolarFCD öznitelikleri). Çıktı: asama2_sonuclar/az_ornek_gruplu.csv, az_ornek_gruplu.txt"""
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from torchvision import transforms

sys.stdout.reconfigure(encoding="utf-8")
KURU = "--kuru" in sys.argv
SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
cih = "cuda"
T = pd.read_csv("dis_dogrulama/roboflow/ortusme.csv", encoding="utf-8-sig")
T = T[~T.ortusen & T.label.isin(SIN)].reset_index(drop=True)
y = T.label.map(SIN.index).to_numpy()
ac = lambda p: ImageOps.exif_transpose(Image.open(p)).convert("RGB")
kok = T.path.map(lambda p: re.split(r"[._]rf[._]|_jpg\.rf", Path(p).name)[0])
on = Path("hakem_revizyon/analiz/rf_phash.npy")
H = np.load(on) if on.exists() else np.array([imagehash.phash(ac(p)).hash.flatten() for p in T.path])
assert len(H) == len(T)


def aile(s):
    m = re.match(r"^IMG_(\d{8})_", s)
    if m:
        return "IMGgun_" + m.group(1)
    if re.match(r"^IMG_\d+$", s):
        return "IMGno"
    if re.match(r"^\d+_F_", s):
        return "stock_F"
    m = re.match(r"^([A-Za-z][A-Za-z_\-]*?)[_\-]?\d", s)
    if m:
        return m.group(1).lower()
    m = re.match(r"^([A-Za-z][A-Za-z_\-]*)", s)
    return m.group(1).lower() if m else "numeric"


F = kok.map(aile).to_numpy().astype(object)
no = sorted({int(k[4:]) for k in kok if re.match(r"^IMG_\d+$", k)})
oturum, s = {}, 0
for i, n in enumerate(no):
    if i and n - no[i - 1] > 10:
        s += 1
    oturum[n] = s
for i, k in enumerate(kok):
    if F[i] == "IMGno":
        F[i] = f"IMGno_{oturum[int(k[4:])]}"
kk = kok.to_numpy()
A = (H[:, None, :] != H[None, :, :]).sum(2) <= 6
A |= kk[:, None] == kk[None, :]
ince_n, ince = connected_components(A, directed=False)
A |= F[:, None] == F[None, :]
ng, grp = connected_components(A, directed=False)
bo = np.bincount(grp)
print(f"görüntü {len(T)} | kopya grubu {ince_n} | oturum/alt koleksiyon grubu {ng} | en büyük {np.sort(bo)[::-1][:6]}",
      flush=True)
print(pd.crosstab(F, T.label).assign(n=lambda d: d.sum(1)).sort_values("n", ascending=False).head(15).to_string())
BUT = [round(o * len(T)) for o in (0.05, 0.10, 0.20)]
TEKRAR = 10


def bol(r):
    """Grupları rastgele iki yarıya atar; her sınıfın ve toplamın %30-70'i testte olana dek yeniden çeker."""
    rng = np.random.default_rng(1000 + r)
    while True:
        tg = rng.random(ng) < 0.5
        te = tg[grp]
        oran = [te[y == c].mean() for c in (1, 2, 3)]
        if all(0.3 <= o_ <= 0.7 for o_ in oran) and 0.35 <= te.mean() <= 0.65:
            return np.where(te)[0], np.where(~te)[0]  # test, havuz


for r in range(TEKRAR):
    ti, hv = bol(r)
    print(r, "test", len(ti), np.bincount(y[ti], minlength=4)[1:], "havuz", len(hv), np.bincount(y[hv], minlength=4)[1:])
if KURU:
    sys.exit()

NM = "convnext_tiny.fb_in22k_ft_in1k"
nrm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
tf_e = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                           transforms.ToTensor(), nrm])
tf_t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), nrm])
IM = {}
im = lambda p: IM.setdefault(p, ac(p).resize((256, 256)))
AGIR = "runs_asama2/grouped0_convnext_tiny_224_s42_kat_v1_asama2_none/best.pt"


def model(bas):
    m = timm.create_model(NM, pretrained=(bas == "imagenet"), num_classes=4)
    if bas == "solarfcd":
        m.load_state_dict(torch.load(AGIR, map_location="cpu"))
    return m.to(cih)


@torch.no_grad()
def tahmin(m, idx):
    m.eval()
    P = []
    for i in range(0, len(idx), 64):
        x = torch.stack([tf_t(im(T.path[j])) for j in idx[i:i + 64]]).to(cih)
        P.append(torch.stack([m(x).softmax(1), m(x.flip(3)).softmax(1), m(x.flip(2)).softmax(1)]).mean(0).cpu())
    return torch.cat(P).numpy().argmax(1)


def egit(m, idx, tohum):
    torch.manual_seed(tohum)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.05)
    for ep in range(15):
        m.train()
        sira = idx[np.random.default_rng(tohum * 1000 + ep).permutation(len(idx))]
        for i in range(0, len(sira), 16):
            b = sira[i:i + 16]
            if len(b) < 2:
                continue
            x = torch.stack([tf_e(im(T.path[j])) for j in b]).to(cih)
            loss = torch.nn.functional.cross_entropy(m(x), torch.tensor(y[b]).to(cih))
            opt.zero_grad(); loss.backward(); opt.step()
    return m


@torch.no_grad()
def oznitelik(bas):
    m = model(bas).eval()
    Z = []
    for i in range(0, len(T), 64):
        x = torch.stack([tf_t(im(p)) for p in T.path[i:i + 64]]).to(cih)
        Z.append(torch.nn.functional.normalize(m.forward_head(m.forward_features(x), pre_logits=True), dim=1).cpu())
    return torch.cat(Z).numpy()


def olc(yy, p):
    d = dict(acc=(p == yy).mean())
    for c, ad in ((1, "H"), (2, "SF"), (3, "SO")):
        d[f"rec_{ad}"] = ((p == c) & (yy == c)).sum() / max((yy == c).sum(), 1)
    d["f1"] = np.mean([2 * ((p == c) & (yy == c)).sum() / max((p == c).sum() + (yy == c).sum(), 1) for c in (1, 2, 3)])
    return d


Z = {b: oznitelik(b) for b in ("imagenet", "solarfcd")}
R = []
for r in range(TEKRAR):
    ti, hv = bol(r)
    hg = np.unique(ince[hv])
    temel = dict(tekrar=r, n_test=len(ti), test_cogunluk=np.bincount(y[ti]).max() / len(ti))
    m = model("solarfcd")
    R.append(dict(temel, butce=0, n_egitim=0, yontem="ince_ayar_solarfcd", **olc(y[ti], tahmin(m, ti))))
    print(R[-1], flush=True)
    for bi, b in enumerate(BUT):
        sira = np.random.default_rng(10 * r + bi).permutation(hg)
        sec, n = [], 0
        for g_ in sira:
            if n >= b:
                break
            sec.append(g_); n += (ince[hv] == g_).sum()
        tr = hv[np.isin(ince[hv], sec)]
        d0 = dict(temel, butce=b, n_egitim=len(tr))
        for bas in ("solarfcd", "imagenet"):
            m = egit(model(bas), tr, r)
            R.append(dict(d0, yontem=f"ince_ayar_{bas}", **olc(y[ti], tahmin(m, ti))))
            print(R[-1], flush=True)
            s = Z[bas][ti] @ Z[bas][tr].T
            R.append(dict(d0, yontem=f"1nn_{bas}", **olc(y[ti], y[tr][s.argmax(1)])))
            lr = LogisticRegression(C=1.0, max_iter=5000).fit(Z[bas][tr], y[tr])
            R.append(dict(d0, yontem=f"dogrusal_{bas}", **olc(y[ti], lr.predict(Z[bas][ti]))))
        del m
        torch.cuda.empty_cache()
    pd.DataFrame(R).to_csv("asama2_sonuclar/az_ornek_gruplu.csv", index=False)
R = pd.DataFrame(R)
R.to_csv("asama2_sonuclar/az_ornek_gruplu.csv", index=False)
o = R.groupby(["butce", "yontem"]).agg(n=("n_egitim", "mean"), acc=("acc", "mean"), acc_sd=("acc", "std"),
                                         acc_min=("acc", "min"), acc_max=("acc", "max"), f1=("f1", "mean"),
                                         rec_H=("rec_H", "mean"), rec_SF=("rec_SF", "mean"),
                                         rec_SO=("rec_SO", "mean")).round(4)
bas = (f"Roboflow {len(T)} görüntü | kopya grubu {ince_n} | oturum/alt koleksiyon grubu {ng} | tekrar {TEKRAR}\n"
       f"test çoğunluk tabanı ort. {R.test_cogunluk.mean():.4f} | test n ort. {R.n_test.mean():.1f}\n")
open("asama2_sonuclar/az_ornek_gruplu.txt", "w", encoding="utf-8").write(bas + o.to_string())
print(bas + o.to_string())
