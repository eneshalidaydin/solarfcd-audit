"""Adım 28 - Az örnekle hedef alan uyarlaması (kullanıcı onayı 22 Kasım). Roboflow dış test (örtüşmeyen, 4 sınıf etiketi
bilinen 1.108). Sızıntı önlemi: Roboflow içi yakın kopyalar (".rf." öncesi ad kökü + pHash Hamming ≤ 6, bağlı bileşen)
tek grup; bölme grup düzeyinde. Oranlar %0/5/10/20 (eğitim), kalan grup havuzunun sabit %50'si test (her tekrar aynı
test, 3 tekrar farklı eğitim örneklemi). Başlangıç: (i) SolarFCD ConvNeXt-T (kat 0, tohum 42, orijinal etiket);
(ii) yalnız ImageNet (convnext_tiny.fb_in22k_ft_in1k). 15 epok, AdamW lr 1e-4, bs 16, yatay/dikey çevirme, son epok, TTA.
Metrik: acc, makro-F1 (dış sette bulunan 3 sınıf: H, SF, SO). Çıktı: asama2_sonuclar/az_ornek.txt, .csv"""
import re
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageOps
from scipy.sparse.csgraph import connected_components
from torchvision import transforms

SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
cih = "cuda"
T = pd.read_csv("dis_dogrulama/roboflow/ortusme.csv", encoding="utf-8-sig")
T = T[~T.ortusen & T.label.isin(SIN)].reset_index(drop=True)
y = T.label.map(SIN.index).to_numpy()
ac = lambda p: ImageOps.exif_transpose(Image.open(p)).convert("RGB")
kok = T.path.map(lambda p: re.split(r"[._]rf[._]|_jpg\.rf", Path(p).name)[0])
H = np.array([imagehash.phash(ac(p)).hash.flatten() for p in T.path])
A = (H[:, None, :] != H[None, :, :]).sum(2) <= 6
A |= (kok.to_numpy()[:, None] == kok.to_numpy()[None, :])
ng, grp = connected_components(A, directed=False)
print("görüntü", len(T), "grup", ng, flush=True)
rng = np.random.default_rng(20261122)
gp = rng.permutation(ng)
test_g, havuz_g = gp[: ng // 2], gp[ng // 2:]
te = np.isin(grp, test_g)
NM = "convnext_tiny.fb_in22k_ft_in1k"
tf_e = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                           transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
tf_t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                           transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
IM = {}
im = lambda p: IM.setdefault(p, ac(p).resize((256, 256)))


def model(bas):
    m = timm.create_model(NM, pretrained=(bas == "imagenet"), num_classes=4)
    if bas == "solarfcd":
        m.load_state_dict(torch.load("runs_asama2/grouped0_convnext_tiny_224_s42_kat_v1_asama2_none/best.pt", map_location="cpu"))
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
    for _ in range(15):
        m.train()
        for i in range(0, len(idx), 16):
            b = np.random.default_rng(tohum + i + _ * 997).permutation(idx)[:16] if False else idx[np.random.default_rng(tohum * 1000 + _).permutation(len(idx))][i:i + 16]
            if len(b) < 2:
                continue
            x = torch.stack([tf_e(im(T.path[j])) for j in b]).to(cih)
            loss = torch.nn.functional.cross_entropy(m(x), torch.tensor(y[b]).to(cih))
            opt.zero_grad(); loss.backward(); opt.step()
    return m


def f1(yy, p):
    return np.mean([2 * ((p == c) & (yy == c)).sum() / max((p == c).sum() + (yy == c).sum(), 1) for c in (1, 2, 3)])


ti = np.where(te)[0]
R = []
for oran in (0.0, 0.05, 0.10, 0.20):
    for tek in range(3 if oran else 1):
        r2 = np.random.default_rng(100 * tek + 7)
        hg = r2.permutation(havuz_g)
        sec, n = [], 0
        for g_ in hg:
            if n >= oran * len(T):
                break
            sec.append(g_); n += (grp == g_).sum()
        tr = np.where(np.isin(grp, sec))[0]
        for bas in ("solarfcd", "imagenet"):
            if oran == 0 and bas == "imagenet":
                continue
            m = model(bas)
            if len(tr):
                m = egit(m, tr, tek)
            p = tahmin(m, ti)
            R.append(dict(oran=oran, tekrar=tek, baslangic=bas, n_egitim=len(tr), n_test=len(ti),
                          acc=(p == y[ti]).mean(), f1=f1(y[ti], p)))
            print(R[-1], flush=True)
R = pd.DataFrame(R)
R.to_csv("asama2_sonuclar/az_ornek.csv", index=False)
o = R.groupby(["oran", "baslangic"]).agg(n=("n_egitim", "mean"), acc=("acc", "mean"), acc_sd=("acc", "std"),
                                          f1=("f1", "mean"), f1_sd=("f1", "std")).round(4)
open("asama2_sonuclar/az_ornek.txt", "w", encoding="utf-8").write(f"Roboflow {len(T)} görüntü, {ng} grup; test {len(ti)}\n" + o.to_string())
print(o.to_string())
