"""Adım 29b - Grad-CAM şeklinin 300 dpi sürümü (gönderim öncesi iç değerlendirme O1, Fig. 10). 29_gradcam.py ile aynı örnekler, aynı
modeller (ConvNeXt-T 224, tohum 42, her görüntü kendi kat modeliyle, OOF) ve aynı Grad-CAM hesabı (CPU). Farklar:
- Grad-CAM özgün (bulanıklaştırılmamış) görüntü üzerinde hesaplanır; yalnız GÖSTERİMDE, filigran taşıyan tek görüntüde
  (Imgdirty_207_1.jpg) şirket adı yazısı ('CRYSTAL SOLAR CLEANING') Gauss bulanıklığıyla okunamaz hale getirilir.
  Filigranın panel logosu ve yazı bloğu görünür kalır (modelin filigrana bakması şeklin konusudur).
- 13 cm baskı genişliğinde okunabilirlik için 7,2 inç genişlik, iki satırlı başlıklar, 300 dpi.
Çıktı: asama2_sonuclar/gradcam_v2.png (asama2_sonuclar/gradcam.png değişmez)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms

SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
K = {"Electrical_Fault": "EF", "Healthy": "H", "Structural_Fault": "SF", "Surface_Obstruction": "SO"}
R = "runs_asama2/grouped{}_convnext_tiny_224_s42_kat_v1_asama2_none"
# filigran yazısının kutusu (özgün 1080x1080 piksel koordinatı; logo hariç, yalnız şirket adı)
FILIGRAN = {"Imgdirty_207_1.jpg": ((440, 866, 962, 1006), 16)}
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})

D = pd.concat([pd.read_csv(R.format(k) + "/test_predictions.csv").assign(kat=k) for k in range(5)])
S = []
for c in SIN:
    S.append(D[(D.label == c) & (D.tahmin == c)].sample(2, random_state=0))
for a, b in [("Healthy", "Surface_Obstruction"), ("Surface_Obstruction", "Healthy"), ("Electrical_Fault", "Structural_Fault"),
             ("Structural_Fault", "Electrical_Fault")]:
    x = D[(D.label == a) & (D.tahmin == b)]
    S.append(x.sample(min(2, len(x)), random_state=0))
S = pd.concat(S).reset_index(drop=True)
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
M = {}
nr = -(-len(S) // 4)
fig, ax = plt.subplots(nr, 4, figsize=(7.2, 2.12 * nr))
bulanik = 0
for i, r in S.iterrows():
    if r.kat not in M:
        m = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=False, num_classes=4)
        m.load_state_dict(torch.load(R.format(r.kat) + "/best.pt", map_location="cpu"))
        M[r.kat] = m.eval()
    m = M[r.kat]
    st = {}
    h = m.stages[-1].register_forward_hook(lambda mod, i_, o: [st.__setitem__("a", o), o.register_hook(lambda g: st.__setitem__("g", g))] and None)
    im = ImageOps.exif_transpose(Image.open(r.path)).convert("RGB")
    out = m(tf(im)[None])
    out[0, SIN.index(r.tahmin)].backward()
    h.remove()
    w = st["g"].mean((2, 3), keepdim=True)
    cam = torch.relu((w * st["a"]).sum(1))[0].detach()
    cam = torch.nn.functional.interpolate(cam[None, None], size=(224, 224), mode="bilinear")[0, 0].numpy()
    cam = cam / (cam.max() + 1e-8)
    goster = im
    ad = Path(r.path).name
    if ad in FILIGRAN:  # yalnız gösterim: şirket adını bulanıklaştır (Grad-CAM yukarıda özgün görüntüden hesaplandı)
        kutu, yaricap = FILIGRAN[ad]
        assert im.size == (1080, 1080), im.size
        goster = im.copy()
        goster.paste(goster.crop(kutu).filter(ImageFilter.GaussianBlur(yaricap)), kutu)
        bulanik += 1
    a = ax.flat[i]
    a.imshow(goster.resize((224, 224)))
    a.imshow(cam, cmap="jet", alpha=0.4)
    dogru = r.label == r.tahmin
    a.set_title(f"{'✓' if dogru else '✗'} label {K[r.label]}\n→ predicted {K[r.tahmin]}", fontsize=9,
                color="green" if dogru else "red", linespacing=1.15, pad=3)
    a.axis("off")
for a in ax.flat[len(S):]:
    a.axis("off")
assert bulanik == 1, bulanik
plt.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.5)
plt.savefig("asama2_sonuclar/gradcam_v2.png", dpi=300, bbox_inches="tight")
print(len(S), "örnek; bulanıklaştırılan filigran:", bulanik)
