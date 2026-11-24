"""Adım 29 - Grad-CAM nitel analiz (kullanıcı onayı 22 Kasım). ConvNeXt-T 224, tohum 42; her görüntü kendi kat modeliyle
(OOF). Örnekler (tohum 0): her sınıf için 2 doğru + en sık karışma H<->SO ve EF/SF hatalarından 2'şer. Son evre (stages[-1])
çıktısına göre Grad-CAM (tahmin edilen sınıf). Çıktı: asama2_sonuclar/gradcam.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageOps
from torchvision import transforms

SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
K = {"Electrical_Fault": "EF", "Healthy": "H", "Structural_Fault": "SF", "Surface_Obstruction": "SO"}
R = "runs_asama2/grouped{}_convnext_tiny_224_s42_kat_v1_asama2_none"
D = pd.concat([pd.read_csv(R.format(k) + "/test_predictions.csv").assign(kat=k) for k in range(5)])
rng = np.random.default_rng(0)
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
fig, ax = plt.subplots(-(-len(S) // 4), 4, figsize=(13, 3.4 * -(-len(S) // 4)))
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
    a = ax.flat[i]
    a.imshow(im.resize((224, 224)))
    a.imshow(cam, cmap="jet", alpha=0.4)
    a.set_title(f"{'✓' if r.label == r.tahmin else '✗'} label {K[r.label]} → predicted {K[r.tahmin]}", fontsize=11,
                color="green" if r.label == r.tahmin else "red")
    a.axis("off")
for a in ax.flat[len(S):]:
    a.axis("off")
plt.tight_layout()
plt.savefig("asama2_sonuclar/gradcam.png", dpi=150)
print(len(S), "örnek")
