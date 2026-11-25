"""Adım 37b - Dış veri setlerinden örnek görüntüler ve bir örtüşme taraması örneği (Renewable Energy gönderimi,
25 Kasım kullanıcı isteği: metinde anlatılanı şekle dökmek). Yeni eğitim ya da yeni sonuç yoktur; yalnız mevcut
çıktılar (dis_dogrulama/roboflow/ortusme.csv, dis_dogrulama/melbourne/ortusme.csv, asama2/manifest_asama2.csv) ve
ham görüntüler okunur.
(a) Örtüşme taramasından sonra kalan RF-Fault görüntüleri, metindeki eşleme öncelik sırasıyla (soldan sağa SF, yalnız
    arıza var/yok analizine giren türü bilinmeyen arıza, SO, H) birer görüntü. Başlıkta kutu sınıfları öncelik
    sırasıyla ve eşlenen sınıf yazılır. Kutu ve çokgen dış hatları, sınıfın eşlendiği grubun rengindedir (Physical
    Damage kırmızı, Defective turuncu, kirlenme ya da kar mavi, Non Defective yeşil).
(b) Taramadan sonra kalan bir RF-Soiling görüntüsü (her kutu SO).
(c) Taramayla çıkarılan bir RF-Fault görüntüsü ve DINOv2 kosinüsüne göre en yakın SolarFCD görüntüsü (ortusme.csv,
    'en_yakin'). Çiftin pHash uzaklığı 25_dis_dogrulama.py ile aynı biçimde yeniden hesaplanır ve ortusme.csv'deki en
    küçük uzaklığa eşit olduğu denetlenir.
Seçim: her eşlenen sınıfın taramadan sonra en sık görülen kutu sınıfı birleşimi (SO: Dusty; SF: Defective + Physical
Damage; arıza: Defective + Snow; RF-Soiling: Dust + bird-droppings) içinden, alan kaymasını gösteren (dış mekân,
yere konmuş panel, karlı çatı), insan (ayakkabı dahil), okunur firma adı ve plaka içermeyen görüntüler elle seçilmiştir.
Seçilen kalan görüntülerin aynı kaynak adlı Roboflow türevlerinin hiçbiri taramada çıkarılmamıştır. Kaggle PV Panel
Defect görüntüleri lisansı bilinmediği için gösterilmez; (c)'deki SolarFCD görüntüsü K (Kaggle temiz/kirli arşivi)
ailesindendir. Ham veri (C:/solarfcd/data) salt okunur.
Çıktı: makale/fig_external_examples.png (basılı genişlik 16 cm, 300 dpi)."""
from pathlib import Path

import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.patches import Polygon
from PIL import Image, ImageOps

OUT = Path("makale")
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, C4 = "#3b6ea5", "#d9822b", "#2e8b57", "#c0392b"
RF_AD = ["Bird Drop", "Defective", "Dust", "Dusty", "Non Defective", "Physical Damage", "Snow"]  # roboflow/data.yaml
RS_AD = ["Dust", "bird-droppings"]                                                              # melbourne/.../data.yaml
RF_RENK = {0: C1, 1: C2, 2: C1, 3: C1, 4: C3, 5: C4, 6: C1}
RS_RENK = {0: C1, 1: C1}
ONCELIK = [5, 1, 0, 2, 3, 6, 4]   # Physical Damage > Defective > kirlenme/kar > Non Defective (25_dis_dogrulama.rob)
ESL = {"Healthy": "H", "Surface_Obstruction": "SO", "Structural_Fault": "SF", "Ariza": "unknown fault"}

R = pd.read_csv("dis_dogrulama/roboflow/ortusme.csv", encoding="utf-8-sig")
M = pd.read_csv("dis_dogrulama/melbourne/ortusme.csv", encoding="utf-8-sig")
S = pd.read_csv("asama2/manifest_asama2.csv", encoding="utf-8-sig").set_index("rel_path")
R["ust"], M["ust"] = R.ust.astype(str), M.ust.astype(str)


def ac(p):
    return ImageOps.exif_transpose(Image.open(p)).convert("RGB")


def sec(X, ad):
    s = X[X.path.str.replace("\\", "/", regex=False).str.endswith(ad)]
    assert len(s) == 1, ad
    return s.iloc[0]


def etiketler(p):
    """YOLO etiket satırları: 5 değer = kutu (cx, cy, w, h), daha fazlası = çokgen (x1 y1 x2 y2 ...)."""
    p = Path(p)
    out = []
    for s in (p.parent.parent / "labels" / (p.stem + ".txt")).read_text().splitlines():
        v = s.split()
        if not v:
            continue
        c, z = int(v[0]), list(map(float, v[1:]))
        if len(z) == 4:
            cx, cy, w, h = z
            z = [cx - w / 2, cy - h / 2, cx + w / 2, cy - h / 2, cx + w / 2, cy + h / 2, cx - w / 2, cy + h / 2]
        out.append((c, list(zip(z[0::2], z[1::2]))))
    return out


# ---------- seçim ve denetimler ----------
T = R[~R.ortusen]
en_sik = T.groupby("label").ust.agg(lambda s: s.value_counts().index[0]).to_dict()
A = [sec(R, ad) for ad in ("train/images/Physical_Damge_290_jpg.rf.51d7951053a32d4e6cbcea5feadcd6f6.jpg",
                           "test/images/18_png_jpg.rf.92b68e8fe9a8c709a3078ada4e91ad53.jpg",
                           "/IMG_20241003_094255_737_jpg.rf.ed14d252b4b2a0eb9ee6519b1f987ff6.jpg",
                           "valid/images/IMG_20230325_091003_jpg.rf.f027f05967a2277e778c8d898f7ec1e2.jpg")]
for r, lab in zip(A, ["Structural_Fault", "Ariza", "Surface_Obstruction", "Healthy"]):   # metindeki öncelik sırası
    assert (not r.ortusen) and r.label == lab and r.ust == en_sik[lab], (r.path, r.ust, en_sik[lab])
B = sec(M, "train/images/ROI_58_jpg.rf.e1c58e6c7e9e84e5217b335352b7987c.jpg")
assert (not B.ortusen) and B.ust == M[~M.ortusen].ust.value_counts().index[0]
kok = lambda p: Path(p).name.split(".rf.")[0]
for X, r in [(R, a) for a in A] + [(M, B)]:
    assert not X[X.path.map(kok) == kok(r.path)].ortusen.any(), r.path
C = sec(R, "train/images/Imgdirty_269_1_jpg.rf.44e767a4f082484c9059b71af26de1ab.jpg")
assert C.ortusen and not C.md5_eslesme and C.en_yakin == "Train/Surface_Obstruction/Imgdirty_269_1.jpg"
assert S.loc[C.en_yakin, "kaynak"] == "kaggle_toz"
ph_cift = int(imagehash.phash(ac(C.path)) - imagehash.phash(ac(S.loc[C.en_yakin, "path"])))
assert ph_cift == C.ph_min == 10 and C.cos_max >= 0.85, (ph_cift, C.ph_min, C.cos_max)
print("en sık birleşimler:", en_sik, "| RF-Soiling:", B.ust)
print(f"(c) {Path(C.path).name} -> {C.en_yakin}: pHash {ph_cift}, kosinüs {C.cos_max:.4f}")

# ---------- yerleşim (cm) ----------
t, gi, go = 2.16, 0.07, 0.28          # karo yüksekliği, grup içi ve gruplar arası boşluk
y0, lh = 0.05, 0.30                   # karo alt kenarı, başlık satır aralığı
sol = S.loc[C.en_yakin, "path"]
oran = ac(sol).size[0] / ac(sol).size[1]
xa = [i * (t + gi) for i in range(4)]
xb = xa[-1] + t + go
xc = [xb + t + go, xb + 2 * t + go + gi]
W, H = xc[1] + oran * t, y0 + t + 0.06 + 3 * lh + 0.08 + 0.33
fig = plt.figure(figsize=(W / 2.54, H / 2.54))
L = fig.add_axes([0, 0, 1, 1])
L.set_xlim(0, W); L.set_ylim(0, H); L.axis("off")
ust = y0 + t


def karo(x, p, w=t, kutu=None, rozet=None):
    a = fig.add_axes([x / W, y0 / H, w / W, t / H])
    im = ac(p)
    a.imshow(im, aspect="auto")
    a.set_xlim(0, im.size[0]); a.set_ylim(im.size[1], 0); a.axis("off")
    for c, nk in (etiketler(p) if kutu else []):
        a.add_patch(Polygon([(u * im.size[0], v * im.size[1]) for u, v in nk], closed=True, fill=False, ec=kutu[c],
                            lw=0.85, path_effects=[pe.Stroke(linewidth=1.8, foreground="white"), pe.Normal()]))
    if rozet:
        a.text(0.04, 0.96, rozet, transform=a.transAxes, ha="left", va="top", fontsize=7, color="white",
               bbox=dict(boxstyle="square,pad=0.2", fc="black", alpha=0.6, lw=0))


def baslik(xm, satirlar):
    for k, (s, renk) in enumerate(reversed(satirlar)):
        L.text(xm, ust + 0.06 + k * lh, s, ha="center", va="bottom", fontsize=7, color=renk)


def sinif_satirlari(r, adlar, renk, sira):
    cs = sorted({int(c) for c in r.ust.split(",")}, key=sira.index)
    assert cs == sorted({c for c, _ in etiketler(r.path)}, key=sira.index)
    return [(("+ " if i else "") + adlar[c], renk[c]) for i, c in enumerate(cs)]


for x, r in zip(xa, A):
    karo(x, r.path, kutu=RF_RENK)
    baslik(x + t / 2, sinif_satirlari(r, RF_AD, RF_RENK, ONCELIK) + [("→ " + ESL[r.label], "black")])
karo(xb, B.path, kutu=RS_RENK)
baslik(xb + t / 2, sinif_satirlari(B, RS_AD, RS_RENK, [0, 1]) + [("→ " + ESL[B.label], "black")])
karo(xc[0], C.path, rozet="RF-Fault")
karo(xc[1], sol, w=oran * t, rozet="SolarFCD")
baslik((xc[0] + W) / 2, [(f"pHash distance {ph_cift} (≤ 12)", "black"),
                         (f"DINOv2 cosine {C.cos_max:.2f} (≥ 0.85)", "black")])
for x, s in ((0, "(a) RF-Fault"), (xb, "(b) RF-Soiling"), (xc[0], "(c) Overlap screening")):
    L.text(x, ust + 0.06 + 3 * lh + 0.08, s, ha="left", va="bottom", fontsize=8)
plt.savefig(OUT / "fig_external_examples.png", dpi=300, bbox_inches="tight")

def pdf_kaydet(ad, fig=None):
    """Dergiye ayrı yüklenecek vektörel sürüm (SEKIL_PDF=1 iken), makale/yuksek/<ad>.pdf; PNG çıktısı değişmez."""
    import os
    if not os.environ.get("SEKIL_PDF"):
        return
    (OUT / "yuksek").mkdir(exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42}):
        (fig or plt.gcf()).savefig(OUT / "yuksek" / (Path(ad).stem + ".pdf"), bbox_inches="tight", dpi=600)
pdf_kaydet("fig_external_examples.png")
plt.close()
w_px, h_px = Image.open(OUT / "fig_external_examples.png").size
print(f"fig_external_examples.png {w_px}x{h_px} px = {w_px / 300 * 2.54:.2f} x {h_px / 300 * 2.54:.2f} cm; "
      f"16 cm genişlikte yükseklik {16 * h_px / w_px:.2f} cm")
