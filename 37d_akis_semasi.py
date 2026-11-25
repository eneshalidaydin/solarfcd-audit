"""Adım 37d - Sınıflandırma ve ön eleme akış şeması (yeni ana metin şekli, metindeki yer tutucusu [AKIS]).

Amaç: eğitimden (sahne gruplu çapraz doğrulama) yeni bir görüntünün E3 ile sınıflandırılıp seçici olarak kabul ya da
insan incelemesine yönlendirilmesine ve işletme-bakım (O&M) kullanımına kadar olan akışı tek şemada göstermek.
Şekil 1'in alt O&M şeridi bu şemaya taşındı (31_makale_ek_sekiller.py, Şekil 1 bloğu). Yeni hesap ya da eğitim yok,
ham veriye dokunulmaz.
Girdi: yok. Kutulardaki tüm bilgiler makale/en_kaynak6.md'den alınır: Yöntem paragrafları "Of the three protocols
compared" (artırma, AdamW, 30 epok, sahne gruplu beş kat CV, üç tohum, TTA), "Six pre-trained backbones" (altı timm
omurgası, 224 piksel, DINOv2 392 piksel, E3 üyeleri), "The selective classifier accepts an image" (LTT, α = %5,
δ = 0,1), "Least-ambiguous set-valued classifier" (LAC, ε = 0,05) ve Şekil 1 (2.095 görüntü, O&M adımları).
Çıktı: makale/fig_pipeline.png (300 dpi, basılı 16 cm genişlik, en fazla 6,5 cm yükseklik, en küçük yazı 7 pt).
Koordinatlar cm cinsindendir, figür basılı boyutta çizilir (bbox_inches="tight" her kenara 0,1 inç ekler).
Denetim düzeltmesi (25 Kasım): alt şerit başlığı "Intended O&M use" oldu (kullanım öngörülen, sahada doğrulanmadı),
"rejected" etiketi eğik okla çakışmayacak biçimde etiketin alt kenarına göre konumlandı, "accepted" ok ucundan ayrıldı."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("makale")
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, C4, GRI = "#3b6ea5", "#d9822b", "#2e8b57", "#c0392b", "#9e9e9e"
DOLGU = {C1: "#eef3f9", C2: "#f6efe6", C3: "#eaf4ee"}  # kutu iç renkleri Şekil 1 ile aynı
SERIT = {C1: "#f7f9fc", C2: "#fdf9f4", C3: "#f4f9f6"}  # şerit zeminleri (eski O&M şeridiyle aynı ton)
GEN, YUK_MAX = 16.0, 6.5  # basılı boyut sınırı (cm)
FS, FS_BASLIK = 7.0, 7.5  # pt
MX, MY, PAD = 0.15, 0.11, 0.07  # kutu iç boşlukları ve köşe dolgusu (cm)
W = GEN - 0.508

fig = plt.figure(figsize=(W / 2.54, 2.0))
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
RND = fig.canvas.get_renderer()


def olc(t, fs=FS, **kw):
    """Metnin cm cinsinden genişliği ve yüksekliği."""
    tt = ax.text(0, 0, t, fontsize=fs, ha="center", va="center", **kw)
    bb = tt.get_window_extent(RND)
    tt.remove()
    return bb.width / fig.dpi * 2.54, bb.height / fig.dpi * 2.54


class Kutu:
    """Metne göre boyutlanan kutu; konum sonradan verilir (xc, yc: merkez)."""

    def __init__(self, t, renk, kesik=False):
        self.t, self.renk, self.kesik = t, renk, kesik
        w, h = olc(t)
        self.w, self.h = w + 2 * MX, h + 2 * MY
        self.xc = self.yc = 0.0

    x0 = property(lambda s: s.xc - s.w / 2)
    x1 = property(lambda s: s.xc + s.w / 2)
    y0 = property(lambda s: s.yc - s.h / 2)
    y1 = property(lambda s: s.yc + s.h / 2)

    def ciz(self):
        ax.add_patch(FancyBboxPatch((self.x0 + PAD, self.y0 + PAD), self.w - 2 * PAD, self.h - 2 * PAD,
                                    boxstyle=f"round,pad={PAD}", fc=DOLGU[self.renk], ec=self.renk, lw=1,
                                    ls=(0, (3, 1.5)) if self.kesik else "-"))
        ax.text(self.xc, self.yc, self.t, ha="center", va="center", fontsize=FS)


def ok(x1, y1, x2, y2, kesik=False):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=0.9, ls=(0, (3, 1.5)) if kesik else "-"))


def etiket(x, y, t, ha="left", va="center"):
    ax.text(x, y, t, ha=ha, va=va, fontsize=FS, color="#555", style="italic")


# ---------- kutular (metinler en_kaynak6.md Yöntem bölümünden) ----------
T1 = Kutu("Labelled RGB\nimages (2,095)\n4 classes", C1)
T2 = Kutu("Augmentation: random\nresized crop, flip,\nrotation, colour jitter", C1)
T3 = Kutu("Fine-tuning of 6 pre-trained timm backbones\n224 px (DINOv2 also 392 px), AdamW, 30 epochs\n"
          "scene-grouped 5-fold CV × 3 seeds", C1)
I1 = Kutu("TTA: original\n+ two flipped\nviews", C2)
I2 = Kutu("E3: mean of\nConvNeXt-T (224 px)\nSwin-T (224 px)\nDINOv2 (392 px)", C2)
I3 = Kutu("Class\nprobabilities p\nH · SO · SF · EF", C2)
I4 = Kutu("Accept if max p >\nLTT threshold\n(α = 5%, δ = 0.1)", C2)
LAC = Kutu("Alternative output: LAC\nconformal set (ε = 0.05)", C2, kesik=True)
O0 = Kutu("Field image\n(drone or\nhandheld)", C3)
O1 = Kutu("Human review\nof rejected\nimages", C3)
O2 = Kutu("Electrical confirmation\n(e.g. string current\nor I–V data)", C3)
O3 = Kutu("Action:\ncleaning\nor repair", C3)
SERITLER = [(C1, "Training"), (C2, "Inference\nand selective\ntriage"), (C3, "Intended\nO&M use")]  # O&M kullanımı öngörülen, doğrulanmış değil

# ---------- yatay yerleşim ----------
KEN = 0.03  # dış kenar (cm)
XB = KEN + 0.14  # şerit başlıklarının sol kenarı
XL = XB + max(olc(ad, FS_BASLIK, fontweight="bold")[0] for _, ad in SERITLER) + 0.22
SAG = W - KEN - 0.14  # en sağdaki kutunun sağ kenarı
# ok boyu A, O&M şeridi sağ kenara ulaşacak biçimde: O3.x1 = XL + I1..I3 + I4/2 + O2/2 + O3 + 4A
A = (SAG - XL - I1.w - I2.w - I3.w - I4.w / 2 - O2.w / 2 - O3.w) / 4
assert A >= 0.42, f"oklar çok kısa: {A:.2f} cm"
A = min(A, 0.9)
x = XL
for k in (I1, I2, I3, I4):  # çıkarım şeridi soldan sağa
    k.xc = x + k.w / 2
    x = k.x1 + A
LAC.xc = I4.xc + 0.25  # olasılık kutusundan gelen kesik ok için biraz sağa
O0.xc = I1.xc  # saha görüntüsü TTA'nın altında (yukarı ok)
O2.xc = I4.xc  # kabul edilenler doğrudan elektriksel doğrulamaya (aşağı ok)
O1.xc = O2.x0 - A - O1.w / 2
O3.xc = O2.x1 + A + O3.w / 2
x = XL
for k in (T1, T2, T3):  # eğitim şeridi
    k.xc = x + k.w / 2
    x = k.x1 + A
assert max(T3.x1, O3.x1, LAC.x1) <= SAG + 1e-9

# ---------- dikey yerleşim (alttan üste) ----------
BM, GB12, GB23 = 0.10, 0.12, 0.40  # şerit iç kenarı, şeritler arası boşluklar (cm; alttaki ok etiketleri için)
h3 = max(k.h for k in (O0, O1, O2, O3))
s3 = (KEN, KEN + h3 + 2 * BM)
for k in (O0, O1, O2, O3):
    k.yc = (s3[0] + s3[1]) / 2
h2 = max(k.h for k in (I1, I2, I3, I4))
y2c = s3[1] + GB23 + BM + h2 / 2
for k in (I1, I2, I3, I4):
    k.yc = y2c
LAC.yc = I4.y1 + 0.20 + LAC.h / 2
s2 = (s3[1] + GB23, max(LAC.y1, I2.y1) + BM)
h1 = max(k.h for k in (T1, T2, T3))
s1 = (s2[1] + GB12, s2[1] + GB12 + h1 + 2 * BM)
for k in (T1, T2, T3):
    k.yc = (s1[0] + s1[1]) / 2
H = s1[1] + KEN
fig.set_size_inches(W / 2.54, H / 2.54)
ax.set_xlim(0, W); ax.set_ylim(0, H)
assert H + 0.508 <= YUK_MAX + 1e-9, f"yükseklik {H + 0.508:.2f} cm > {YUK_MAX} cm"

# ---------- şeritler ve başlıklar ----------
for (ya, yb), (renk, ad) in zip((s1, s2, s3), SERITLER):
    ax.add_patch(FancyBboxPatch((KEN, ya), W - 2 * KEN, yb - ya, boxstyle="round,pad=0,rounding_size=0.17",
                                fc=SERIT[renk], ec=renk, lw=0.8, ls=(0, (4, 2))))
    ax.text(XB, (ya + yb) / 2, ad, ha="left", va="center", fontsize=FS_BASLIK, color=renk, fontweight="bold")

for k in (T1, T2, T3, I1, I2, I3, I4, LAC, O0, O1, O2, O3):
    k.ciz()

# ---------- oklar ----------
for a, b in ((T1, T2), (T2, T3), (I1, I2), (I2, I3), (I3, I4), (O1, O2), (O2, O3)):
    ok(a.x1, a.yc, b.x0, b.yc)
# eğitilmiş modeller -> E3: eğitim kutusunun altından inip E3'ün üstüne dirsekli ok
xv = T3.x0 + 0.35
yh = (I2.y1 + s2[1]) / 2 - 0.08
assert xv > I2.x1 + 0.2 and yh > max(I3.y1, I1.y1) + 0.1 and xv < LAC.x0 - 0.3
ax.plot([xv, xv, I2.xc], [T3.y0, yh, yh], color="#555", lw=0.9, solid_capstyle="butt", solid_joinstyle="miter")
ok(I2.xc, yh + 0.005, I2.xc, I2.y1)
etiket((xv + I2.xc) / 2, yh + 0.06, "trained models", ha="center", va="bottom")
ok(O0.xc, O0.y1, I1.xc, I1.y0)  # saha görüntüsü -> TTA
ok(I3.x1 - 0.3, I3.y1, LAC.x0, LAC.yc, kesik=True)
ya = s3[1] + GB23 / 2  # ok etiketleri iki şerit arasındaki boşluğun ortasında
ok(I4.xc, I4.y0, O2.xc, O2.y1)
etiket(I4.xc + 0.15, ya, "accepted")  # ok ucundan ayrık dursun
xr0, xr1 = I4.x0 + 0.25, O1.x1 - 0.35
ok(xr0, I4.y0, xr1, O1.y1)
# eğik ok sola-aşağı indiği için etiketin sağ alt köşesi oka en yakın noktadır: x etiketin alt kenarında hesaplanır
he = olc("rejected", style="italic")[1]
xe = xr0 + (xr1 - xr0) * (I4.y0 - (ya - he / 2)) / (I4.y0 - O1.y1)
etiket(xe - 0.10, ya, "rejected", ha="right")

plt.savefig(OUT / "fig_pipeline.png", dpi=300, bbox_inches="tight")

def pdf_kaydet(ad, fig=None):
    """Dergiye ayrı yüklenecek vektörel sürüm (SEKIL_PDF=1 iken), makale/yuksek/<ad>.pdf; PNG çıktısı değişmez."""
    import os
    if not os.environ.get("SEKIL_PDF"):
        return
    (OUT / "yuksek").mkdir(exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42}):
        (fig or plt.gcf()).savefig(OUT / "yuksek" / (Path(ad).stem + ".pdf"), bbox_inches="tight", dpi=600)
pdf_kaydet("fig_pipeline.png")
plt.close()
print(f"fig_pipeline.png basılı boyut: {GEN:.1f} × {H + 0.508:.2f} cm; ok boyu {A:.2f} cm; XL {XL:.2f} cm")
