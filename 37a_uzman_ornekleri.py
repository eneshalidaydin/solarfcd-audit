"""Adım 37a - Uzman etiket denetimi için yeni ana metin şekli ve küçültülmüş Ek Şekil S4 (kullanıcı isteği 25 Kasım: metinle
anlatılanları şekle dökmek; uzman denetiminde uzmanların sağlıklı işaretlediği bir görüntü). Yeni eğitim ya da hesap yok.

Çıktılar (C:\\solarfcd\\pilot içinden çalıştırılır):
  makale/fig_expert_audit_examples.png  basılı genişlik 16 cm.
      (a) Orijinal etiket -> uzman uzlaşısı (en az 2/3) sayım matrisi, n = 150; sayılar eski fig06 (b) ile aynı.
      (b) Dört örnek görüntü ve altında orijinal etiket, 1-3. uzman yanıtları, uzlaşı ve E3 kat dışı (OOF) tahmini
          (en yüksek sınıf olasılığıyla). Örnekler (a'daki üç sonuçtan birer tane): E058 (SO, üç uzman da SO, korundu;
          E3 H dedi), E056 (SO, üç uzman da H, tek oybirlikli SO->H), E002 (H, üç uzman da SO), E037 (SO; emin
          değil/SO/H, belirsiz; belirsizlerin en sık örüntüsü H/SO/emin değil). Denetim düzeltmesi: ilk taslaktaki E000
          (EF; EF/SF/karışık, belirsiz) yerine korunmuş bir aday (E058) alındı. 112 adayın 48'i korunmuştur ve E3 bu
          48 adayın hiçbirinde uzlaşıyla uyuşmaz (belirsiz olmayan 77 adayın yalnız 28'inde uyuşur, Ek Tablo S12e);
          yalnız değişen ve belirsiz örnekler E3'ü uzmanlarla olduğundan uyumlu gösterirdi. Seçilen görüntülerde
          kişi, logo, filigran ya da okunur yazı yoktur (E012'de işçi, E019'da fotoğrafçının gölgesi, E068'de işçi ve
          şirket adı, E103'te filigran olduğu için seçilmedi), bu yüzden bulanıklaştırma gerekmedi. Kırpma ve yeniden
          boyutlandırma yalnız gösterim içindir; ham veriye yazılmaz.
  makale/figS4_expert_labels.png  basılı genişlik yaklaşık 9 cm; eski fig06 (a) paneli tek başına (uzman başına yanıt
      dağılımı), 1. uzmanın yanıtsız bıraktığı iki görüntü ayrı (taralı) dilim olarak gösterilir.
Veri (hepsi salt okunur):
  Makale_RenewableEnergy/solarfcd-audit/expert_audit.csv  uzman yanıtları ve uzlaşı (Zenodo paketi);
      insan_kontrol/etiket/sonuc.csv ile (eski fig06'nın kaynağı) satır satır aynı olduğu denetlenir.
  runs_asama2/grouped{k}_{omurga}_s{tohum}_kat_v1_asama2_none/test_predictions.csv  E3 OOF olasılıkları
      (31_makale_ek_sekiller.py oof() ile aynı); Zenodo oof_predictions.csv ile çapraz denetlenir.
  C:\\solarfcd\\data\\SolarFCD-Final\\<rel_path>  görüntüler (yalnız okunur).
Dikkat: expert_audit.csv'deki ensemble_prediction 224 piksellik işaretleme topluluğudur, E3 değildir; şekilde E3 kullanılır."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

OUT = Path("makale")
VERI = Path(r"<SOLARFCD_DIR>")
# expert_audit.csv ve oof_predictions.csv: depoda betikle aynı klasörde, pilotta paket klasöründe
DEPO = Path(__file__).resolve().parent
if not (DEPO / "expert_audit.csv").exists():
    DEPO = Path("<OUTPUT_DIR>/solarfcd-audit")
SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
K = ["EF", "H", "SF", "SO"]
PC = [f"p_{c}" for c in SIN]
E3 = ["convnext_tiny_224", "swin_tiny_patch4_window7_224_224", "vit_base_patch14_dinov2_392"]
YAN = dict(zip(SIN, K)) | {"mixed": "mixed", "not_sure": "not sure", "ambiguous": "ambiguous"}
plt.rcParams.update({"font.size": 7.5, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
C1, C2, C3, C4 = "#3b6ea5", "#d9822b", "#2e8b57", "#c0392b"
GRI = "#9e9e9e"
CM = 1 / 2.54


def pdf_kaydet(ad, fig=None):
    """Dergiye ayrı yüklenecek vektörel sürüm (SEKIL_PDF=1 iken), makale/yuksek/<ad>.pdf; PNG çıktısı değişmez."""
    import os
    if not os.environ.get("SEKIL_PDF"):
        return
    (OUT / "yuksek").mkdir(exist_ok=True)
    with plt.rc_context({"pdf.fonttype": 42}):
        (fig or plt.gcf()).savefig(OUT / "yuksek" / (Path(ad).stem + ".pdf"), bbox_inches="tight", dpi=600)


def kaydet(ad):
    plt.savefig(OUT / ad, dpi=300, bbox_inches="tight")
    pdf_kaydet(ad)
    plt.close()
    w, h = Image.open(OUT / ad).size
    print(f"{ad}: {w} x {h} px = {w / 300 * 2.54:.2f} x {h / 300 * 2.54:.2f} cm (300 dpi)")


# ---------- veri: uzman yanıtları (Zenodo) ve eski fig06 kaynağıyla denetim ----------
A = pd.read_csv(DEPO / "expert_audit.csv")
S = pd.read_csv("insan_kontrol/etiket/sonuc.csv", encoding="utf-8-sig")
TRK = {"karisik": "mixed", "emin_degil": "not_sure", "belirsiz": "ambiguous", "bos": "-"}  # bos = yanıtsız
assert len(A) == len(S) == 150 and (A.image_id.to_numpy() == S["no"].to_numpy()).all()
assert (A.label_original.to_numpy() == S.label.to_numpy()).all()
for a_, s_ in [("expert1", "uzman1"), ("expert2", "uzman2"), ("expert3", "uzman3"), ("consensus", "uzlasi")]:
    assert (A[a_].fillna("-").to_numpy() == S[s_].map(lambda v: TRK.get(v, v)).fillna("-").to_numpy()).all(), a_
HED = SIN + ["ambiguous"]
ct = pd.crosstab(A.label_original, A.consensus).reindex(index=SIN, columns=HED, fill_value=0)
ESKI = [[3, 0, 1, 0, 6], [0, 9, 0, 27, 10], [0, 0, 3, 1, 3], [0, 8, 0, 52, 27]]  # eski fig06 (b) ile aynı olmalı
assert ct.to_numpy().tolist() == ESKI, ct
d_ = ct.to_numpy()
print("kept", int(np.trace(d_[:, :4])), "changed", int(d_[:, :4].sum() - np.trace(d_[:, :4])), "ambiguous", int(d_[:, 4].sum()))

# ---------- E3 OOF olasılıkları (31_makale_ek_sekiller.py oof() ile aynı) ----------
D = [pd.concat([pd.read_csv(f"runs_asama2/grouped{k}_{u}_s{s}_kat_v1_asama2_none/test_predictions.csv") for k in range(5)])
     .set_index("path").sort_index() for u in E3 for s in (42, 43, 44)]
assert all((d.index == D[0].index).all() for d in D)
P = pd.DataFrame(np.mean([d[PC].to_numpy() for d in D], 0), columns=PC,
                 index=[Path(p).relative_to(VERI).as_posix() for p in D[0].index])
assert abs((P.to_numpy().argmax(1) == D[0].label.map(SIN.index).to_numpy()).mean() - 0.9103) < 5e-5  # E3 %91,0
O = pd.read_csv(DEPO / "oof_predictions.csv")
O = O[(O.protocol == "P3_scene_grouped") & (O.model + "_" + O.img_size.astype(str)).isin(
    ["convnext_tiny_224", "swin_tiny_patch4_window7_224_224", "vit_base_patch14_dinov2_392"])]
assert len(O) == 9 * 2095
Pz = O.groupby("rel_path")[PC].mean().loc[P.index]
assert np.abs(Pz.to_numpy() - P.to_numpy()).max() < 1e-4  # Zenodo dosyası 5 basamaklı yuvarlanmış
assert set(A.rel_path) <= set(P.index)

# ---------- örnekler: (kimlik, kırpma kutusu [sol, üst, sağ, alt] özgün piksel) ----------
ORN = [("E058", (255, 150, 545, 440), "kept"),      # SO etiketli, üç uzman da SO (korundu), E3 H; temiz/tozlu sınır
       ("E056", (340, 105, 800, 565), "changed"),     # SO etiketli, üç uzman da H (tek oybirlikli SO -> H); çatı FV
       ("E002", (424, 141, 2074, 1791), "changed"),   # H etiketli, üç uzman da SO; tozlu iki modül
       ("E037", (1000, 0, 1921, 921), "ambiguous")]   # SO etiketli; emin değil / SO / H -> belirsiz; araçlar kırpma dışı
Ai = A.set_index("image_id")
SATIR = []
for i, kut, durum in ORN:
    r = Ai.loc[i]
    assert r.candidate and r.status == durum, (i, r.candidate, r.status)  # dördü de modelin işaretlediği adaylar
    p = P.loc[r.rel_path]
    j = int(np.argmax(p.to_numpy()))
    assert j == int(np.argmax(Pz.loc[r.rel_path].to_numpy())) and f"{p.iloc[j]:.2f}" == f"{Pz.loc[r.rel_path].iloc[j]:.2f}"
    ex =[YAN.get(r[f"expert{u}"], "–") if isinstance(r[f"expert{u}"], str) else "–" for u in (1, 2, 3)]
    SATIR.append(dict(id=i, yol=r.rel_path, kutu=kut, orij=YAN[r.label_original], uzman=", ".join(ex),
                      uzlasi=YAN[r.consensus], e3=f"{K[j]} ({p.iloc[j]:.2f})"))
    print(i, r.rel_path, "| orij", YAN[r.label_original], "| uzmanlar", ex, "| uzlaşı", YAN[r.consensus],
          "| E3", K[j], f"{p.iloc[j]:.4f}", "| p(etiket)", f"{p['p_' + r.label_original]:.4f}")


def goruntu(yol, kutu, boy=720):
    im = ImageOps.exif_transpose(Image.open(VERI / yol)).convert("RGB")
    assert 0 <= kutu[0] < kutu[2] <= im.size[0] and 0 <= kutu[1] < kutu[3] <= im.size[1], (yol, im.size, kutu)
    im = im.crop(kutu)
    return im.resize((boy, round(boy * im.size[1] / im.size[0])), Image.LANCZOS)


# ---------- ana şekil: cm cinsinden yerleşim ----------
# bbox_inches="tight" her kenara 0,1 inç (0,254 cm) boşluk ekler: içerik 15,45 cm'de tutulur, PNG <= 16 cm olur
# (16 cm'de basıldığında küçültme yok, en küçük yazı 7 punto kalır).
W, H = 15.45, 4.2
fig = plt.figure(figsize=(W * CM, H * CM))


def eksen(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


hc, bos_ = 0.54, 0.24          # matris hücresi ve "belirsiz" sütunu önündeki boşluk (cm)
ust = H - 0.42                 # görüntü ve matris üst kenarı
mx = 0.85                      # matris sol kenarı (solunda y etiketi ve sınıf adları)
a = eksen(mx, ust - 4 * hc, 5 * hc + bos_, 4 * hc)
xe = np.array([0, 1, 2, 3, 4, 4 + bos_ / hc, 5 + bos_ / hc])     # 5. sütun (belirsiz) ayrık
Z = np.ma.masked_invalid(np.insert(ct.to_numpy().astype(float), 4, np.nan, axis=1))
a.pcolormesh(xe, np.arange(5), Z, cmap="Blues", vmin=0, vmax=ct.to_numpy().max(), edgecolors="white", linewidth=0.8)
xm = [0.5, 1.5, 2.5, 3.5, 4.5 + bos_ / hc]
for i in range(4):
    for j in range(5):
        v = ct.iat[i, j]
        a.text(xm[j], i + 0.5, v, ha="center", va="center", fontsize=8,
               color="white" if v >= 0.55 * ct.to_numpy().max() else "black")
a.set_xlim(0, xe[-1]); a.set_ylim(4, 0)
a.set_xticks(xm, K + ["ambig."], fontsize=7.5); a.set_yticks(np.arange(4) + 0.5, K, fontsize=7.5)
a.tick_params(length=0, pad=2)
for sp in a.spines.values():
    sp.set_visible(False)
a.set_xlabel("Expert consensus", fontsize=7.5, labelpad=3); a.set_ylabel("Original label", fontsize=7.5, labelpad=3)
fig.text(0.05 / W, (ust + 0.12) / H, "(a)", fontsize=8.5, ha="left", va="bottom")

gw, ara = 2.385, 0.12          # görüntü kenarı ve aralık (cm)
x0 = W - 4 * gw - 3 * ara      # ilk görüntünün sol kenarı (5,55 cm)
BAS = [("Original", "orij"), ("Experts 1–3", "uzman"), ("Consensus", "uzlasi"), ("E3", "e3")]
sy = [ust - gw - 0.23 - 0.31 * k for k in range(4)]
for (bas, _), y in zip(BAS, sy):
    fig.text((x0 - 0.1) / W, y / H, bas, ha="right", va="center", fontsize=7, color="#555555")
fig.text((x0 - 1.55) / W, (ust + 0.12) / H, "(b)", fontsize=8.5, ha="left", va="bottom")
for k, s in enumerate(SATIR):
    xa = x0 + k * (gw + ara)
    g = eksen(xa, ust - gw, gw, gw)
    g.imshow(goruntu(s["yol"], s["kutu"]))
    g.set_xticks([]); g.set_yticks([])
    for sp in g.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.5); sp.set_color(GRI)
    for (_, alan), y in zip(BAS, sy):
        fig.text((xa + gw / 2) / W, y / H, s[alan], ha="center", va="center", fontsize=7,
                 fontweight="bold" if alan == "uzlasi" else "normal")
kaydet("fig_expert_audit_examples.png")

# ---------- Ek Şekil S4: uzman başına yanıt dağılımı (eski fig06 a) ----------
SEC = SIN + ["mixed", "not_sure"]
SL = K + ["mixed", "not sure"]
renk = [C4, "#7fb069", "#8e6c8a", C2, "#bdbdbd", "#5d5d5d"]  # eski fig06 renkleri
fig, ax = plt.subplots(figsize=(9 * CM, 5.2 * CM))
alt = np.zeros(3)
EX = ["Expert 1", "Expert 2", "Expert 3"]
for s_, l_, c_ in zip(SEC, SL, renk):
    v_ = np.array([(A[f"expert{u}"] == s_).sum() for u in (1, 2, 3)])
    ax.bar(EX, v_, 0.62, bottom=alt, label=l_, color=c_); alt += v_
bos = np.array([A[f"expert{u}"].isna().sum() for u in (1, 2, 3)])
ax.bar(EX, bos, 0.62, bottom=alt, label="unanswered", color="white", edgecolor=GRI, hatch="//////", lw=0.5)
assert (alt + bos == 150).all() and bos.tolist() == [2, 0, 0]
ax.set_ylabel("Images"); ax.set_ylim(0, 150); ax.set_yticks([0, 50, 100, 150])
h_, l_ = ax.get_legend_handles_labels()
ax.legend(h_[::-1], l_[::-1], frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.2,
          handleheight=1.0)
print("S4:", {u: A[f"expert{u}"].map(lambda v: YAN.get(v, v)).value_counts(dropna=False).to_dict() for u in (1, 2, 3)})
plt.tight_layout(); kaydet("figS4_expert_labels.png")
