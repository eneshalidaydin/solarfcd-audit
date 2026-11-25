"""Adım 12 - İnsan kontrol seti (kullanıcı kararı 19 Kasım). Çıktı: insan_kontrol/
  A) r4 kör denetiminin hakemli sızıntı/belirsiz çiftleri (14) + r4 eleştirmeninin bulduğu yeni çift (589~872)
  B) gorsel_etiketler.csv'den rastgele 100 karar (tohum 20261119): 50 pozitif/belirsiz, 50 ilgisiz (kesin)
  C) etiketi şüpheli EF/SF görüntüleri (r3 azınlık denetçisinin notları)
kontrol.html (görüntüler yan yana, yapay zekâ kararı ve gerekçe) + kontrol_formu.csv (insan_karari sütunu boş:
ayni_foto / ayni_sahne / ilgisiz / emin_degil) + efsf_formu.csv (uzman_etiketi sütunu boş). Ham veriye dokunmaz;
küçük resimler insan_kontrol/img/ altına yazılır."""
import html
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

OUT = Path("insan_kontrol")
(OUT / "img").mkdir(parents=True, exist_ok=True)
m = pd.read_csv("asama2/manifest_asama2.csv")
yol = m.set_index("rel_path")["path"]
d2r = m.groupby("dosya")["rel_path"].first()
n_img = 0


def kucuk(rel):
    global n_img
    n_img += 1
    ad = f"{n_img:04d}.jpg"
    im = ImageOps.exif_transpose(Image.open(yol[rel])).convert("RGB")
    im.thumbnail((520, 520))
    im.save(OUT / "img" / ad, quality=88)
    return f"img/{ad}"


satir = []
pf = pd.read_csv("dogrulama/asama2_r4/birlestir/pairs_final.csv")
for r in pf[pf["j3"] != "unrelated"].itertuples():
    satir.append(dict(bolum="A", rel_a=r.query_rel, rel_b=r.cand_rel, yz_karari=r.j3, gerekce=str(r.j3_reason)[:300]))
satir.append(dict(bolum="A", rel_a=d2r["Imgdirty_589_1.jpg"], rel_b=d2r["Imgclean_872_0.jpg"], yz_karari="same_scene",
                  gerekce="r4 eleştirmeni: aynı kamelya, beyaz sandalye, yuvarlak beton masa, 2+1 eğik dizi (önce/sonra)"))
g = pd.read_csv("asama2/gorsel_etiketler.csv")
g = g[g["rel_a"].isin(yol.index) & g["rel_b"].isin(yol.index)]
poz = g[g["sinif"].isin(["ayni_foto", "ayni_sahne", "belirsiz"])].sample(50, random_state=20261119)
neg = g[(g["sinif"] == "ilgisiz") & (g["kesinlik"] == "kesin")].sample(50, random_state=20261119)
for r in pd.concat([poz, neg]).sample(frac=1, random_state=1).itertuples():
    satir.append(dict(bolum="B", rel_a=r.rel_a, rel_b=r.rel_b, yz_karari=f"{r.sinif} ({r.kesinlik})",
                      gerekce=str(r.kaynaklar)[:200]))
K = pd.DataFrame(satir)
K.insert(0, "no", [f"K{i:03d}" for i in range(len(K))])
K["insan_karari"] = ""
K["not"] = ""
EF = [("Electrical (103).jpg", "mermi/dolu delikleri, radyal cam çatlakları: fiziksel hasar, SF'ye benziyor"),
      ("Electrical (73).jpg", "kırık cam ağı + üst baradaki darbe izleri: SF'ye benziyor"),
      ("Electrical (9).jpg", "görünür kusur yok, yalnız güneş parlaması: Healthy'ye benziyor"),
      ("Electrical (13).jpg", "lekeli gölgelenme/kir, belirgin elektrik kusuru yok: H/SO'ya benziyor"),
      ("Physical (83).jpg", "kırık cam yok; yanık benzeri beyazlaşma ve delaminasyon: EF'ye benziyor"),
      ("Electrical (129).jpg", "kasıtlı dış yangın testi, elektrik arızası değil"),
      ("Electrical (144).jpg", "çerçeve yanığı + kırık cam: karışık EF/SF"),
      ("Electrical (118).jpg", "turuncu ışımalar yansıma gibi; zayıf kanıt")]
E = pd.DataFrame([dict(dosya=f, rel=d2r[f], mevcut_etiket=m.set_index("rel_path").at[d2r[f], "label"],
                       kat_v1=int(m.set_index("rel_path").at[d2r[f], "kat_v1"]), yz_notu=n, uzman_etiketi="", not_="")
                  for f, n in EF])
K.to_csv(OUT / "kontrol_formu.csv", index=False, encoding="utf-8-sig")
E.to_csv(OUT / "efsf_formu.csv", index=False, encoding="utf-8-sig")
h = ["<!doctype html><meta charset='utf-8'><title>SolarFCD insan kontrol</title><style>body{font-family:sans-serif;"
     "margin:16px}td{vertical-align:top;padding:6px;border-bottom:1px solid #ccc}img{max-width:420px;max-height:420px}"
     "</style><h1>SolarFCD insan kontrolü</h1><p>Her satır için kontrol_formu.csv'de <b>insan_karari</b> sütununa "
     "ayni_foto / ayni_sahne / ilgisiz / emin_degil yazın. Bölüm A: r4 denetiminin sızıntı/belirsiz çiftleri; "
     "Bölüm B: rastgele 100 yapay zekâ kararı. En altta EF/SF etiket listesi (efsf_formu.csv).</p><table>"]
for r in K.itertuples():
    h.append(f"<tr><td><b>{r.no}</b> ({r.bolum})<br>YZ: {html.escape(r.yz_karari)}<br><small>"
             f"{html.escape(r.gerekce)}</small><br><small>{html.escape(r.rel_a)}<br>{html.escape(r.rel_b)}</small></td>"
             f"<td><img src='{kucuk(r.rel_a)}'></td><td><img src='{kucuk(r.rel_b)}'></td></tr>")
h.append("</table><h2>EF/SF: etiketi şüpheli görüntüler</h2><table>")
for r in E.itertuples():
    h.append(f"<tr><td><b>{html.escape(r.dosya)}</b><br>mevcut: {r.mevcut_etiket}<br><small>{html.escape(r.yz_notu)}"
             f"</small></td><td><img src='{kucuk(r.rel)}'></td></tr>")
h.append("</table>")
(OUT / "kontrol.html").write_text("\n".join(h), encoding="utf-8")
print(len(K), "çift;", len(E), "EF/SF görüntüsü;", n_img, "küçük resim")
