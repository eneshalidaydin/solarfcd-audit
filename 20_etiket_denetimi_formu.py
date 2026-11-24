"""Adım 20 - Model destekli etiket denetimi, 3 bağımsız uzman (kullanıcı kararı 21 Kasım; öneri A).
Aday seçimi (önceden sabit): 3 model (ConvNeXt, Swin, DINOv2) × 3 tohum havuzlanmış OOF olasılık ortalamasında verilen
etikete olasılık < 0,2 olan tüm görüntüler. Kontrol: kalan görüntülerden etiket katmanlı rastgele 38 görüntü (tohum
20261121); uzmanlar hangisinin aday olduğunu bilmez (doğrulama yanlılığını ve uzmanların taban uyumunu ölçmek için).
Kör: mevcut etiket ve model tahmini gösterilmez; her uzmanın sırası farklı karıştırılır.
Karar kuralı (sonuçlar görülmeden sabitlendi): 3 uzmandan en az 2'si aynı sınıfı seçerse etiket o sınıf olur; aksi hâlde
(çoğunluk yok, ya da çoğunluk "karisik"/"emin_degil") görüntü "belirsiz" sayılır: temiz analizde çıkarılır, orijinal
analizde kalır. Uyum: Fleiss kappa. Çıktı: insan_kontrol/etiket/ (uzman1-3.html, adaylar.csv, img/). Ham veri salt okunur."""
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

SIN = ["Electrical_Fault", "Healthy", "Structural_Fault", "Surface_Obstruction"]
PC = [f"p_{c}" for c in SIN]
OUT = Path("insan_kontrol/etiket")
(OUT / "img").mkdir(parents=True, exist_ok=True)
D = []
for mdl in ["convnext_tiny_224", "swin_tiny_patch4_window7_224_224", "vit_base_patch14_dinov2_224"]:
    for s in (42, 43, 44):
        D.append(pd.concat([pd.read_csv(f"runs_asama2/grouped{k}_{mdl}_s{s}_kat_v1_asama2_none/test_predictions.csv")
                            for k in range(5)]).set_index("path").sort_index())
P = np.mean([d[PC].to_numpy() for d in D], 0)
T = D[0][["label"]].copy()
T["p_etiket"] = P[np.arange(len(T)), T["label"].map(SIN.index)]
T["model_tahmini"] = np.array(SIN)[P.argmax(1)]
T["aday"] = T["p_etiket"] < 0.2
kalan = T[~T.aday]
kontrol = pd.concat([g.sample(round(38 * len(g) / len(kalan)), random_state=20261121) for _, g in kalan.groupby("label")])
S = pd.concat([T[T.aday], kontrol]).reset_index()
S["no"] = [f"E{i:03d}" for i in range(len(S))]
S["img"] = ""
for i, r in S.iterrows():
    im = ImageOps.exif_transpose(Image.open(r.path)).convert("RGB")
    im.thumbnail((1000, 1000))
    im.save(OUT / "img" / f"{r.no}.jpg", quality=90)
    S.at[i, "img"] = f"img/{r.no}.jpg"
S.to_csv(OUT / "adaylar.csv", index=False, encoding="utf-8-sig")
SEC = [("Electrical_Fault", "Elektriksel arıza (EF)"), ("Structural_Fault", "Yapısal/fiziksel arıza (SF)"),
       ("Healthy", "Sağlıklı"), ("Surface_Obstruction", "Yüzey örtüsü (SO)"), ("karisik", "Birden fazla sınıf"),
       ("emin_degil", "Emin değilim / görüntü yetersiz")]
ACIK = ("<b>EF</b>: elektriksel kusur (sıcak nokta, yanık/kahverengileşme, diyot/bağlantı kutusu, hücre renk bozulması). "
        "<b>SF</b>: fiziksel hasar (kırık/çatlak cam, kırık çerçeve, delaminasyon, darbe). <b>SO</b>: yüzeyi örten etken "
        "(toz, kir, kar, kuş pisliği, yaprak, gölge). <b>Sağlıklı</b>: belirgin kusur ya da örtü yok.")
for u in (1, 2, 3):
    sira = S.sample(frac=1, random_state=u)
    kart = "".join(
        f"<div class=kart data-id='{r.no}'><div class=bas><b>{r.no}</b></div><img src='{r.img}' loading=lazy>"
        f"<div class=sec>{''.join(f'<label><input type=radio name={r.no} value={v}> {t}</label>' for v, t in SEC)}"
        f"</div><input class=not placeholder='Not (isteğe bağlı)'></div>" for r in sira.itertuples())
    ids = json.dumps(list(sira.no))
    sayfa = f"""<!doctype html><html lang=tr><meta charset=utf-8><title>SolarFCD etiket denetimi - Uzman {u}</title>
<style>body{{font-family:Segoe UI,sans-serif;margin:0;background:#f4f4f4;color:#222}}
header{{position:sticky;top:0;background:#1f3b57;color:#fff;padding:10px 16px;display:flex;gap:16px;align-items:center;
flex-wrap:wrap;z-index:9}}header button{{font-size:15px;padding:6px 14px;cursor:pointer}}
main{{max-width:1050px;margin:0 auto;padding:16px}}
.kart{{background:#fff;border:2px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}}.kart.tamam{{border-color:#3a9d5d}}
.kart img{{max-width:100%;max-height:640px;display:block}}
.sec label{{display:inline-block;margin:6px 8px 0 0;padding:8px 12px;border:1px solid #bbb;border-radius:6px;cursor:pointer;
user-select:none}}.sec label:has(input:checked){{background:#3a9d5d;color:#fff;border-color:#3a9d5d}}
.not{{margin-top:8px;width:100%;box-sizing:border-box;padding:6px}}</style>
<header><b>SolarFCD etiket denetimi — Uzman {u}</b><span id=sayac></span><button id=sonraki>İlk boş karta git</button>
<button id=kaydet>Kaydet (CSV indir)</button></header><main>
<p>Her görüntü için panelin <b>asıl sınıfını</b> fareyle seçin. Diğer uzmanlarla görüşmeden, bağımsız karar verin.
Seçimler bu tarayıcıda kendiliğinden saklanır. Bitince <b>Kaydet</b>'e basın: <code>etiket_uzman{u}.csv</code> İndirilenler'e
iner; bu dosyayı gönderin.</p><p>{ACIK}</p>{kart}</main>
<script>const IDS={ids};const ANAHTAR="solarfcd_etiket_uzman{u}";let durum={{}};
try{{durum=JSON.parse(localStorage.getItem(ANAHTAR)||"{{}}")}}catch(e){{}}
const kartlar=[...document.querySelectorAll(".kart")];
function sakla(){{try{{localStorage.setItem(ANAHTAR,JSON.stringify(durum))}}catch(e){{}}}}
function guncelle(){{let n=0;kartlar.forEach(k=>{{const ok=(durum[k.dataset.id]||{{}}).k;k.classList.toggle("tamam",!!ok);
if(ok)n++}});document.getElementById("sayac").textContent=n+" / "+kartlar.length+" işaretlendi"}}
kartlar.forEach(k=>{{const id=k.dataset.id,d=durum[id]||{{}};k.querySelectorAll("input[type=radio]").forEach(r=>{{
if(r.value===d.k)r.checked=true;r.addEventListener("change",()=>{{durum[id]=Object.assign(durum[id]||{{}},{{k:r.value}});
sakla();guncelle()}})}});const t=k.querySelector(".not");t.value=d.n||"";
t.addEventListener("input",()=>{{durum[id]=Object.assign(durum[id]||{{}},{{n:t.value}});sakla()}})}});
document.getElementById("sonraki").onclick=()=>{{const k=kartlar.find(k=>!(durum[k.dataset.id]||{{}}).k);
if(k)k.scrollIntoView({{behavior:"smooth",block:"start"}});else alert("Hepsi işaretlendi.")}};
document.getElementById("kaydet").onclick=()=>{{const q=s=>'"'+String(s||"").replace(/"/g,'""')+'"';
const satir=["uzman,no,karar,not"].concat(IDS.map(i=>{{const d=durum[i]||{{}};return ["{u}",i,d.k,d.n].map(q).join(",")}}));
const b=new Blob(["\\ufeff"+satir.join("\\r\\n")],{{type:"text/csv"}});const a=document.createElement("a");
a.href=URL.createObjectURL(b);a.download="etiket_uzman{u}.csv";a.click()}};guncelle();</script></html>"""
    (OUT / f"uzman{u}.html").write_text(sayfa, encoding="utf-8")
print(len(S), "görüntü:", int(S.aday.sum()), "aday +", int((~S.aday).sum()), "kontrol")
print(S.groupby(["aday", "label"]).size().to_string())
