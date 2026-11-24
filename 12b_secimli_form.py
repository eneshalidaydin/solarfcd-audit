"""Adım 12b - İnsan kontrolü için tıklamalı form (kullanıcı isteği 20 Kasım: yazmak yerine fareyle seçim).
12'nin ürettiği kontrol_formu.csv, efsf_formu.csv ve img/ küçük resimlerini kullanır (12 yeniden çalıştırılmaz).
Küçük resim sırası 12'deki gibidir: her K satırı için a, b; sonra EF/SF görüntüleri.
Kör değerlendirme: yapay zekâ kararı, gerekçesi ve mevcut etiket sayfada gösterilmez.
Seçimler tarayıcıda otomatik saklanır; "Kaydet" düğmesi insan_kararlari.csv dosyasını indirir.
Çıktı: insan_kontrol/kontrol_secimli.html"""
import html
import json
from pathlib import Path

import pandas as pd

OUT = Path("insan_kontrol")
K = pd.read_csv(OUT / "kontrol_formu.csv", encoding="utf-8-sig")
E = pd.read_csv(OUT / "efsf_formu.csv", encoding="utf-8-sig")
SEC_K = [("ayni_foto", "Aynı fotoğraf"), ("ayni_sahne", "Aynı sahne/panel"), ("ilgisiz", "İlgisiz"),
         ("emin_degil", "Emin değilim")]
SEC_E = [("Electrical_Fault", "Elektriksel arıza (EF)"), ("Structural_Fault", "Yapısal/fiziksel arıza (SF)"),
         ("Healthy", "Sağlıklı"), ("Surface_Obstruction", "Yüzey örtüsü (kir/kar/gölge)"),
         ("emin_degil", "Emin değilim")]
ogeler = []
for i, r in enumerate(K.itertuples()):
    ogeler.append(dict(id=r.no, tur="cift", img=[f"img/{2 * i + 1:04d}.jpg", f"img/{2 * i + 2:04d}.jpg"]))
for j, r in enumerate(E.itertuples()):
    ogeler.append(dict(id=r.dosya, tur="efsf", img=[f"img/{2 * len(K) + j + 1:04d}.jpg"]))
for o in ogeler:
    for p in o["img"]:
        assert (OUT / p).exists(), p


def radyo(o, secenekler):
    ad = html.escape(o["id"], quote=True)
    return "".join(f"<label><input type=radio name='{ad}' value='{v}'> {t}</label>" for v, t in secenekler)


kartlar = []
for o in ogeler:
    sec = SEC_K if o["tur"] == "cift" else SEC_E
    resim = "".join(f"<img src='{p}' loading=lazy>" for p in o["img"])
    soru = "Bu iki görüntü:" if o["tur"] == "cift" else "Bu görüntünün doğru sınıfı:"
    kartlar.append(f"<div class=kart data-id='{html.escape(o['id'], quote=True)}'><div class=bas>"
                   f"<b>{html.escape(o['id'])}</b></div><div class=resim>{resim}</div><div class=soru>{soru}</div>"
                   f"<div class=sec>{radyo(o, sec)}</div><input class=not placeholder='Not (isteğe bağlı)'></div>")
n_k = len(K)
sayfa = f"""<!doctype html><html lang=tr><meta charset=utf-8><title>SolarFCD insan kontrolü</title>
<style>
body{{font-family:Segoe UI,sans-serif;margin:0;background:#f4f4f4;color:#222}}
header{{position:sticky;top:0;background:#1f3b57;color:#fff;padding:10px 16px;display:flex;gap:16px;
align-items:center;flex-wrap:wrap;z-index:9}}
header button{{font-size:15px;padding:6px 14px;cursor:pointer}}
main{{max-width:1000px;margin:0 auto;padding:16px}}
h2{{margin:28px 0 8px}}
.kart{{background:#fff;border:2px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}}
.kart.tamam{{border-color:#3a9d5d}}
.resim{{display:flex;gap:10px;flex-wrap:wrap}}
.resim img{{max-width:min(470px,100%);max-height:420px;cursor:zoom-in}}
.soru{{margin-top:8px;font-weight:600}}
.sec label{{display:inline-block;margin:6px 8px 0 0;padding:8px 12px;border:1px solid #bbb;border-radius:6px;
cursor:pointer;user-select:none}}
.sec label:has(input:checked){{background:#3a9d5d;color:#fff;border-color:#3a9d5d}}
.not{{margin-top:8px;width:100%;box-sizing:border-box;padding:6px}}
#buyuk{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:20;align-items:center;
justify-content:center}}
#buyuk img{{max-width:95vw;max-height:95vh}}
</style>
<header><b>SolarFCD insan kontrolü</b><span id=sayac></span>
<button id=sonraki>İlk boş karta git</button><button id=kaydet>Kaydet (CSV indir)</button></header>
<main>
<p>Her kart için fareyle bir seçenek işaretleyin. Seçimler bu tarayıcıda kendiliğinden saklanır; sayfayı kapatıp
sonra devam edebilirsiniz. Bitirince (ya da ara verirken) <b>Kaydet</b>'e basın: <code>insan_kararlari.csv</code>
İndirilenler klasörüne iner. Görüntüye tıklarsanız büyür. Yapay zekânın kararı bilerek gösterilmiyor.</p>
<p><b>Aynı fotoğraf</b>: aynı çekim (kırpma, döndürme, filigran, renk farkı olabilir). <b>Aynı sahne/panel</b>: farklı
çekim ama aynı fiziksel panel/yer (ör. önce-sonra, başka açı). <b>İlgisiz</b>: farklı yerler.</p>
<h2>Bölüm 1: Görüntü çiftleri ({n_k})</h2>
{''.join(kartlar[:n_k])}
<h2>Bölüm 2: EF/SF etiket kontrolü ({len(E)})</h2>
{''.join(kartlar[n_k:])}
</main><div id=buyuk><img></div>
<script>
const OGE={json.dumps([dict(id=o["id"], tur=o["tur"]) for o in ogeler], ensure_ascii=False)};
const ANAHTAR="solarfcd_insan_kontrol_v1";
let durum={{}};
try{{durum=JSON.parse(localStorage.getItem(ANAHTAR)||"{{}}")}}catch(e){{}}
const kartlar=[...document.querySelectorAll(".kart")];
function sakla(){{try{{localStorage.setItem(ANAHTAR,JSON.stringify(durum))}}catch(e){{}}}}
function guncelle(){{let n=0;kartlar.forEach(k=>{{const d=durum[k.dataset.id];const ok=d&&d.k;
k.classList.toggle("tamam",!!ok);if(ok)n++}});
document.getElementById("sayac").textContent=n+" / "+kartlar.length+" işaretlendi"}}
kartlar.forEach(k=>{{const id=k.dataset.id,d=durum[id]||{{}};
k.querySelectorAll("input[type=radio]").forEach(r=>{{if(r.value===d.k)r.checked=true;
r.addEventListener("change",()=>{{durum[id]=Object.assign(durum[id]||{{}},{{k:r.value}});sakla();guncelle()}})}});
const t=k.querySelector(".not");t.value=d.n||"";
t.addEventListener("input",()=>{{durum[id]=Object.assign(durum[id]||{{}},{{n:t.value}});sakla()}})}});
document.getElementById("sonraki").onclick=()=>{{const k=kartlar.find(k=>!(durum[k.dataset.id]||{{}}).k);
if(k)k.scrollIntoView({{behavior:"smooth",block:"start"}});else alert("Hepsi işaretlendi.")}};
document.getElementById("kaydet").onclick=()=>{{const q=s=>'"'+String(s||"").replace(/"/g,'""')+'"';
const satir=["tur,id,karar,not"].concat(OGE.map(o=>{{const d=durum[o.id]||{{}};
return [o.tur,o.id,d.k,d.n].map(q).join(",")}}));
const b=new Blob(["\\ufeff"+satir.join("\\r\\n")],{{type:"text/csv"}});const a=document.createElement("a");
a.href=URL.createObjectURL(b);a.download="insan_kararlari.csv";a.click()}};
const B=document.getElementById("buyuk");
document.querySelectorAll(".resim img").forEach(i=>i.onclick=()=>{{B.firstChild.src=i.src;B.style.display="flex"}});
B.onclick=()=>B.style.display="none";
guncelle();
</script></html>"""
(OUT / "kontrol_secimli.html").write_text(sayfa, encoding="utf-8")
print(len(ogeler), "kart:", n_k, "çift +", len(E), "EF/SF")
