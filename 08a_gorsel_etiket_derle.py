"""Adım 8a - Bağımsız denetimlerdeki görsel kararları tek tabloda derler: asama2/gorsel_etiketler.csv

Her satır, bir denetim ajanının montajda/özgün görüntülerde gözle incelediği bir çifttir. Bu kararlar
08_asama2_manifest.py'de (a) elle onaylanmış kenar, (b) zorunlu ayrıklık ("ilgisiz" çiftler arasında otomatik kenar
kurulmaz; "aynı sahne" çiftleri aynı-fotoğraf sayılmaz) ve (c) eşleştirici kalibrasyonu için kullanılır.
Not: kararları yapay zekâ ajanları verdi; makale için bir insanın örneklem üzerinden kontrol etmesi gerekir.

Kaynaklar
  r1 denetimi (dogrulama/asama2_*):
    asama2_leak/l15_visual_labels.csv, asama2_sift/s05_gorsel_etiketler.csv (+ s03_ornek.csv), asama2_sift/s13_X_etiketli.csv,
    asama2_sift/s15_ayni_foto_farkli_etiket.csv, asama2_conflict/tespit_edilmemis_celiskiler.csv, asama2_critic/k18_gorsel_kararlar.csv
  r2 denetimi (dogrulama/asama2_r2/):
    gozden1/labels.csv + gozden2/labels.csv (iki kör gözden geçiren; birlestir/ hakem kararıyla birleştirilir),
    kod/cikti/k32_gorsel_kararlar.csv, celiski/c11_t1_*, c11_t2_*, c18_*, c21_t34_*, c28_*, azinlik/b/yargilar.csv,
    azinlik/b/sizinti_ciftleri.csv
Sınıflar: ayni_foto, ayni_sahne, belirsiz, ilgisiz; kesinlik: kesin / olasi ("?" ekli, "olası", tek gözden geçiren).
    r2 eleştirmeni (elestirmen/e10_kararlar.csv): bizzat baktığı 73 çiftte diğer tüm kaynakları geçersiz kılar
  r3 hazırlığı (benim görsel kararlarım; montajla): aynı ürün ilanı serisi, P12-P ~ P68/P75, iPhone 11 oturumları
  r3 denetimi (dogrulama/asama2_r3/): kod k15/k27/k28, yakın y03/y04/y06/y09/y10, azınlık yargıları, örneklem
    (birlestir/pairs_final.csv), hakemin filigran nüfus taraması, eleştirmen e08
  r4 görsel taramaları (dogrulama/asama2_r4_hazirlik/): tur 1 (girdi), tur 2 (girdi_tur2), r4.1 Kaggle (girdi_r41b);
    tarayıcı + kör doğrulayıcı (ikisi de pozitif → kesin; biri ya da "belirsiz" → belirsiz; tek tarayıcı "ilgisiz" →
    olası, yani sahne kenarını engellemez) + gözden geçirenlerin liste dışı notları
  r4 denetimi (dogrulama/asama2_r4/): örneklem (birlestir/pairs_final.csv) ve eleştirmenin kararları
Birleştirme (aynı çift için birden çok karar): eleştirmen kararı varsa o; yoksa sınıf önceliği ayni_foto > ayni_sahne >
belirsiz > ilgisiz (sızıntı açısından ihtiyatlı). Kesinlik: seçilen sınıfı veren kararlardan biri kesinse kesin.
Kullanım (08_asama2_manifest.py): ayni_foto/ayni_sahne (kesin+olası) elle kenar; kesin "ilgisiz" ve tüm "ayni_sahne"
çiftleri aynı-fotoğraf kenarı kuramaz (sahne kenarı engellenmez); kalibrasyona yalnız kesinler girer.

Kullanım:  python 08a_gorsel_etiket_derle.py
"""
import argparse
from pathlib import Path

import pandas as pd

from ortak import utf8_konsol

DG = Path("dogrulama")
ONCELIK = {"ayni_foto": 3, "ayni_sahne": 2, "belirsiz": 1, "ilgisiz": 0}
ESLEME = {
    "same_photo": ("ayni_foto", "kesin"), "same_scene": ("ayni_sahne", "kesin"),
    "same_scene_probable": ("ayni_sahne", "olasi"), "unrelated": ("ilgisiz", "kesin"),
    "unrelated_watermark": ("ilgisiz", "kesin"), "unrelated_template": ("ilgisiz", "kesin"),
    "unrelated(agreed)": ("ilgisiz", "kesin"),
    "not_confirmed": ("belirsiz", "kesin"), "SAME_PHOTO": ("ayni_foto", "kesin"), "SAME_SCENE": ("ayni_sahne", "kesin"),
    "UNRELATED": ("ilgisiz", "kesin"), "WATERMARK": ("ilgisiz", "kesin"), "AMBIGUOUS": ("belirsiz", "kesin"),
    "same_photo_label_conflict": ("ayni_foto", "kesin"), "same_scene_exif": ("ayni_sahne", "kesin"),
    "same_scene_before_after": ("ayni_sahne", "kesin"), "same_scene_same_panel": ("ayni_sahne", "kesin"),
    "same_session_exif": ("ayni_sahne", "kesin"), "uncertain": ("belirsiz", "kesin"), "unsure": ("belirsiz", "kesin"),
    "same_photo_or_scene": ("ayni_sahne", "kesin"), "same_scene?": ("ayni_sahne", "olasi"),
    "unrelated?": ("ilgisiz", "olasi"), "same_photo?": ("ayni_foto", "olasi"), "probable": ("ayni_sahne", "olasi"),
    # Türkçe kararlar
    "ayni_sahne": ("ayni_sahne", "kesin"), "ayni_foto": ("ayni_foto", "kesin"), "ilgisiz": ("ilgisiz", "kesin"),
    "belirsiz": ("belirsiz", "kesin"), "olasi": ("ayni_sahne", "olasi"),
}


def main():
    utf8_konsol()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cikti", default="asama2")
    ap.add_argument("--r2-manifest", default="asama2/r2/manifest_asama2.csv", help="k32 kat numaraları için (arşiv)")
    args = ap.parse_args()
    m = pd.read_csv("denetim_final/manifest.csv")
    m = m[m["ok"]].reset_index(drop=True)
    gecerli = set(m["rel_path"])
    d2r = m.groupby("dosya")["rel_path"].apply(list).to_dict()
    satir = []

    def ekle(a, b, ham, kaynak, not_="", kesinlik=None, ust=False):
        if a == b or pd.isna(a) or pd.isna(b):
            return
        if ham not in ESLEME:
            raise SystemExit(f"eşlenmeyen karar: {ham!r} ({kaynak})")
        sinif, kes = ESLEME[ham]
        satir.append(dict(rel_a=min(a, b), rel_b=max(a, b), sinif=sinif, kesinlik=kesinlik or kes, ham_karar=ham,
                          kaynak=kaynak, not_=str(not_)[:300], ust=ust))

    def tek(dosya):
        a = d2r.get(dosya, [])
        if len(a) != 1:
            raise SystemExit(f"dosya adı tekil değil ya da yok: {dosya} → {a}")
        return a[0]

    # ---------------- r1 denetimi
    for r in pd.read_csv(DG / "asama2_leak/l15_visual_labels.csv").to_dict("records"):
        ekle(r["rel_i"], r["rel_j"], r["visual_class"], f"r1_leak_l15:{r['tag']}")
    o = pd.read_csv(DG / "asama2_sift/s03_ornek.csv").set_index("kimlik")
    for r in pd.read_csv(DG / "asama2_sift/s05_gorsel_etiketler.csv").to_dict("records"):
        ekle(o.loc[r["kimlik"], "rel_i"], o.loc[r["kimlik"], "rel_j"], r["gorsel_sinif"], f"r1_sift_s05:{r['kimlik']}",
             r["not"])
    for k, r in enumerate(pd.read_csv(DG / "asama2_sift/s13_X_etiketli.csv").to_dict("records")):
        ekle(r["rel_i"], r["rel_j"], r["vis"], f"r1_sift_s13:X_{k:02d}")
    for r in pd.read_csv(DG / "asama2_sift/s15_ayni_foto_farkli_etiket.csv").to_dict("records"):
        ekle(r["rel_i"], r["rel_j"], "same_photo", "r1_sift_s15")
    t = pd.read_csv(DG / "asama2_conflict/tespit_edilmemis_celiskiler.csv")
    for g, d in t.groupby("ayni_foto"):
        rels = [tek(f) for f in d["dosya"]]
        for x in rels[1:]:
            ekle(rels[0], x, "same_photo", f"r1_conflict_tespit:{g}")
    for r in pd.read_csv(DG / "asama2_critic/k18_gorsel_kararlar.csv").to_dict("records"):
        ekle(r["rel_a"], r["rel_b"], r["karar"], "r1_critic_k18", r["not"])

    # ---------------- r2 denetimi
    R2 = DG / "asama2_r2"
    # iki kör gözden geçiren + hakem (birlestir/pairs_final.csv: j3 = son karar; 'agreed' = iki gözden geçiren aynı)
    pf = pd.read_csv(R2 / "birlestir/pairs_final.csv")
    for r in pf.to_dict("records"):
        kes = "olasi" if r["c3"] == "low" else "kesin"
        ekle(r["query_rel"], r["cand_rel"], r["j3"], f"r2_gozden+hakem:{r['cand_id']}", kesinlik=kes)
    for r in pd.read_csv(R2 / "birlestir/adjudicated_extra.csv").to_dict("records"):
        ekle(r["query_rel"], r["cand_rel"], r["judgement"], f"r2_hakem_ek:{r['query_id']}", r["reason"],
             kesinlik="olasi" if r["confidence"] == "low" else "kesin")
    for r in pd.read_csv(R2 / "birlestir/incidental_pairs.csv").to_dict("records"):
        ekle(r["rel_a"], r["rel_b"], r["judgement"], "r2_hakem_rastlanti", r["reason"],
             kesinlik="olasi" if r["confidence"] == "low" else "kesin")
    # r3 hazırlığında benim görsel kararım: aynı ürün ilanı serisi (aynı 3 sütunlu panel modeli; beyaz zemin / siyah
    # stand; birleştirici hakemin bulgusu, montajla doğrulandı; 426/428/429/432/434/508 farklı model → dahil değil)
    seri = [f"Imgclean_{n}_0.jpg" for n in (414, 415, 416, 417, 420, 421, 422, 423, 424, 431, 501, 503, 505, 510, 512)]         + ["Clean (30)-cc.jpg", "Clean (57).jpg", "Clean (59).jpg"]
    seri_rel = [tek(f) for f in seri]
    for x in seri_rel[1:]:
        ekle(seri_rel[0], x, "same_scene", "r3_benim_urun_serisi", "aynı ürün ilanı serisi (elle grup)", kesinlik="olasi")
    k32 = pd.read_csv(R2 / "kod/cikti/k32_gorsel_kararlar.csv")
    # k32'deki kat numaraları r2 manifestine aittir (arşiv: asama2/r2/)
    sk = pd.read_csv(args.r2_manifest, usecols=["rel_path", "dosya", "label", "kat_v1"])
    for r in k32.to_dict("records"):
        ab = []
        for f, kat, lab in ((r["a"], r["kat_a"], r["lab_a"]), (r["b"], r["kat_b"], r["lab_b"])):
            c = sk[(sk["dosya"] == f)]
            if len(c) > 1:
                c = c[(c["label"] == lab)]
            if len(c) > 1:
                c = c[c["kat_v1"] == kat]
            if len(c) != 1:
                raise SystemExit(f"k32 dosya eşlenemedi: {f}")
            ab.append(c["rel_path"].iloc[0])
        ekle(ab[0], ab[1], r["karar"], f"r2_kod_k32:{r['no']}", r["gerekce"])
    for r in pd.read_csv(R2 / "celiski/c11_t1_karar_tablosu.csv").to_dict("records"):
        ekle(r["a"], r["b"], r["karar"], "r2_celiski_c11t1", r["gerekce"])
    for r in pd.read_csv(R2 / "celiski/c11_t2_karar_tablosu.csv").to_dict("records"):
        ekle(r["rel_i"], r["rel_j"], r["karar"], "r2_celiski_c11t2", r["gerekce"])
    for r in pd.read_csv(R2 / "celiski/c18_t2_ek_kararlar.csv").to_dict("records"):
        ekle(r["rel_a"], r["rel_b"], r["karar"], "r2_celiski_c18", r["gerekce"])
    for r in pd.read_csv(R2 / "celiski/c21_t34_karar_tablosu.csv").to_dict("records"):
        ekle(r["rel_i"], r["rel_j"], r["karar"], "r2_celiski_c21", r.get("gerekce", ""))
    for r in pd.read_csv(R2 / "celiski/c28_sizinti_listesi.csv").to_dict("records"):
        ekle(r["rel_a"], r["rel_b"], r["karar"], "r2_celiski_c28", r["gerekce"])
    for r in pd.read_csv(R2 / "azinlik/b/yargilar.csv").to_dict("records"):
        ekle(r["relA"], r["relB"], r["karar"], "r2_azinlik_yargi", r["neden"])
    for r in pd.read_csv(R2 / "azinlik/b/sizinti_ciftleri.csv").to_dict("records"):
        kes = "kesin" if r["guven"] == "kesin" else "olasi"
        ekle(r["A"], r["B"], r["karar"], "r2_azinlik_sizinti", r["neden"], kesinlik=kes)

    # r2 eleştirmeni: 73 çifte bizzat bakarak karar verdi; bu çiftlerde diğer kaynakları geçersiz kılar (ör. kod
    # denetçisinin Imgclean_984 ve 751/811 "aynı sahne" iddialarını kanıtlanmamış buldu)
    for r in pd.read_csv(R2 / "elestirmen/e10_kararlar.csv").to_dict("records"):
        ekle(r["rel_a"], r["rel_b"], r["verdict"], f"r2_elestirmen:{r['id']}", r["reason"],
             kesinlik="olasi" if r["confidence"] == "low" else None, ust=True)
    # r3 hazırlığında benim kararım (azınlık denetçisi + eleştirmen önerisi): P12-P, P68/P75 ile aynı kuru tarla sahası
    # olabilir; SF'de ihtiyatlı birleştirme (yalnız kat dengesini etkiler)
    for x in ("Physical (68).jpg", "Physical (75).jpg"):
        ekle(tek("Physical (12)-P.jpg"), tek(x), "same_scene", "r3_benim_azinlik", "olası aynı saha (ihtiyatlı)",
             kesinlik="olasi")
    # r3 hazırlığında benim kararım: aynı iPhone 11 ile ~20 saat arayla iki kuş pisliği oturumu (aynı modül tipi ve beton
    # parapet; katlar arası 84 EXIF çiftinin tamamı bu iki oturumdan) → olası aynı çatı (ihtiyatlı birleştirme)
    for x in ("Bird (51)-bb.jpg", "Bird (104).jpg"):
        ekle(tek("Bird (86).jpg"), tek(x), "same_scene", "r3_benim_exif_oturumu", "aynı iPhone 11, ~20 sa arayla",
             kesinlik="olasi")
    # ---------------- r3 denetimi (dogrulama/asama2_r3/): r3'ün katlar arası kaçaklarını arayan ajanların kararları
    R3 = DG / "asama2_r3"
    TR = {"same_photo": "same_photo", "same_scene": "same_scene", "unrelated": "unrelated", "unsure": "unsure",
          "uncertain": "uncertain"}
    for dosya_ in ("kod/k15_gorsel_kararlar.csv", "kod/k27_kararlar.csv"):
        for r in pd.read_csv(R3 / dosya_).to_dict("records"):
            ekle(r["rel_a"], r["rel_b"], r["karar"], f"r3_denetim_kod:{Path(dosya_).stem}", r["gerekce"],
                 kesinlik=r["kesinlik"])
    for r in pd.read_csv(R3 / "kod/k28_kararlar.csv").to_dict("records"):  # 'kompakt' = küçük montajda bakıldı
        ekle(r["a"], r["b"], r["karar"], "r3_denetim_kod:k28", r["gerekce"],
             kesinlik="kesin" if r["kesinlik"] == "kesin" else "olasi")
    Y = R3 / "yakin"
    for kod_d, yar_d in (("y01_kor_kodlar.csv", "y03_yargilar.csv"), ("y04_B_dusuk_derece.csv", "y04_B_dusuk_yargilar.csv"),
                         ("y09_ek_kor_kodlar.csv", "y09_ek_yargilar.csv")):
        kk = pd.read_csv(Y / kod_d).set_index("kod")
        for r in pd.read_csv(Y / yar_d).to_dict("records"):
            ekle(kk.loc[r["kod"], "a"], kk.loc[r["kod"], "b"], TR[r["karar"]], f"r3_denetim_yakin:{r['kod']}", r["gerekce"],
                 kesinlik="olasi" if r["guven"] == "dusuk" else None)
    Y6 = {"yanlis": ("unrelated", None), "dogru": ("same_scene", None), "belirsiz": ("unsure", None),
          "olasi_dogru": ("same_scene", "olasi")}
    for r in pd.read_csv(Y / "y06_grup_kenar_yargilari.csv").to_dict("records"):
        ham, kes = Y6[r["karar"]]
        ekle(r["rel_i"], r["rel_j"], ham, f"r3_denetim_yakin:{r['kod']}", r["gerekce"], kesinlik=kes)
    for r in pd.read_csv(Y / "y10_sizinti_ciftleri.csv").to_dict("records"):
        ekle(r["a"], r["b"], TR[r["karar"]], f"r3_denetim_yakin_sizinti:{r['kaynak']}", r["gerekce"])
    etk = m.set_index("rel_path")["label"]
    for r in pd.read_csv(R3 / "azinlik/yargilar.csv").to_dict("records"):
        ab = []
        for f, lab in ((r["q_dosya"], r["label_q"]), (r["c_dosya"], r["label_c"])):
            c = [x for x in d2r.get(f, []) if etk[x] == lab]
            if len(c) != 1:
                raise SystemExit(f"azınlık dosyası eşlenemedi: {f} ({lab}) → {c}")
            ab.append(c[0])
        ekle(ab[0], ab[1], TR[r["karar"]], f"r3_denetim_azinlik:{r['kaynak']}", r["neden"])
    # kör örneklem denetimi (iki gözden geçiren + hakem): j3 = hakemin son kararı
    pf3 = R3 / "birlestir/pairs_final.csv"
    if pf3.exists():
        pf = pd.read_csv(pf3)
        for r in pf.to_dict("records"):
            kes = "olasi" if str(r.get("c3", "")).lower() == "low" else "kesin"
            ekle(r["query_rel"], r["cand_rel"], r["j3"], f"r3_denetim_orneklem:{r['cand_id']}", kesinlik=kes)
        print(f"r3 kör örneklem kararları eklendi: {len(pf)}")
    else:
        print(f"[!] {pf3} yok: r3 kör örneklem kararları EKLENMEDİ (resmî koşuda olmalı)")
    # hakemin filigran vetosu nüfus taraması ve r3 eleştirmeninin kendi görsel kararları
    for r in pd.read_csv(R3 / "birlestir/filigran_populasyon_kararlar.csv").to_dict("records"):
        ekle(r["rel_i"], r["rel_j"], r["j3"], "r3_denetim_hakem_filigran", r["reason"])
    for r in pd.read_csv(R3 / "elestirmen/e08_gorsel_kararlar.csv").to_dict("records"):
        ekle(tek(r["a"]), tek(r["b"]), "unrelated" if r["karar"] == "false_edge" else r["karar"],
             f"r3_denetim_elestirmen:{r['id']}", r["gerekce"], kesinlik="olasi" if r["guven"] == "low" else None)
    # r4 taramasında gözden geçirenlerin kendi listeleri DIŞINDA rastlayıp notlarına yazdığı bağlar (tek gözle, montaj
    # düzeyinde; oneri/r4/cikti/tarama_notlari.txt). İkisini iki gözden geçiren ayrı ayrı gördü → kesin; diğerleri olası
    for a_, b_, kes_ in (("Imgclean_561_0.jpg", "Imgdirty_176_1.jpg", "kesin"), ("Imgclean_500_0.jpg", "Imgclean_621_0.jpg", "kesin"),
                         ("Imgdirty_656_1.jpg", "Imgclean_962_0.jpg", "kesin"), ("Dust (103).jpg", "Clean (89).jpg", "olasi"),
                         ("Imgclean_559_0.jpg", "Imgclean_563_0.jpg", "olasi"), ("Imgclean_559_0.jpg", "Imgclean_565_0.jpg", "olasi"),
                         ("Imgclean_1031_0.jpg", "Imgdirty_730_1.jpg", "olasi"), ("Clean (61).jpg", "Clean (112).jpg", "olasi"),
                         ("Imgclean_944_0.jpg", "Imgdirty_713_1.jpg", "olasi"), ("Imgclean_1014_0.jpg", "Imgclean_1022_0.jpg", "olasi"),
                         ("Imgclean_462_0.jpg", "Imgdirty_271_1.jpg", "olasi"), ("Imgdirty_262_1.jpg", "Imgclean_557_0.jpg", "olasi"),
                         ("Imgclean_693_0.jpg", "Dust (76).jpg", "olasi"), ("Imgclean_477_0.jpg", "Imgclean_485_0.jpg", "olasi"),
                         ("Imgdirty_497_1.jpg", "Imgdirty_499_1.jpg", "olasi")):
        ekle(tek(a_), tek(b_), "same_scene", "r4_tarama_rastlanti", "gözden geçiren notu (liste dışı)", kesinlik=kes_)
    # r4.1 Kaggle taramasında liste dışı notlar (oneri/r4/cikti/tarama41_notlari.txt); iki ayrı notta geçen → kesin
    for a_, b_, kes_ in (("Imgclean_868_0.jpg", "Imgdirty_587_1.jpg", "olasi"), ("Imgclean_868_0.jpg", "Imgclean_872_0.jpg", "olasi"),
                         ("Imgclean_876_0.jpg", "Imgdirty_584_1.jpg", "olasi"), ("Imgdirty_600_1.jpg", "Imgclean_976_0.jpg", "olasi"),
                         ("Imgdirty_661_1.jpg", "Imgdirty_663_1.jpg", "olasi"), ("Imgdirty_661_1.jpg", "Imgdirty_664_1.jpg", "olasi"),
                         ("Imgclean_598_0.jpg", "Imgclean_597_0.jpg", "olasi"), ("Imgdirty_260_1.jpg", "Imgclean_545_0.jpg", "kesin"),
                         ("Imgdirty_259_1.jpg", "Imgclean_545_0.jpg", "olasi"), ("Imgdirty_268_1.jpg", "Imgclean_545_0.jpg", "olasi"),
                         ("Imgdirty_226_1.jpg", "Imgclean_615_0.jpg", "olasi"), ("Imgdirty_511_1.jpg", "Dust (65)-D.jpg", "olasi"),
                         ("Dust (49).jpg", "Imgdirty_248_1.jpg", "olasi"), ("Imgclean_559_0.jpg", "Imgdirty_181_1.jpg", "olasi"),
                         ("Imgclean_559_0.jpg", "Imgdirty_184_1.jpg", "olasi"), ("Imgdirty_245_1.jpg", "Imgclean_524_0.jpg", "olasi"),
                         ("Dust (47).jpg", "Imgclean_525_0.jpg", "olasi"), ("Imgdirty_474_1.jpg", "Imgclean_755_0.jpg", "olasi"),
                         ("Imgdirty_477_1.jpg", "Imgclean_757_0.jpg", "olasi"), ("Imgdirty_722_1.jpg", "Imgdirty_724_1.jpg", "olasi")):
        ekle(tek(a_), tek(b_), "same_scene", "r41_tarama_rastlanti", "gözden geçiren notu (liste dışı)", kesinlik=kes_)
    # r4 hazırlık görsel taraması (dogrulama/asama2_r4_hazirlik): tarayıcı + kör doğrulayıcı. İkisi de aynı/sahne
    # derse kesin; biri derse belirsiz (yine sahne kenarı, sızıntı açısından ihtiyatlı); ikisi de ilgisiz/yoksa ilgisiz
    R4 = DG / "asama2_r4_hazirlik"
    # turlar: 1 (girdi), 2 (girdi_tur2), r4.1 hedefli Kaggle taraması (girdi_r41b; girdi_r41 ve girdi2 kullanılmayan denemeler)
    for tur, gd in (("", "girdi"), ("2", "girdi_tur2"), ("41", "girdi_r41b")):
        girdi = R4 / gd
        if not girdi.exists():
            continue
        for p_ in sorted(girdi.glob("parca_*.csv")):
            k_ = p_.stem.split("_")[1]
            t_ = R4 / f"tarama{tur}_{k_}" / "kararlar.csv"
            if not t_.exists():
                print(f"[!] tarama kararı yok: {t_}")
                continue
            T = pd.read_csv(t_)
            d_ = R4 / f"dogrulama{tur}_{k_}" / "kararlar.csv"
            V = pd.read_csv(d_).set_index("kod") if d_.exists() else pd.DataFrame(columns=["karar"])
            for r in T.to_dict("records"):
                pz = {"same_photo", "same_scene"}
                k1, k2 = r["karar"], (V.at[r["kod"], "karar"] if r["kod"] in V.index else None)
                if k1 in pz and k2 in pz:
                    ham = "same_photo" if k1 == k2 == "same_photo" else "same_scene"
                elif k1 in pz or k2 in pz or k1 == "unsure" or k2 == "unsure":
                    ham = "unsure"
                else:
                    ham = "unrelated"
                # r4.1: doğrulayıcı yalnız pozitiflere baktı; tek tarayıcının "ilgisiz" kararı kesin sayılmaz (yoksa
                # geometrik sahne kenarını tek bir yapay zekâ kararı engelleyebilirdi; r4 kod denetçisinin bulgusu)
                kes = "olasi" if ham == "unrelated" else None
                ekle(r["rel_i"], r["rel_j"], ham, f"r4_tarama{tur}:{r['kod']}", f"{k1}/{k2}: {r['gerekce']}",
                     kesinlik=kes)

    # ---------------- r4 denetimi (dogrulama/asama2_r4/): kör örneklem (iki gözden geçiren + hakem) ve eleştirmen
    R4D = DG / "asama2_r4"
    pf4 = R4D / "birlestir/pairs_final.csv"
    if pf4.exists():
        for r in pd.read_csv(pf4).to_dict("records"):
            kes = "olasi" if str(r.get("j3_conf", "")).lower().startswith("low") else "kesin"
            ekle(r["query_rel"], r["cand_rel"], r["j3"], f"r4_denetim_orneklem:{r['cand_id']}", kesinlik=kes)
    # r4 eleştirmeninin gözle doğruladığı çiftler (elestirmen/kararlar.csv; dosya bozuk tırnaklı olduğu için elle)
    for a_, b_, s_, k_ in (("Imgclean_870_0.jpg", "Imgclean_866_0.jpg", "same_scene", "kesin"),
                           ("Imgdirty_605_1.jpg", "Imgdirty_608_1.jpg", "same_scene", "kesin"),
                           ("Imgclean_892_0.jpg", "Imgclean_887_0.jpg", "same_scene", "olasi"),
                           ("Imgdirty_532_1.jpg", "Imgdirty_855_1.jpg", "same_scene", "kesin"),
                           ("Imgdirty_532_1.jpg", "Imgdirty_860_1.jpg", "same_scene", "kesin"),
                           ("Imgdirty_532_1.jpg", "Imgdirty_863_1.jpg", "same_scene", "kesin"),
                           ("Dust (8).jpg", "Imgdirty_164_1.jpg", "same_scene", "olasi"),
                           ("Imgdirty_589_1.jpg", "Imgclean_872_0.jpg", "same_scene", "kesin"),
                           ("Imgdirty_610_1.jpg", "Imgdirty_605_1.jpg", "same_scene", "olasi"),
                           ("Imgdirty_862_1.jpg", "Imgdirty_863_1.jpg", "same_scene", "kesin"),
                           ("Imgclean_872_0.jpg", "Imgclean_870_0.jpg", "unsure", "olasi"),
                           ("Imgdirty_590_1.jpg", "Imgclean_887_0.jpg", "unsure", "olasi"),
                           ("Imgdirty_696_1.jpg", "Imgdirty_732_1.jpg", "unsure", "olasi"),
                           ("Imgdirty_586_1.jpg", "Imgclean_869_0.jpg", "unrelated", "olasi"),
                           ("Imgdirty_582_1.jpg", "Imgclean_873_0.jpg", "unrelated", "kesin"),
                           ("Dust (63).jpg", "Imgclean_653_0.jpg", "unrelated", "olasi")):
        ekle(tek(a_), tek(b_), s_, "r4_denetim_elestirmen", "r4 eleştirmeni (elestirmen/kararlar.csv)", kesinlik=k_)

    E = pd.DataFrame(satir)
    eksik = sorted((set(E["rel_a"]) | set(E["rel_b"])) - gecerli)
    if eksik:
        raise SystemExit(f"manifestte olmayan yol: {eksik[:5]}")
    E["oncelik"] = E["sinif"].map(ONCELIK)

    def birlestir(d):
        siniflar = "+".join(sorted(set(d["sinif"])))  # süzmeden önce: anlaşmazlık görünür kalsın
        if d["ust"].any():
            # r2 eleştirmeni yalnız r1/r2 kaynaklarını geçersiz kılar; daha yeni kaynakların (r3/r4 denetimleri,
            # taramalar, benim kararlarım) kararları onunla birlikte sınıf önceliğine girer (sızıntı açısından ihtiyatlı)
            d = d[d["ust"] | ~d["kaynak"].str.startswith(("r1_", "r2_"))]
        p = d["oncelik"].max()
        sec = d[d["oncelik"] == p]
        return pd.Series(dict(
            sinif={v: k for k, v in ONCELIK.items()}[p],
            kesinlik="kesin" if (sec["kesinlik"] == "kesin").any() else "olasi",
            karar_sayisi=len(d), siniflar=siniflar,
            kaynaklar="; ".join(sorted(set(d["kaynak"])))[:500]))

    ozet = E.groupby(["rel_a", "rel_b"]).apply(birlestir, include_groups=False).reset_index()
    ozet = ozet.sort_values(["sinif", "rel_a", "rel_b"])
    cikti = Path(args.cikti)
    cikti.mkdir(exist_ok=True)
    ozet.to_csv(cikti / "gorsel_etiketler.csv", index=False, encoding="utf-8-sig")
    E.drop(columns="oncelik").to_csv(cikti / "gorsel_etiketler_ham.csv", index=False, encoding="utf-8-sig")
    print(f"ham karar: {len(E)}  tekil çift: {len(ozet)}")
    print(pd.crosstab(ozet["sinif"], ozet["kesinlik"]).to_string())
    anl = ozet[ozet["siniflar"].str.contains(r"\+")]
    print(f"\nfarklı kararlı çift: {len(anl)}")
    print(anl["siniflar"].value_counts().to_string())


if __name__ == "__main__":
    main()
