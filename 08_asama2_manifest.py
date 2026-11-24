"""Adım 8 (r4.1) - 2. aşama için temiz, grup bazlı manifest (sahne/oturum düzeyinde gruplama).

Veri dosyalarına dokunmaz (salt okunur). Çıktılar ve önbellekler --cikti klasöründe (varsayılan asama2/).

Sürüm geçmişi (ayrıntı: NOTLAR.md §9; denetimler dogrulama/asama2_*/ ve dogrulama/asama2_r2/)
  r1 (asama2/r1/): katı = MD5 ∪ pHash; geniş = DINOv2 adaylarında SIFT. Denetim: sahne düzeyinde ~%18 sızıntı.
  r2 (asama2/r2/): 64 yön çiftli pHash, modelden bağımsız kapsamlı aday taraması, karşılıklı-NN RootSIFT + MAGSAC,
      aynı-fotoğraf düzeyinde çelişki/temsilci, EXIF/dosya no/görsel kenarlar. Kör denetim: ~%11-12,5 sızıntı;
      eşleştirici yön sırasına bağımlıydı, NCC örtüşme dışı sıfırlarla kirleniyordu, eşikler katıydı.
  r3 (asama2/r3/; kör denetim dogrulama/asama2_r3/):
    * derin, simetrik eşleştirici: RootSIFT 1400 px / 5000 nokta (arka plandaki ev, araç, baca gibi ortak işaretler
      ancak yüksek çözünürlükte eşleşiyor), karşılıklı en yakın komşu + Lowe 0,8 oran testi iki yönden birlikte,
      MAGSAC 5 px, hedef aynalı/aynasız; homografi geçerliliği yön bakımından simetrik. Karar çift sırasından bağımsız.
    * NCC normalleştirilmiş evrişimle (örtüşme dışı sızmaz), 800 px'te, küçük görüntünün çerçevesinde
    * adaylar katı grubun en büyük üyesi üzerinden; + aynı ailede |ΔN| ≤ 10 (8 aile), + Healthy↔SO çapraz etiketli ilk-20
      (önce/sonra), + Kaggle temiz−kirli numara farkı pencereleri (240-310, 350-410; S ≥ 0,80; r4: 200-420), gevşek tarama eşiği
    * sahne kenarı: geçerli homografi, inlier ve ayrık konum ≥ 10, altbilgi şeridi ya da çakışık çerçevede farklı içerik
      (filigran/şablon) değil. Eşikler r2 denetiminin etiketli pozitif/zor negatiflerinde seçildi.
    * aynı-fotoğraf kenarı: + küçük görüntüde inlier kapsamı ≥ 0,10; farklı etiketli çiftlerde NCC_hp ≥ 0,95 (önce/sonra
      çekimleri 0,85-0,90 verebiliyor; arada kalanlar sahne kenarı olur ve celiski_supheleri.csv'ye yazılır); görsel
      "aynı sahne"/"ilgisiz" çiftler aynı-fotoğraf bağı kuramaz (sahne kenarları etiketle engellenmez)
    * dosya numarasına dayalı doğrulamasız kenar varsayılan olarak KAPALI (--dosyano-kenar): ilgisiz işleri zincirleyip
      v1'in %16'sı büyüklüğünde bir grup kuruyordu; dosya no komşuları artık aday olarak derin doğrulamadan geçiyor
    * eşleşme önbelleği çift bazında ve artımlı; parametre/kod özetiyle doğrulanır; sızıntı tanısı temsilci düzeyinde

  r4 (asama2/r4/; r3 denetiminin bulguları; taze kör denetim dogrulama/asama2_r4/: %3,1, GA %1,3-7,1):
    * iki yönlü eşleştirme: MAGSAC örneklemesi, yeniden izdüşüm hatası ve kimlik ölçüsü yön bağımlıydı (790 çiftte 28
      sahne kararı sıraya göre değişiyordu); her aday çift A→B ve B→A ölçülür, kenar = iki yönün VEYA'sı (ihtiyatlı)
    * filigran/şablon vetosu daraltıldı: aynı noktadan çekilmiş önce/sonra fotoğrafları da "çakışık çerçeve, farklı
      içerik" imzası veriyordu (inl 1.711'e varan sızıntılar). Veto artık yalnız eşleşme altbilgide yoğunsa
      (alt_pay ≥ 0,5), inlier'lar küçük bir bölgedeyse (kapsam < 0,05) ya da görüntüler benzemiyorsa (S < 0,85)
    * çakışık çerçevede (kimlik < 0,10) NCC her zaman ölçülür (önce: ölçülmeyince filigran sayılıyordu)
    * kapsam (kap_kucuk) gerçek piksel alanı küçük olan görüntüde ölçülür (önce: 800 px NCC çerçevesine göre)
    * görsel "belirsiz" kararlar her sınıfta sahne kenarıdır (önce yalnız iki SF/EF arasında)
    * kesin "ilgisiz" görsel kararı geometrik sahne kenarını da engeller (doku/şablon zincirleri; ör. grup 113/140/325)
    * aynı etiketli, inl ≥ 200, kapsam ≥ 0,8, NCC_hp ≥ 0,6 çiftler aynı-fotoğraf (farklı filigranlı aynı stok foto)
    * önbellek anahtarları: tüm eşleştirme sabitleri ve yardımcı kaynakları, OpenCV sürümü, görüntü dosyalarının
      (yol, boyut, değişiklik zamanı) imzası; DINOv2 ve karma önbellekleri de bu imzayla doğrulanır
    * görsel etiketlere r3 denetiminin kararları eklendi (08a)

  r4.1 (bu sürüm): kod aynı (eşleşme önbelleği geçerli); yalnız görsel etiketler değişti (08a): r4 denetiminin
    örneklem ve eleştirmen kararları, 2.121 çiftlik hedefli Kaggle taraması (aynı boyut |ΔN| ≤ 5 ve en iyi temiz/kirli
    eşleri), tek tarayıcının "ilgisiz" kararı artık olası (sahne kenarını engellemez)

Gruplar (iç içe: kati ⊆ foto ⊆ sahne)
  kati_grup  : MD5 eşitliği ∪ dihedral pHash (64 yön çifti) Hamming ≤ --phash-esik (4).
  foto_grup  : kati ∪ aynı-fotoğraf kenarları [geçerli homografi, inlier ≥ --foto-inlier, ≥ 8 ayrık inlier konumu,
               küçük görüntüde inlier kapsamı ≥ --foto-kapsam, çakıştırılmış yüksek-geçiren NCC ≥ --foto-ncc]
               ∪ görsel "aynı fotoğraf" çiftleri. → celiskili ve temsilci bununla belirlenir.
  sahne_grup : foto ∪ sahne kenarları: (a) geometrik [geçerli homografi, inlier ≥ --sahne-inlier, her iki görüntüde
               ≥ --sahne-ayrik ayrık inlier konumu, altbilgi şeridi ya da filigran/şablon eşleşmesi değil; SF/EF içeren
               çiftlerde --azinlik-inlier]; (b) DINOv2 dihedral, katı-grup düzeyinde karşılıklı-10NN, S ≥ 0,97; (c) EXIF:
               aynı kamera modeli, |Δt| ≤ 10 dk; (d) görsel "aynı sahne" çiftleri (iki SF/EF arasında "belirsiz" de
               birleştirilir; r4: her sınıfta "belirsiz" de); (e) isteğe bağlı (--dosyano-kenar) Kaggle |ΔN| ≤ 2 ve S ≥ 0,90. Kat ataması bununla yapılır.
               Fazla gruplama sızıntı açısından güvenlidir (yalnız kat dengesini etkiler); grup bu yüzden "aynı sahne,
               aynı çekim oturumu ya da aynı fotoğrafçı serisi" anlamına gelebilir.
  Görsel etiketler: asama2/gorsel_etiketler.csv (08a_gorsel_etiket_derle.py; denetim ajanlarının gözle verdiği kararlar).

Dışlama ve varyantlar (02_egit.py --protocol grouped --fold-col <kat sütunu> --group-col sahne_grup)
  termal     : C#/S# (PVMD termal) dosyaları → dışlanır (yalnız RGB kıyaslama)
  celiskili  : foto_grup birden çok sınıf etiketi taşıyor → tüm foto_grup dışlanır
  temsilci   : uygun her foto_grup'tan bir dosya (en büyük piksel alanı → en kısa rel_path → sözlük sırası)
  kat_v1     : RGB, çelişkisiz, foto_grup başına tek temsilci (ÖNERİLEN birincil analiz). Dışlananlarda -1.
  kat_v2     : RGB, çelişkisiz, tüm kopyalar (aynı sahne_grup → aynı kat; yalnız duyarlılık analizi)
  kat_v1_t<s>: aynı gruplama, farklı kat tohumu (tekrarlı gruplu CV için; --tekrar-tohumlari)
  Katlar: StratifiedGroupKFold (shuffle, --tohum), tabaka = etiket×kaynak (--tabaka), grup = sahne_grup.

Kullanım:
    python 08a_gorsel_etiket_derle.py      # görsel kararları derle (denetimlerden sonra)
    python 08_asama2_manifest.py           # eşleşmeler artımlı önbellekli (ilk koşu ~45 dk GPU, sonra ~1 dk)
"""
import argparse
import hashlib
import inspect
import json
import re
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ortak import load_rgb, utf8_konsol

TERMAL = re.compile(r"^[CS]\d+\.", re.I)
AILE = [("Imgclean", r"^Imgclean_\d+_\d+\."), ("Imgdirty", r"^Imgdirty_\d+_\d+\."), ("C#", r"^C\d+\."),
        ("S#", r"^S\d+\."), ("Clean", r"^Clean \(\d+\)"), ("Bird", r"^Bird \(\d+\)"), ("Dust", r"^Dust \(\d+\)"),
        ("Snow", r"^Snow \(\d+\)"), ("Electrical", r"^Electrical \(\d+\)"), ("Physical", r"^Physical \(\d+\)")]
KAYNAK = {"Imgclean": "kaggle_toz", "Imgdirty": "kaggle_toz", "C#": "pvmd_termal", "S#": "pvmd_termal",
          "Clean": "pv_panel_defect", "Bird": "pv_panel_defect", "Dust": "pv_panel_defect", "Snow": "pv_panel_defect",
          "Electrical": "pv_panel_defect", "Physical": "pv_panel_defect"}
NO_AILELERI = ("Imgclean", "Imgdirty", "Clean", "Bird", "Dust", "Snow", "Electrical", "Physical")
NO_DESEN = r"^(Imgclean|Imgdirty)_(\d+)_|^(Clean|Bird|Dust|Snow|Electrical|Physical) \((\d+)\)"
AZINLIK = ("Structural_Fault", "Electrical_Fault")
UZUN = 1400         # derin doğrulama: SIFT görüntüsünün uzun kenarı (arka plan ayrıntısı için yüksek)
NKP = 5000          # derin doğrulama: SIFT nokta sayısı
UZUN_NCC = 800      # NCC (aynı-fotoğraf ölçüsü) için gri görüntünün uzun kenarı
NKP_TARAMA = 600    # kapsamlı tarama için (yanıta göre ilk 600)
ALTBILGI = 0.12     # stok fotoğraf altbilgi şeridi: görüntünün alt %12'si
MAGSAC_PX = 5.0     # 1400 px çerçevede yeniden izdüşüm eşiği
ESLESTIRICI = "r4-derin"  # eşleştirme fonksiyonlarının kaynak kodu da önbellek anahtarına girer
KIMLIK_NCC = 0.10   # çakışık çerçevede (kimlik < bu) NCC her zaman ölçülür (filigran kararı NCC'ye dayanır)


def dosya_imzasi(yollar):
    """Görüntü dosyalarının (yol, bayt boyu, değişiklik zamanı) özeti: önbellekler bayat kalmasın."""
    h = hashlib.md5()
    for y in yollar:
        st = Path(y).stat()
        h.update(f"{y}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def aile(ad):
    for a, rx in AILE:
        if re.search(rx, ad, re.I):
            return a
    return "diger"


def dosya_no(dosyalar):
    """→ (aile, numara) dizileri; aileler: NO_AILELERI (yoksa None / nan)."""
    x = pd.Series(dosyalar).str.extract(NO_DESEN)
    fam = x[0].fillna(x[2]).to_numpy(dtype=object)
    no = pd.to_numeric(x[1].fillna(x[3]), errors="coerce").to_numpy()
    return fam, no


def md5_dosya(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest() if Path(p).exists() else None


# ------------------------------------------------------------------ MD5 + dihedral pHash
def hash_isci(yol):
    import imagehash
    from PIL import Image
    T = Image.Transpose
    donusum = [[], [T.ROTATE_90], [T.ROTATE_180], [T.ROTATE_270], [T.FLIP_LEFT_RIGHT],
               [T.FLIP_LEFT_RIGHT, T.ROTATE_90], [T.FLIP_LEFT_RIGHT, T.ROTATE_180], [T.FLIP_LEFT_RIGHT, T.ROTATE_270]]
    with open(yol, "rb") as fh:
        veri = fh.read()
    kayit = {"path": yol, "md5": hashlib.md5(veri).hexdigest()}
    im = Image.open(yol)
    if im.format in ("JPEG", "MPO"):
        im.draft("L", (512, 512))
    try:
        im.seek(0)
    except Exception:
        pass
    g = im.convert("L")
    for k, ops in enumerate(donusum):
        t = g
        for op in ops:
            t = t.transpose(op)
        kayit[f"ph{k}"] = str(imagehash.phash(t))  # 16 haneli onaltılık metin (CSV'de taşmasın)
    return kayit


def dihedral_uzaklik(ph):
    """ph: (N, 8) uint64, 8 dihedral dönüşümün pHash'i. D[i, j] = min_{a,b} Hamming(ph_a(i), ph_b(j)).
    pHash yansımaya eşdeğişken olmadığından (DCT işaretleri medyanı kaydırır) 64 çiftin tümüne bakılır."""
    N = len(ph)
    D = np.full((N, N), 64, dtype=np.uint8)
    for a in range(8):
        for b in range(8):
            np.minimum(D, np.bitwise_count(ph[:, a][:, None] ^ ph[:, b][None, :]).astype(np.uint8), out=D)
    np.fill_diagonal(D, 0)
    return D


# ------------------------------------------------------------------ DINOv2 dihedral gömmeler
def dinov2_gommeler(yollar, batch=64):
    import timm
    import torch
    from torchvision import transforms
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=224)
    model = model.to(cihaz).eval()
    dc = timm.data.resolve_data_config({}, model=model)
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(dc["mean"], dc["std"])])
    E = np.zeros((8, len(yollar), model.num_features), dtype=np.float16)
    with torch.no_grad():
        for i in range(0, len(yollar), batch):
            x = torch.stack([tf(load_rgb(p, 224)) for p in yollar[i:i + batch]]).to(cihaz)
            k = 0
            for ayna in (False, True):
                xa = torch.flip(x, dims=[3]) if ayna else x
                for r in range(4):
                    xr = torch.rot90(xa, r, dims=[2, 3])
                    with torch.autocast(device_type=cihaz, dtype=torch.bfloat16, enabled=(cihaz == "cuda")):
                        f = model(xr).float()
                    E[k, i:i + len(x)] = torch.nn.functional.normalize(f, dim=1).cpu().numpy().astype(np.float16)
                    k += 1
    return E


def dihedral_benzerlik(E):
    """S[i, j] = max_k cos(e0_i, ek_j), simetrik (GPU varsa GPU'da)."""
    import torch
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    e0 = torch.from_numpy(E[0]).to(cihaz).float()
    S = None
    for k in range(8):
        s = e0 @ torch.from_numpy(E[k]).to(cihaz).float().T
        S = s if S is None else torch.maximum(S, s)
    S = torch.maximum(S, S.T)
    S.fill_diagonal_(-1.0)
    return S.cpu().numpy()


def grup_benzerligi(S, g):
    """S (n, n), g (n,) 0..G-1 → Sg (G, G): iki grubun üyeleri arasındaki en yüksek S (köşegen -1)."""
    import torch
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    St = torch.from_numpy(np.ascontiguousarray(S, dtype=np.float32)).to(cihaz)
    gt = torch.from_numpy(g.astype(np.int64)).to(cihaz)
    G, n = int(g.max()) + 1, St.shape[0]
    R = torch.full((G, n), -2.0, device=cihaz).scatter_reduce(0, gt[:, None].expand(n, n), St, reduce="amax")
    C = torch.full((G, G), -2.0, device=cihaz).scatter_reduce(1, gt[None, :].expand(G, n), R, reduce="amax")
    C.fill_diagonal_(-1.0)
    return C.cpu().numpy()


# ------------------------------------------------------------------ EXIF
def exif_isci(yol):
    from PIL import Image
    model = dt = None
    try:
        ex = Image.open(yol).getexif()
        model = ex.get(272)
        dt = ex.get(306)
        try:
            dt = ex.get_ifd(0x8769).get(36867) or dt
        except Exception:
            pass
    except Exception:
        pass
    zaman = None
    if isinstance(dt, str):
        try:  # .timestamp() Windows'ta 1970 öncesi/bozuk tarihlerde OSError verir; sabit başlangıca göre saniye
            zaman = (datetime.strptime(dt.strip()[:19], "%Y:%m:%d %H:%M:%S") - datetime(1900, 1, 1)).total_seconds()
        except (ValueError, OverflowError, OSError):
            zaman = None
    model = model.strip().strip("\x00") if isinstance(model, str) and model.strip().strip("\x00") else None
    return model, dt if isinstance(dt, str) else None, zaman


# ------------------------------------------------------------------ RootSIFT özellikleri
def ozellik_isci(yol):
    """→ (gri_ncc (uzun kenar ≤ 800), olcek (gri_ncc / özellik çerçevesi), şekil (özellik çerçevesi),
    [(P, D, R) normal, (P, D, R) ayna]); D: L2 birim RootSIFT (float16), R: yanıt. Özellikler ≤ 1400 px'te."""
    import cv2
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    cv2.setNumThreads(1)
    im = Image.open(yol)
    try:
        im.seek(0)
    except Exception:
        pass
    im = im.convert("L")
    W0, H0 = im.size

    def olcekle(uzun):
        s_ = uzun / max(W0, H0)
        return im.resize((max(1, round(W0 * s_)), max(1, round(H0 * s_))), Image.LANCZOS) if s_ < 1 else im

    g = np.asarray(olcekle(UZUN), dtype=np.uint8)
    gn = np.asarray(olcekle(UZUN_NCC), dtype=np.uint8)
    out = []
    for ayna in (False, True):
        gg = np.ascontiguousarray(g[:, ::-1]) if ayna else g
        kp, des = cv2.SIFT_create(nfeatures=NKP).detectAndCompute(gg, None)
        if des is None or not kp:
            out.append((np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float16), np.zeros(0, np.float32)))
            continue
        des = np.sqrt(des / (des.sum(1, keepdims=True) + 1e-7))
        des = des / (np.linalg.norm(des, axis=1, keepdims=True) + 1e-7)
        out.append((np.float32([k.pt for k in kp]), des.astype(np.float16), np.float32([k.response for k in kp])))
    return gn, gn.shape[1] / g.shape[1], g.shape, out


def kapsamli_tarama(F, cihaz, arka=50, TB=128):
    """Tüm çiftler için veri tabanı düzeyinde oran testi. Sorgu noktası q, hedef j'deki en iyi kosinüsü (M[q, j]) arka
    planın (tüm 2N hedef arasında 50. en yüksek) oran² katından iyiyse j için sayılır. → r70, r80 (N, N) int16 simetrik.
    GPU: ~1 GB tanımlayıcı + TB=128'de ~0,75 GB çarpım bloğu."""
    import torch
    N, K = len(F), NKP_TARAMA
    T = np.zeros((2, N, K, 128), np.float16)
    n0 = np.zeros(N, np.int64)
    for yon in (0, 1):
        for i, f in enumerate(F):
            _, Dd, R = f[3][yon]
            o = np.argsort(-R)[:K]
            T[yon, i, :len(o)] = Dd[o]
            if yon == 0:
                n0[i] = len(o)
    T = torch.from_numpy(T.reshape(-1, 128)).to(cihaz)  # [normal (N*K); ayna (N*K)]
    nt = torch.from_numpy(n0).to(cihaz)
    oran = {"r80": 0.64, "r70": 0.49}
    C = {k: torch.zeros((N, 2 * N), dtype=torch.int16, device=cihaz) for k in oran}
    QB = 8
    kidx = torch.arange(K, device=cihaz)
    with torch.no_grad():
        for q0 in range(0, N, QB):
            q1 = min(N, q0 + QB)
            qb = q1 - q0
            Q = T[q0 * K:q1 * K]
            M = torch.empty((qb * K, 2 * N), dtype=torch.float16, device=cihaz)
            for a in range(0, 2 * N, TB):
                b = min(2 * N, a + TB)
                M[:, a:b] = (Q @ T[a * K:b * K].T).view(qb * K, b - a, K).amax(-1)
            Mf = M.float()
            bg = torch.topk(Mf, arka, dim=1).values[:, -1:]
            d1 = (2 - 2 * Mf).clamp_min(0)
            dbg = (2 - 2 * bg).clamp_min(1e-6)
            gecerli = (kidx[None, :] < nt[q0:q1, None]).reshape(-1, 1)
            for k, r2 in oran.items():
                C[k][q0:q1] = ((d1 < r2 * dbg) & gecerli).view(qb, K, 2 * N).sum(1).to(torch.int16)
    sonuc = {}
    for k, c in C.items():
        a = torch.maximum(c[:, :N], c[:, N:])
        a = torch.maximum(a, a.T)
        a.fill_diagonal_(-1)
        sonuc[k] = a.cpu().numpy()
    del T, C
    torch.cuda.empty_cache()
    return sonuc["r70"], sonuc["r80"]


def _kapsam(cv2, p, alan):
    if len(p) < 3:
        return 0.0
    return float(cv2.contourArea(cv2.convexHull(p.astype(np.float32))) / max(alan, 1))


def _disbukey(cv2, H, shp):
    h, w = shp
    k = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64) @ H.T
    if not (np.isfinite(k).all() and (k[:, 2] > 1e-9).all()):
        return False, None
    q = k[:, :2] / k[:, 2:3]
    return bool(cv2.isContourConvex(q.astype(np.float32).reshape(-1, 1, 2))), q


def geometri(cv2, src, dst, shp_a, shp_b, ayna=False):
    """MAGSAC homografisi (A → B) ve sağlamlık ölçüleri. Geçerlilik yön bakımından simetrik: det > 0 ve A'nın köşeleri
    H ile ya da B'nin köşeleri H⁻¹ ile dışbükey dörtgene gider (geniş çekim/yakın çekimde biri ufkun ötesine düşebilir).
    kimlik: A'nın köşelerinin (B boyutuna göre normalleştirilmiş) birim karenin köşelerinden en büyük sapması; ~0 ise
    iki görüntü aynı çerçevede çakışıyor (aynı filigran/şablon ya da aynı kadraj)."""
    out = dict(inl=0, umin=0, gecerli=False, alt_pay=np.nan, kap_a=0.0, kap_b=0.0, olcek=np.nan, kimlik=np.nan, H=None)
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, MAGSAC_PX, maxIters=3000, confidence=0.999)
    if H is None or mask is None or not np.isfinite(H).all():
        return out
    k = mask.ravel().astype(bool)
    out["inl"] = int(k.sum())
    if out["inl"] < 4:
        return out
    sa, sb = src[k], dst[k]
    ha, wa = shp_a
    hb, wb = shp_b
    ua = len(np.unique(np.round(sa / 3), axis=0))
    ub = len(np.unique(np.round(sb / 3), axis=0))
    det = float(np.linalg.det(H[:2, :2]))
    gec_a, q = _disbukey(cv2, H, shp_a)
    gec_b = False
    if det > 0 and abs(np.linalg.det(H)) > 1e-12:
        gec_b, _ = _disbukey(cv2, np.linalg.inv(H), shp_b)
    gecerli = bool(det > 0 and (gec_a or gec_b))
    kimlik = np.nan
    if q is not None and not ayna:
        kimlik = float(np.abs(q / np.array([wb, hb]) - np.array([[0, 0], [1, 0], [1, 1], [0, 1]])).max())
    alt = min(float((sa[:, 1] > (1 - ALTBILGI) * ha).mean()), float((sb[:, 1] > (1 - ALTBILGI) * hb).mean()))
    out.update(umin=min(ua, ub), gecerli=gecerli, alt_pay=alt, kap_a=_kapsam(cv2, sa, ha * wa),
               kap_b=_kapsam(cv2, sb, hb * wb), olcek=float(np.sqrt(abs(det))), kimlik=kimlik, H=H)
    return out


def _nbulanik(cv2, x, m, s):
    """Maskeli (normalleştirilmiş evrişimli) Gauss bulanıklığı: maske dışındaki pikseller sonuca sızmaz."""
    return cv2.GaussianBlur(x * m, (0, 0), s) / np.maximum(cv2.GaussianBlur(m, (0, 0), s), 1e-3)


def ncc_olc(cv2, gf, gm, H):
    """gm'yi gf'nin çerçevesine çakıştır (H: gf koordinatı → gm koordinatı); örtüşmede bulanık ve yüksek-geçiren NCC.
    Bulanıklıklar örtüşme maskesiyle normalleştirilir; NCC maskenin 9×9 aşınmış içinde ölçülür."""
    h, w = gf.shape
    inv = cv2.WARP_INVERSE_MAP
    wb = cv2.warpPerspective(gm.astype(np.float32), H, (w, h), flags=cv2.INTER_LINEAR | inv, borderValue=0)
    m = cv2.warpPerspective(np.ones_like(gm, np.float32), H, (w, h), flags=cv2.INTER_NEAREST | inv, borderValue=0)
    m = cv2.erode((m > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8))
    ic = cv2.erode(m, np.ones((9, 9), np.uint8)).astype(bool)
    m = m.astype(np.float32)
    ort = float(m.mean())
    if ic.sum() < 500:
        return dict(ortusme=ort, ncc=np.nan, ncc_hp=np.nan)
    a = _nbulanik(cv2, gf.astype(np.float32), m, 1.5)
    b = _nbulanik(cv2, wb, m, 1.5)

    def ncc(x, y):
        x = x[ic] - x[ic].mean()
        y = y[ic] - y[ic].mean()
        return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-6))

    return dict(ortusme=ort, ncc=ncc(a, b), ncc_hp=ncc(a - _nbulanik(cv2, a, m, 6), b - _nbulanik(cv2, b, m, 6)))


def eslestir(ciftler, F, cihaz):
    """ciftler: [(i, j)] yerel (RGB) indeksler; F[i] = ozellik_isci çıktısı. Simetrik eşleştirme: karşılıklı en yakın
    komşu ve Lowe 0.8 oran testi İKİ yönden birlikte (satırda ya da sütunda geçen karşılıklı çiftler). MAGSAC
    örneklemesi, yeniden izdüşüm hatası ve kimlik ölçüsü yön bağımlıdır; bu yüzden main() her çifti iki yönde de
    ölçer ve kenarları VEYA ile birleştirir. Hedef aynalı/aynasız denenir. NCC yalnız gerekince (aynı-fotoğraf adayı ya da çakışık çerçeve),
    800 px gri görüntüde, küçük görüntünün çerçevesinde ölçülür. Tanımlayıcılar GPU'ya tembel yüklenir."""
    import cv2
    import torch

    def yukle(i, y):
        return torch.from_numpy(F[i][3][y][1]).to(cihaz)

    def karsilikli(da, db):
        if len(da) < 8 or len(db) < 8:
            return None
        d2 = (2 - 2 * (da.float() @ db.float().T)).clamp_min(0)
        vr, ir = torch.topk(d2, 2, dim=1, largest=False)
        vc, ic = torch.topk(d2, 2, dim=0, largest=False)
        q = torch.arange(len(da), device=da.device)
        j = ir[:, 0]
        ok = (ic[0, j] == q) & ((vr[:, 0] < 0.64 * vr[:, 1]) | (vc[0, j] < 0.64 * vc[1, j]))
        return q[ok].cpu().numpy(), j[ok].cpu().numpy()

    def skor(r):
        return (bool(r["gecerli"] and r["umin"] >= 10), r["umin"], r["inl"])

    sonuc = [None] * len(ciftler)
    sira = sorted(range(len(ciftler)), key=lambda k: ciftler[k])  # i'ye göre sıralı: A tanımlayıcısı yeniden kullanılır
    a_i, a_t = None, None
    t0 = time.time()
    for n, k in enumerate(sira):
        i, j = ciftler[k]
        if a_i != i:
            a_i, a_t = i, yukle(i, 0)
        en, iyi_en = None, 0
        for ayna in (0, 1):
            r = karsilikli(a_t, yukle(j, ayna))
            if r is None:
                continue
            qi, ti = r
            iyi_en = max(iyi_en, len(qi))
            if len(qi) < 8:
                continue
            g = geometri(cv2, F[i][3][0][0][qi], F[j][3][ayna][0][ti], F[i][2], F[j][2], ayna=bool(ayna))
            g["iyi"], g["ayna"] = len(qi), bool(ayna)
            if en is None or skor(g) > skor(en):
                en = g
            if ayna == 0 and g["gecerli"] and g["umin"] >= 30:
                break
        if en is None:
            en = dict(inl=0, umin=0, gecerli=False, alt_pay=np.nan, kap_a=0.0, kap_b=0.0, olcek=np.nan, kimlik=np.nan,
                      H=None, iyi=iyi_en, ayna=False)
        Gi, Gj = F[i][0], F[j][0]
        kucuk_i = Gi.size <= Gj.size
        r = dict(inl=en["inl"], iyi=en["iyi"], umin=en["umin"], gecerli=en["gecerli"], alt_pay=en["alt_pay"],
                 olcek=en["olcek"], kimlik=en["kimlik"], ayna=en["ayna"], kap_i=en["kap_a"], kap_j=en["kap_b"],
                 kap_kucuk=en["kap_a"] if kucuk_i else en["kap_b"], i=i, j=j)
        ncc_gerek = en["H"] is not None and en["gecerli"] and (
            (en["inl"] >= 15 and en["umin"] >= 8 and r["kap_kucuk"] >= 0.05) or (en["kimlik"] < KIMLIK_NCC))
        if ncc_gerek:
            ra, rb = F[i][1], F[j][1]  # özellik çerçevesi → NCC çerçevesi
            Hn = np.diag([rb, rb, 1.0]) @ en["H"] @ np.diag([1 / ra, 1 / ra, 1.0])
            gj = np.ascontiguousarray(Gj[:, ::-1]) if en["ayna"] else Gj
            r.update(ncc_olc(cv2, Gi, gj, Hn) if kucuk_i else ncc_olc(cv2, gj, Gi, np.linalg.inv(Hn)))
        else:
            r.update(ortusme=np.nan, ncc=np.nan, ncc_hp=np.nan)
        sonuc[k] = r
        if n % 20000 == 0:
            print(f"  eşleştirme {n}/{len(ciftler)}  {time.time() - t0:.0f} sn", flush=True)
    torch.cuda.empty_cache()
    return sonuc


# ------------------------------------------------------------------ grafik yardımcıları
def bilesenler(n, kenar_listesi):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    ed = [np.asarray(e, dtype=np.int64).reshape(-1, 2) for e in kenar_listesi if len(e)]
    ed = np.concatenate(ed) if ed else np.zeros((0, 2), np.int64)
    A = coo_matrix((np.ones(len(ed), dtype=np.int8), (ed[:, 0], ed[:, 1])), shape=(n, n))
    return connected_components(A, directed=False)[1]


def grup_ozeti(g, etiket, maske=None):
    if maske is not None:
        g, etiket = g[maske], etiket[maske]
    s = pd.Series(g)
    boy = s.value_counts()
    karisik = pd.DataFrame({"g": g, "y": etiket}).groupby("g")["y"].nunique()
    return dict(grup=int(s.nunique()), en_buyuk=int(boy.max()), tekil_olmayan_grup=int((boy > 1).sum()),
                karisik_etiketli=int((karisik > 1).sum()), ilk5=boy.head(5).astype(int).tolist())


def karsilikli_knn(Sg, k, tau):
    G = len(Sg)
    idx = np.argpartition(-Sg, k - 1, axis=1)[:, :k]
    komsu = [set(r) for r in idx.tolist()]
    out = []
    for a in range(G):
        for b in idx[a]:
            b = int(b)
            if b > a and Sg[a, b] >= tau and a in komsu[b]:
                out.append((a, b))
    return out


def main():
    utf8_konsol()
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="denetim_final/manifest.csv")
    ap.add_argument("--cikti", default="asama2")
    ap.add_argument("--onbellek", default=None, help="karmalar/DINOv2/eşleşme önbelleği klasörü (varsayılan --cikti)")
    ap.add_argument("--gorsel", default="asama2/gorsel_etiketler.csv", help="08a çıktısı; yoksa elle kenar eklenmez")
    ap.add_argument("--phash-esik", type=int, default=4)
    ap.add_argument("--aday-knn", type=int, default=30, help="katı-grup düzeyinde DINOv2 ilk-k aday (taban yok)")
    ap.add_argument("--aday-phash", type=int, default=12)
    ap.add_argument("--aday-no", type=int, default=10, help="aynı ailede |ΔN| ≤ bu değer olan çiftler aday")
    ap.add_argument("--aday-capraz", type=int, default=20, help="Healthy↔SO çapraz etiketli ilk-k aday (önce/sonra)")
    ap.add_argument("--aday-ofset", default="200-420",
                    help="Imgclean_N − Imgdirty_M fark pencereleri (önce/sonra işleri); pencerede ve S ≥ --aday-ofset-tau olanlar aday")
    ap.add_argument("--aday-ofset-tau", type=float, default=0.80)
    ap.add_argument("--tarama-r70", type=int, default=6)
    ap.add_argument("--tarama-r80", type=int, default=12)
    ap.add_argument("--foto-inlier", type=int, default=15)
    ap.add_argument("--foto-ncc", type=float, default=0.85, help="aynı etiketli çiftlerde aynı-fotoğraf NCC_hp eşiği")
    ap.add_argument("--foto-ncc-capraz", type=float, default=0.95,
                    help="farklı etiketli çiftlerde (otomatik çelişki dışlaması) NCC_hp eşiği; önce/sonra çekimleri "
                         "0,85-0,90 verebildiği için yüksek; arada kalanlar sahne kenarı olur ve gözden geçirme listesine yazılır")
    ap.add_argument("--foto-kapsam", type=float, default=0.10, help="küçük görüntüde inlier dışbükey zarf kapsamı")
    ap.add_argument("--sahne-inlier", type=int, default=10)
    ap.add_argument("--sahne-ayrik", type=int, default=10)
    ap.add_argument("--azinlik-inlier", type=int, default=10, help="SF/EF içeren çiftlerde sahne inlier ve ayrık eşiği")
    ap.add_argument("--filigran-kimlik", type=float, default=0.10, help="çakışık çerçeve sapması (filigran/şablon)")
    ap.add_argument("--filigran-ncc", type=float, default=0.6, help="çakışık çerçevede bu NCC_hp altı = farklı fotoğraf")
    ap.add_argument("--filigran-altpay", type=float, default=0.5, help="filigran vetosu: inlier'ların altbilgi payı ≥ bu")
    ap.add_argument("--filigran-kapsam", type=float, default=0.05, help="filigran vetosu: ya da kapsam < bu")
    ap.add_argument("--filigran-s", type=float, default=0.85, help="filigran vetosu: ya da DINOv2 S < bu")
    ap.add_argument("--foto-filigranli-inlier", type=int, default=200)
    ap.add_argument("--foto-filigranli-kapsam", type=float, default=0.8)
    ap.add_argument("--foto-filigranli-ncc", type=float, default=0.6)
    ap.add_argument("--belirsiz-yalniz-azinlik", action="store_true",
                    help="(r3 davranışı) görsel 'belirsiz' yalnız iki SF/EF arasında sahne kenarı olsun")
    ap.add_argument("--ilgisiz-sahne-vetosu-yok", action="store_true",
                    help="(r3 davranışı) kesin 'ilgisiz' görsel kararı geometrik sahne kenarını engellemesin")
    ap.add_argument("--dino-k", type=int, default=10)
    ap.add_argument("--dino-tau", type=float, default=0.97)
    ap.add_argument("--exif-dk", type=float, default=10.0)
    ap.add_argument("--dosyano-kenar", action="store_true",
                    help="(varsayılan KAPALI) Kaggle |ΔN| ≤ --dosyano-fark ve S ≥ --dosyano-tau çiftlerini doğrulamasız kenar yap. "
                         "r3'te kapalı: derin eşleştirici dosya no komşularını aday olarak doğruluyor; körlemesine kenarlar "
                         "ilgisiz işleri zincirleyip 343 görüntülük (v1'in %16'sı) bir grup kuruyordu")
    ap.add_argument("--dosyano-fark", type=int, default=2)
    ap.add_argument("--dosyano-tau", type=float, default=0.90)
    ap.add_argument("--kat", type=int, default=5)
    ap.add_argument("--tohum", type=int, default=42)
    ap.add_argument("--tekrar-tohumlari", default="1,2,3,4")
    ap.add_argument("--tabaka", choices=["etiket", "etiket_kaynak"], default="etiket_kaynak")
    ap.add_argument("--isci", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    cikti = Path(args.cikti)
    cikti.mkdir(parents=True, exist_ok=True)
    onb = Path(args.onbellek) if args.onbellek else cikti
    onb.mkdir(parents=True, exist_ok=True)

    m = pd.read_csv(args.manifest)
    m = m[m["ok"]].reset_index(drop=True)
    m["aile"] = m["dosya"].map(aile)
    m["kaynak"] = m["aile"].map(KAYNAK)
    m["termal"] = m["dosya"].map(lambda f: bool(TERMAL.match(f)))
    N = len(m)
    yollar = m["path"].tolist()
    rel = m["rel_path"].to_numpy()
    r2i = {r: i for i, r in enumerate(rel)}
    etiket = m["label"].to_numpy()

    # --- karmalar (önbellek yol anahtarlı; eksik yol varsa yeniden hesaplanır)
    hcsv, himza = onb / "karmalar.csv", onb / "karmalar_imza.txt"
    imza = dosya_imzasi(yollar)
    h = None
    if hcsv.exists() and himza.exists() and himza.read_text(encoding="utf-8").strip() == imza:
        h = pd.read_csv(hcsv, dtype=str)
    if h is None or not set(yollar) <= set(h["path"]):
        with Pool(args.isci) as havuz:
            h = pd.DataFrame(havuz.map(hash_isci, yollar, chunksize=16))
        h.to_csv(hcsv, index=False)
        himza.write_text(imza + "\n", encoding="utf-8")
    h = h.set_index("path").loc[m["path"]].reset_index()
    m["md5"] = h["md5"].to_numpy()
    ph = np.stack([np.array([int(x, 16) for x in h[f"ph{k}"]], dtype=np.uint64) for k in range(8)], 1)
    D = dihedral_uzaklik(ph)
    md5_id = pd.factorize(m["md5"])[0]
    esit = md5_id[:, None] == md5_id[None, :]
    yakin = (D <= args.phash_esik) | esit
    np.fill_diagonal(yakin, False)
    e_kati = np.argwhere(np.triu(yakin, 1))
    m["kati_grup"] = bilesenler(N, [e_kati])
    print("katı gruplar:", grup_ozeti(m["kati_grup"].to_numpy(), etiket), flush=True)

    # --- DINOv2 dihedral benzerlik (önbellek + yol yan dosyası)
    ecache, eyol = onb / "dinov2_d4_gommeler.npy", onb / "dinov2_d4_gommeler_yollar.txt"
    eimza = onb / "dinov2_d4_gommeler_imza.txt"  # dosya imzası | .npy'nin MD5'i
    if (ecache.exists() and eyol.exists() and eimza.exists() and eyol.read_text(encoding="utf-8").splitlines() == yollar
            and eimza.read_text(encoding="utf-8").strip() == f"{imza}|{md5_dosya(ecache)}"):
        E = np.load(ecache)
    else:
        print("DINOv2 gömmeleri hesaplanıyor (önbellek yok ya da imza uyuşmuyor)...", flush=True)
        E = dinov2_gommeler(yollar)
        np.save(ecache, E)
        eyol.write_text("\n".join(yollar) + "\n", encoding="utf-8")
        eimza.write_text(f"{imza}|{md5_dosya(ecache)}\n", encoding="utf-8")
    S = dihedral_benzerlik(E)
    del E

    # --- RGB alt kümesi (termal dosyalar eşleştirmeye girmez; hepsi dışlanır)
    rgb = np.flatnonzero(~m["termal"].to_numpy())
    n_rgb = len(rgb)
    rgb_yerel = {g: k for k, g in enumerate(rgb)}
    S_r = S[np.ix_(rgb, rgb)]
    D_r = D[np.ix_(rgb, rgb)]
    kg_r = pd.factorize(m["kati_grup"].to_numpy()[rgb])[0]
    lab_r = etiket[rgb]
    m["piksel"] = m["width"] * m["height"]
    m["_uz"] = m["rel_path"].str.len()
    # katı grubun eşleştirmede kullanılan üyesi: en büyük piksel → en kısa yol → sözlük sırası
    sira_kg = pd.DataFrame({"y": np.arange(n_rgb), "kg": kg_r, "px": m["piksel"].to_numpy()[rgb],
                            "uz": m["_uz"].to_numpy()[rgb], "rel": rel[rgb]})
    sira_kg = sira_kg.sort_values(["kg", "px", "uz", "rel"], ascending=[True, False, True, True])
    kg_uye = sira_kg.groupby("kg")["y"].first().to_numpy()  # kg → yerel indeks

    # --- EXIF ve dosya numarası
    with Pool(args.isci) as havuz:
        ex = havuz.map(exif_isci, [yollar[i] for i in rgb], chunksize=16)
    ex_model = np.array([e[0] for e in ex], dtype=object)
    ex_zaman = np.array([np.nan if e[2] is None else e[2] for e in ex], dtype=float)
    fam_all, no_all = dosya_no(m["dosya"])
    fam_r, no_r = fam_all[rgb], no_all[rgb]

    def dosyano_ciftleri(maks, aileler=("Imgclean", "Imgdirty")):
        out = []
        for F_ in aileler:
            ii = np.flatnonzero(fam_r == F_)
            sira = ii[np.argsort(no_r[ii], kind="stable")]
            for x in range(len(sira)):
                for y in range(x + 1, len(sira)):
                    if no_r[sira[y]] - no_r[sira[x]] > maks:
                        break
                    out.append((min(sira[x], sira[y]), max(sira[x], sira[y])))
        return out

    def exif_ciftleri(maks_sn):
        out = []
        ii = np.flatnonzero(pd.notna(ex_model) & np.isfinite(ex_zaman))
        for mod in sorted(set(ex_model[ii])):  # sıralı: çıktı süreçten bağımsız (str hash sırası değişir)
            jj = ii[ex_model[ii] == mod]
            jj = jj[np.argsort(ex_zaman[jj], kind="stable")]
            for x in range(len(jj)):
                for y in range(x + 1, len(jj)):
                    if ex_zaman[jj[y]] - ex_zaman[jj[x]] > maks_sn:
                        break
                    out.append((min(jj[x], jj[y]), max(jj[x], jj[y])))
        return out

    gorsel = pd.DataFrame(columns=["rel_a", "rel_b", "sinif", "kesinlik"])
    if Path(args.gorsel).exists():
        gorsel = pd.read_csv(args.gorsel)
        if "kesinlik" not in gorsel.columns:
            gorsel["kesinlik"] = "kesin"
        gorsel = gorsel[gorsel["rel_a"].isin(r2i) & gorsel["rel_b"].isin(r2i)].copy()
    gv = gorsel.assign(gi=gorsel["rel_a"].map(r2i), gj=gorsel["rel_b"].map(r2i))

    # --- aday çiftler: katı-grup çifti başına, iki grubun eşleştirme üyeleri arasında
    Sg = grup_benzerligi(S_r, kg_r)
    Gn = len(Sg)
    aday = {}

    def ekle(i, j, kaynak):
        gi, gj = kg_r[i], kg_r[j]
        if gi == gj:
            return
        key = (gi, gj) if gi < gj else (gj, gi)
        if key in aday:
            aday[key].add(kaynak)
        else:
            aday[key] = {kaynak}

    k = min(args.aday_knn, Gn - 1)
    idx = np.argpartition(-Sg, k - 1, axis=1)[:, :k]
    for a in range(Gn):
        for b in idx[a]:
            ekle(int(kg_uye[a]), int(kg_uye[int(b)]), "dino")
    # önce/sonra temizlik çekimleri: Healthy ↔ Surface_Obstruction çapraz etiketli en yakın komşular (katı-grup düzeyi)
    g_lab = pd.Series(lab_r).groupby(kg_r).agg(lambda s_: set(s_)).to_dict()
    h_m = np.array(["Healthy" in g_lab[g] for g in range(Gn)])
    so_m = np.array(["Surface_Obstruction" in g_lab[g] for g in range(Gn)])
    for k_m, h_m2 in ((h_m, so_m), (so_m, h_m)):
        kaynak_idx, hedef_idx = np.flatnonzero(k_m), np.flatnonzero(h_m2)
        kk = min(args.aday_capraz, len(hedef_idx) - 1)
        top = np.argpartition(-Sg[np.ix_(kaynak_idx, hedef_idx)], kk - 1, axis=1)[:, :kk]
        for x, a in enumerate(kaynak_idx):
            for y in top[x]:
                ekle(int(kg_uye[a]), int(kg_uye[hedef_idx[y]]), "once_sonra")
    # Kaggle önce/sonra numara farkı pencereleri (temiz N − kirli M), yalnız aday
    pencereler = [tuple(int(x) for x in w.split("-")) for w in args.aday_ofset.split(",") if w.strip()]
    ci, di = np.flatnonzero(fam_r == "Imgclean"), np.flatnonzero(fam_r == "Imgdirty")
    fark = no_r[ci][:, None] - no_r[di][None, :]
    pen = np.zeros(fark.shape, bool)
    for lo, hi in pencereler:
        pen |= (fark >= lo) & (fark <= hi)
    pen &= S_r[np.ix_(ci, di)] >= args.aday_ofset_tau
    for x, y in np.argwhere(pen):
        ekle(int(ci[x]), int(di[y]), "ofset")
    for i, j in np.argwhere(np.triu(D_r <= args.aday_phash, 1)):
        ekle(int(i), int(j), "phash")
    az = np.flatnonzero(np.isin(lab_r, AZINLIK))
    for x in range(len(az)):
        for y in range(x + 1, len(az)):
            ekle(int(az[x]), int(az[y]), "azinlik")
    for i, j in dosyano_ciftleri(args.aday_no, NO_AILELERI):
        ekle(int(i), int(j), "dosyano")
    for i, j in exif_ciftleri(86400):
        ekle(int(i), int(j), "exif")
    for r in gv.itertuples():
        if r.gi in rgb_yerel and r.gj in rgb_yerel:
            ekle(rgb_yerel[r.gi], rgb_yerel[r.gj], "gorsel")

    # --- eşleştirme: çift bazında artımlı önbellek (anahtar: kanonik rel_i < rel_j)
    import cv2 as _cv2
    kod_ozeti = hashlib.md5("".join(inspect.getsource(f) for f in (ozellik_isci, geometri, ncc_olc, _nbulanik, eslestir,
                                                                     kapsamli_tarama, _kapsam, _disbukey)).encode()).hexdigest()
    param = dict(eslestirici=ESLESTIRICI, kod=kod_ozeti, uzun=UZUN, nkp=NKP, nkp_tarama=NKP_TARAMA, uzun_ncc=UZUN_NCC,
                 altbilgi=ALTBILGI, magsac_px=MAGSAC_PX, kimlik_ncc=KIMLIK_NCC, opencv=_cv2.__version__, dosyalar=imza)
    mcsv, mjson = onb / "eslesmeler.csv", onb / "eslesmeler_param.json"
    onceki = None
    if mcsv.exists() and mjson.exists() and json.loads(mjson.read_text(encoding="utf-8")) == param:
        onceki = pd.read_csv(mcsv)
    tarama_csv = onb / "tarama.npz"
    tarama_ok = tarama_csv.exists() and onceki is not None
    F = None
    if tarama_ok:
        z = np.load(tarama_csv, allow_pickle=True)
        tarama_ok = list(z["rel"]) == list(rel[rgb])
        if tarama_ok:
            r70, r80 = z["r70"], z["r80"]
    if not tarama_ok:
        import torch
        cihaz = "cuda" if torch.cuda.is_available() else "cpu"
        t1 = time.time()
        with Pool(args.isci) as havuz:
            sonuc = havuz.map(ozellik_isci, [yollar[i] for i in rgb], chunksize=8)
        F = sonuc
        del sonuc
        print(f"özellikler: {n_rgb} görüntü, {time.time() - t1:.0f} sn", flush=True)
        t1 = time.time()
        r70, r80 = kapsamli_tarama(F, cihaz)
        np.savez_compressed(tarama_csv, r70=r70, r80=r80, rel=rel[rgb])
        print(f"kapsamlı tarama: {time.time() - t1:.0f} sn", flush=True)
    for i, j in np.argwhere(np.triu((r70 >= args.tarama_r70) | (r80 >= args.tarama_r80), 1)):
        ekle(int(i), int(j), "tarama")

    ciftler = []
    kaynaklar = []
    for (ga, gb), kay in aday.items():
        i, j = int(kg_uye[ga]), int(kg_uye[gb])
        if rel[rgb[i]] > rel[rgb[j]]:
            i, j = j, i
        ciftler.append((i, j))
        kaynaklar.append("+".join(sorted(kay)))
    ad = pd.DataFrame({"rel_i": rel[rgb[[c[0] for c in ciftler]]], "rel_j": rel[rgb[[c[1] for c in ciftler]]],
                       "kaynaklar": kaynaklar})
    print(f"aday çift (katı-grup çifti): {len(ad)}; kaynak dağılımı: "
          f"{ {k_: int(ad['kaynaklar'].str.contains(k_).sum()) for k_ in ('dino', 'once_sonra', 'ofset', 'phash', 'tarama', 'azinlik', 'dosyano', 'exif', 'gorsel')} }",
          flush=True)
    if onceki is not None:
        bilinen = set(zip(onceki["rel_i"], onceki["rel_j"]))
        eksik = [c for c, a_, b_ in zip(ciftler, ad["rel_i"], ad["rel_j"]) if (a_, b_) not in bilinen]
    else:
        eksik = ciftler
    if eksik:
        import torch
        cihaz = "cuda" if torch.cuda.is_available() else "cpu"
        if F is None:
            t1 = time.time()
            with Pool(args.isci) as havuz:
                sonuc = havuz.map(ozellik_isci, [yollar[i] for i in rgb], chunksize=8)
            F = sonuc
            del sonuc
            print(f"özellikler: {n_rgb} görüntü, {time.time() - t1:.0f} sn", flush=True)
        t1 = time.time()
        yeni = pd.DataFrame(eslestir(eksik, F, cihaz))
        print(f"eşleştirme: {len(eksik)} yeni çift, {time.time() - t1:.0f} sn", flush=True)
        yeni["rel_i"], yeni["rel_j"] = rel[rgb[yeni["i"]]], rel[rgb[yeni["j"]]]
        yeni = yeni.drop(columns=["i", "j"])
        tum = pd.concat([onceki, yeni], ignore_index=True) if onceki is not None else yeni
        tum = tum.sort_values(["rel_i", "rel_j"]).reset_index(drop=True)
        tum.to_csv(mcsv, index=False, encoding="utf-8-sig")
        mjson.write_text(json.dumps(param, ensure_ascii=False, indent=1), encoding="utf-8")
        onceki = tum
    # --- ters yön (B → A): aynı aday çiftleri, ayrı önbellek; kenar kararları iki yönün VEYA'sı
    tcsv, tjson = onb / "eslesmeler_ters.csv", onb / "eslesmeler_ters_param.json"
    tparam = dict(param, yon="ters")
    tonceki = None
    if tcsv.exists() and tjson.exists() and json.loads(tjson.read_text(encoding="utf-8")) == tparam:
        tonceki = pd.read_csv(tcsv)
    if tonceki is not None:
        bilinen = set(zip(tonceki["rel_i"], tonceki["rel_j"]))
        teksik = [c for c, a_, b_ in zip(ciftler, ad["rel_i"], ad["rel_j"]) if (a_, b_) not in bilinen]
    else:
        teksik = ciftler
    if teksik:
        import torch
        cihaz = "cuda" if torch.cuda.is_available() else "cpu"
        if F is None:
            t1 = time.time()
            with Pool(args.isci) as havuz:
                sonuc = havuz.map(ozellik_isci, [yollar[i] for i in rgb], chunksize=8)
            F = sonuc
            del sonuc
            print(f"özellikler: {n_rgb} görüntü, {time.time() - t1:.0f} sn", flush=True)
        t1 = time.time()
        yeni = pd.DataFrame(eslestir([(j, i) for i, j in teksik], F, cihaz))
        print(f"ters eşleştirme: {len(teksik)} yeni çift, {time.time() - t1:.0f} sn", flush=True)
        # eslestir A=j, B=i ile çağrıldı: kanonik (i, j) sırasına ve kanonik kapsam sütunlarına çevir
        yeni = yeni.rename(columns={"i": "j", "j": "i", "kap_i": "kap_j", "kap_j": "kap_i"})
        yeni["rel_i"], yeni["rel_j"] = rel[rgb[yeni["i"]]], rel[rgb[yeni["j"]]]
        yeni = yeni.drop(columns=["i", "j"])
        tum = pd.concat([tonceki, yeni], ignore_index=True) if tonceki is not None else yeni
        tum = tum.sort_values(["rel_i", "rel_j"]).reset_index(drop=True)
        tum.to_csv(tcsv, index=False, encoding="utf-8-sig")
        tjson.write_text(json.dumps(tparam, ensure_ascii=False, indent=1), encoding="utf-8")
        tonceki = tum
    del F
    es = ad.merge(onceki.drop(columns=[c for c in ("kaynaklar",) if c in onceki.columns]), on=["rel_i", "rel_j"],
                  how="left")
    assert es["inl"].notna().all(), "eşleşmesi olmayan aday kaldı"
    est = ad.merge(tonceki.drop(columns=[c for c in ("kaynaklar",) if c in tonceki.columns]), on=["rel_i", "rel_j"],
                   how="left")
    assert est["inl"].notna().all(), "ters eşleşmesi olmayan aday kaldı"
    es["gi"] = es["rel_i"].map(r2i)
    es["gj"] = es["rel_j"].map(r2i)
    es["r70"] = r70[es["gi"].map(rgb_yerel), es["gj"].map(rgb_yerel)]
    es["r80"] = r80[es["gi"].map(rgb_yerel), es["gj"].map(rgb_yerel)]
    es["S"] = S[es["gi"], es["gj"]]
    es["phash_d"] = D[es["gi"], es["gj"]]
    pxl = m["piksel"].to_numpy()
    kucuk_i = pxl[es["gi"]] <= pxl[es["gj"]]  # gerçek piksel alanı küçük olan görüntü (eşitlikte i)
    for x in (es, est):
        for c in ("gecerli", "ayna"):
            x[c] = x[c].astype(bool)
        x["kap_kucuk"] = np.where(kucuk_i, x["kap_i"], x["kap_j"])
        x["S"] = es["S"].to_numpy()

    def bayraklar(x):
        alt = (x["alt_pay"] >= 0.8).to_numpy()
        # filigran/şablon: aynı çerçevede çakışan ama içeriği farklı çift (stok filigranı, reklam şablonu). Aynı
        # noktadan çekilmiş önce/sonra fotoğrafları da bu imzayı verir; onları ayırmak için veto yalnız eşleşme
        # altbilgide yoğunsa, inlier'lar küçük bir bölgedeyse ya da görüntüler genel olarak benzemiyorsa uygulanır
        fil = ((x["kimlik"] < args.filigran_kimlik) & ~(x["ncc_hp"] >= args.filigran_ncc)
               & ((x["alt_pay"] >= args.filigran_altpay) | (x["kap_kucuk"] < args.filigran_kapsam)
                  | (x["S"] < args.filigran_s))).to_numpy()
        return alt, fil

    es["altbilgi"], es["filigran"] = bayraklar(es)
    est["altbilgi"], est["filigran"] = bayraklar(est)

    # --- zorunlu ayrıklık (görsel kararlar): katı-grup çifti anahtarları
    kgm = m["kati_grup"].to_numpy().astype(np.int64)

    def gkey(a, b):
        ka, kb = kgm[np.asarray(a)], kgm[np.asarray(b)]
        return np.minimum(ka, kb) * 1_000_000 + np.maximum(ka, kb)

    es["gkey"] = gkey(es["gi"], es["gj"])
    gv["gkey"] = gkey(gv["gi"], gv["gj"]) if len(gv) else np.zeros(0, np.int64)
    ilgisiz_k = set(gv.loc[(gv["sinif"] == "ilgisiz") & (gv["kesinlik"] == "kesin"), "gkey"])
    foto_yasak = ilgisiz_k | set(gv.loc[gv["sinif"] == "ayni_sahne", "gkey"])
    yasak_foto = es["gkey"].isin(foto_yasak)
    az_cift = np.isin(etiket[es["gi"]], AZINLIK) | np.isin(etiket[es["gj"]], AZINLIK)
    capraz = etiket[es["gi"]] != etiket[es["gj"]]
    es["foto_esik"] = np.where(capraz, args.foto_ncc_capraz, args.foto_ncc)
    sahne_inl = np.where(az_cift, args.azinlik_inlier, args.sahne_inlier)
    sahne_ayr = np.where(az_cift, args.azinlik_inlier, args.sahne_ayrik)
    # kesin "ilgisiz" görsel kararı geometrik sahne kenarını da engeller (doku/şablon zincirleri)
    yasak_sahne = es["gkey"].isin(ilgisiz_k).to_numpy() if not args.ilgisiz_sahne_vetosu_yok \
        else np.zeros(len(es), bool)

    def kenar_kurallari(x):
        foto = (x["gecerli"] & (x["inl"] >= args.foto_inlier) & (x["umin"] >= 8)
                & (x["kap_kucuk"] >= args.foto_kapsam) & (x["ncc_hp"] >= es["foto_esik"]) & ~yasak_foto).to_numpy()
        # aynı stok fotoğrafın farklı filigranlı kopyaları (NCC filigran yüzünden düşük): yalnız aynı etiketlilerde
        foto_fil = (~capraz & x["gecerli"] & (x["inl"] >= args.foto_filigranli_inlier)
                    & (x["kap_kucuk"] >= args.foto_filigranli_kapsam) & (x["ncc_hp"] >= args.foto_filigranli_ncc)
                    & ~yasak_foto).to_numpy()
        supheli = (capraz & x["gecerli"] & (x["inl"] >= args.foto_inlier) & (x["umin"] >= 8)
                   & (x["kap_kucuk"] >= args.foto_kapsam) & (x["ncc_hp"] >= args.foto_ncc)).to_numpy()
        geo = (x["gecerli"] & (x["inl"] >= sahne_inl) & (x["umin"] >= sahne_ayr)).to_numpy() & ~x["altbilgi"] \
            & ~x["filigran"]
        return foto, foto_fil, supheli, geo

    foto_f, fil_f, sup_f, geo_f = kenar_kurallari(es)
    foto_t, fil_t, sup_t, geo_t = kenar_kurallari(est)
    for c in ("inl", "umin", "gecerli", "alt_pay", "kimlik", "kap_kucuk", "ncc_hp", "altbilgi", "filigran", "ayna"):
        es[f"t_{c}"] = est[c].to_numpy()
    es["foto_kenari"] = foto_f | foto_t | fil_f | fil_t
    es["foto_filigranli"] = (fil_f | fil_t) & ~(foto_f | foto_t)
    es["celiski_suphesi"] = (sup_f | sup_t) & ~es["foto_kenari"]
    es["sahne_kenari"] = (geo_f | geo_t) & ~yasak_sahne
    es["sahne_yon"] = np.select([geo_f & geo_t, geo_f, geo_t], ["iki", "ileri", "ters"], "")
    es["ilgisiz_vetosu"] = (geo_f | geo_t) & yasak_sahne

    # --- kenarlar
    gi_g = lambda d: d[["gi", "gj"]].to_numpy()  # noqa: E731
    e_foto_geo = gi_g(es[es["foto_kenari"]])
    e_foto_gorsel = gi_g(gv[gv["sinif"] == "ayni_foto"])
    e_sahne_geo = gi_g(es[es["sahne_kenari"]])
    az_gv = np.isin(etiket[gv["gi"].to_numpy()], AZINLIK) & np.isin(etiket[gv["gj"].to_numpy()], AZINLIK) if len(gv) \
        else np.zeros(0, bool)
    belirsiz_m = (gv["sinif"] == "belirsiz") & (az_gv if args.belirsiz_yalniz_azinlik else True)
    e_sahne_gorsel = gi_g(gv[(gv["sinif"] == "ayni_sahne") | belirsiz_m])

    e_dino = np.array([(rgb[kg_uye[a]], rgb[kg_uye[b]]) for a, b in karsilikli_knn(Sg, args.dino_k, args.dino_tau)],
                      dtype=np.int64).reshape(-1, 2)
    e_exif = np.array([(rgb[a], rgb[b]) for a, b in exif_ciftleri(args.exif_dk * 60)], dtype=np.int64).reshape(-1, 2)
    e_dosyano = np.array([(rgb[a], rgb[b]) for a, b in dosyano_ciftleri(args.dosyano_fark)
                          if S_r[a, b] >= args.dosyano_tau], dtype=np.int64).reshape(-1, 2) if args.dosyano_kenar         else np.zeros((0, 2), np.int64)

    m["foto_grup"] = bilesenler(N, [e_kati, e_foto_geo, e_foto_gorsel])
    m["sahne_grup"] = bilesenler(N, [e_kati, e_foto_geo, e_foto_gorsel, e_sahne_geo, e_dino, e_exif, e_dosyano,
                                     e_sahne_gorsel])
    sahne_oto = bilesenler(N, [e_kati, e_foto_geo, e_sahne_geo, e_dino, e_exif, e_dosyano])
    kenar_sayisi = {"kati(md5∪phash)": len(e_kati), "foto_geometrik": len(e_foto_geo),
                    "foto_gorsel": len(e_foto_gorsel), "sahne_geometrik": len(e_sahne_geo), "dino_mknn": len(e_dino),
                    "exif": len(e_exif), "dosyano": len(e_dosyano), "sahne_gorsel": len(e_sahne_gorsel),
                    "foto_yasagi_ile_dusen": int((yasak_foto & es["gecerli"] & (es["inl"] >= args.foto_inlier)
                                                  & (es["ncc_hp"] >= args.foto_ncc)).sum()),
                    "foto_filigranli": int(es["foto_filigranli"].sum()),
                    "sahne_geo_yalniz_ters_yon": int((es["sahne_yon"] == "ters").sum()),
                    "sahne_geo_yalniz_ileri_yon": int((es["sahne_yon"] == "ileri").sum()),
                    "ilgisiz_vetosu_ile_dusen_sahne": int(es["ilgisiz_vetosu"].sum())}
    print("kenar sayıları:", kenar_sayisi, flush=True)
    # zincirle ihlal: aynı foto grubuna düşmüş "aynı sahne"/"ilgisiz" çiftleri (aynı fotoğraf sayılmamalı; elle bakılmalı)
    fgm = m["foto_grup"].to_numpy()
    sgm = m["sahne_grup"].to_numpy()
    ihlal = gv[gv["sinif"].isin(["ayni_sahne", "ilgisiz"]) & (fgm[gv["gi"]] == fgm[gv["gj"]])
               & (kgm[gv["gi"]] != kgm[gv["gj"]])] if len(gv) else gv

    # --- bayraklar
    fg = m.groupby("foto_grup")
    m["foto_grup_boyutu"] = fg["path"].transform("size")
    m["celiskili"] = fg["label"].transform("nunique") > 1
    m["celiski_etiketleri"] = np.where(m["celiskili"], fg["label"].transform(lambda s: "+".join(sorted(set(s)))), "")
    m["sahne_grup_boyutu"] = m.groupby("sahne_grup")["path"].transform("size")
    m["uygun"] = ~m["termal"] & ~m["celiskili"]
    sira = m[m["uygun"]].sort_values(["foto_grup", "piksel", "_uz", "rel_path"], ascending=[True, False, True, True])
    m["temsilci"] = False
    m.loc[sira.groupby("foto_grup").head(1).index, "temsilci"] = True

    # --- katlar
    from sklearn.model_selection import StratifiedGroupKFold
    v1 = m[m["temsilci"]]
    y = v1["label"] if args.tabaka == "etiket" else v1["label"] + "|" + v1["kaynak"]

    def katla(tohum):
        sgkf = StratifiedGroupKFold(n_splits=args.kat, shuffle=True, random_state=tohum)
        g2k = {}
        for kk, (_, te) in enumerate(sgkf.split(v1, y, v1["sahne_grup"])):
            for g in v1["sahne_grup"].iloc[te].unique():
                g2k[int(g)] = kk
        k1 = np.where(m["temsilci"], m["sahne_grup"].map(g2k).fillna(-1), -1).astype(int)
        k2 = np.where(m["uygun"], m["sahne_grup"].map(g2k).fillna(-1), -1).astype(int)
        return k1, k2

    m["kat_v1"], m["kat_v2"] = katla(args.tohum)
    tekrar = [int(t) for t in args.tekrar_tohumlari.split(",") if t.strip()]
    for t in tekrar:
        m[f"kat_v1_t{t}"] = katla(t)[0]
    eksik_kat = int(((m["kat_v2"] == -1) & m["uygun"]).sum())
    assert eksik_kat == 0, f"uygun ama katsız satır: {eksik_kat}"
    assert m[m["kat_v2"] >= 0].groupby("sahne_grup")["kat_v2"].nunique().max() == 1

    # --- sızıntı tanısı (temsilci düzeyinde): her dosyayı kendi foto grubunun v1 temsilcisine eşle
    temsil_of_fg = m[m["temsilci"]].set_index("foto_grup")["kat_v1"]
    rep_idx_of_fg = pd.Series(m.index[m["temsilci"]], index=m.loc[m["temsilci"], "foto_grup"])
    f_rep = m["foto_grup"].map(temsil_of_fg).fillna(-1).astype(int).to_numpy()  # dosya → temsilcinin v1 katı
    rep_of = m["foto_grup"].map(rep_idx_of_fg).fillna(-1).astype(int).to_numpy()
    kat = {"kat_v1": f_rep, "kat_v2": m["kat_v2"].to_numpy()}

    def kat_arasi(kol):
        f = kat[kol]
        idx = np.flatnonzero(m[kol].to_numpy() >= 0)  # S/D tanısı satırlar üzerinde (v1 satırları zaten temsilci)
        fk = f[idx]
        fark = np.triu(fk[:, None] != fk[None, :], 1)
        Dk, Sk, Ek = D[np.ix_(idx, idx)], S[np.ix_(idx, idx)], esit[np.ix_(idx, idx)]
        r = dict(varyant=kol, n=len(idx), md5_ayni=int((Ek & fark).sum()))
        for e in (4, 6, 8, 10, 12):
            r[f"phash_le{e}"] = int(((Dk <= e) & fark).sum())
        for t in (0.90, 0.93, 0.95, 0.97):
            r[f"dino_ge{t}"] = int(((Sk >= t) & fark).sum())
        fi, fj = f[es["gi"]], f[es["gj"]]
        capraz = (fi >= 0) & (fj >= 0) & (fi != fj)
        r["foto_kenari"] = int((capraz & es["foto_kenari"]).sum())
        r["sahne_kenari"] = int((capraz & es["sahne_kenari"]).sum())
        for a_, b_ in ((12, 10), (10, 8), (8, 8)):  # kenar olmamış geometrik eşleşmeler (vetolu ya da eşik altı)
            z = es["gecerli"] & (es["inl"] >= a_) & (es["umin"] >= b_) & ~es["sahne_kenari"] & ~es["foto_kenari"]
            r[f"kenarsiz_geo_inl{a_}_ayrik{b_}"] = int((capraz & z).sum())
        z = es["gecerli"] & (es["inl"] >= 10) & (es["umin"] >= 10) & ~es["sahne_kenari"] & ~es["foto_kenari"]
        r["kenarsiz_veto_altbilgi"] = int((capraz & z & es["altbilgi"]).sum())
        r["kenarsiz_veto_filigran"] = int((capraz & z & es["filigran"] & ~es["altbilgi"]).sum())
        r["kenarsiz_veto_ilgisiz"] = int((capraz & z & es["ilgisiz_vetosu"]).sum())
        for F_, ad_ in ((("Imgclean", "Imgdirty"), "kaggle"), (("Clean", "Bird", "Dust", "Snow", "Electrical", "Physical"), "pv")):
            dn = np.array([(rgb[a], rgb[b]) for a, b in dosyano_ciftleri(args.dosyano_fark, F_)], dtype=np.int64).reshape(-1, 2)
            if len(dn):
                r[f"dosyano_le{args.dosyano_fark}_{ad_}"] = int(((f[dn[:, 0]] >= 0) & (f[dn[:, 1]] >= 0)
                                                                 & (f[dn[:, 0]] != f[dn[:, 1]])).sum())
        eg = np.array([(rgb[a], rgb[b]) for a, b in exif_ciftleri(86400)], dtype=np.int64).reshape(-1, 2)
        r["exif_ayni_model_24sa"] = int(((f[eg[:, 0]] >= 0) & (f[eg[:, 1]] >= 0) & (f[eg[:, 0]] != f[eg[:, 1]])).sum()) \
            if len(eg) else 0
        for s_ in ("ayni_foto", "ayni_sahne", "belirsiz"):
            q = gv[gv["sinif"] == s_]
            a_, b_ = f[q["gi"].to_numpy()], f[q["gj"].to_numpy()]
            r[f"gorsel_{s_}"] = int(((a_ >= 0) & (b_ >= 0) & (a_ != b_)).sum())
        return r

    sizinti = pd.DataFrame([kat_arasi("kat_v1"), kat_arasi("kat_v2")])

    # yakın kaçak listesi (temsilci düzeyinde, katlar arası, görsel denetim için)
    fi, fj = f_rep[es["gi"]], f_rep[es["gj"]]
    yk = es[(fi >= 0) & (fj >= 0) & (fi != fj)].copy()
    yk = yk[(yk["gecerli"] & (yk["inl"] >= 8)) | (yk["S"] >= 0.90) | (yk["phash_d"] <= 12)
            | yk["kaynaklar"].str.contains("dosyano|exif")]
    yk["temsilci_i"], yk["temsilci_j"] = rel[rep_of[yk["gi"]]], rel[rep_of[yk["gj"]]]
    yk["kat_i"], yk["kat_j"] = f_rep[yk["gi"]], f_rep[yk["gj"]]
    yk["label_i"], yk["label_j"] = etiket[yk["gi"]], etiket[yk["gj"]]
    yk = yk.sort_values(["inl", "S"], ascending=False).drop_duplicates(["temsilci_i", "temsilci_j"])

    # --- kalibrasyon: görsel etiketli (kesin) çiftlerde kural başarımı (eşleşme tablosu katı-grup çifti başına)
    gv["ayni_kati"] = kgm[gv["gi"].to_numpy()] == kgm[gv["gj"].to_numpy()] if len(gv) else []
    termal_arr = m["termal"].to_numpy()
    gv["termal_cift"] = termal_arr[gv["gi"].to_numpy()] | termal_arr[gv["gj"].to_numpy()] if len(gv) else []
    gk = gv[~gv["ayni_kati"] & ~gv["termal_cift"] & (gv["kesinlik"] == "kesin")].merge(
        es.drop(columns=["gi", "gj", "rel_i", "rel_j"]).drop_duplicates("gkey"), on="gkey", how="left")
    eslesmemis = int(gk["inl"].isna().sum())
    gk = gk[gk["inl"].notna()].copy()
    for c in ("gecerli", "altbilgi", "filigran", "foto_kenari", "sahne_kenari"):
        gk[c] = gk[c].astype(bool)
    poz = gk["sinif"].isin(["ayni_foto", "ayni_sahne"])
    neg = gk["sinif"] == "ilgisiz"
    foto_poz = gk["sinif"] == "ayni_foto"
    foto_neg = gk["sinif"].isin(["ayni_sahne", "ilgisiz"])
    kal = []

    def kural_ekle(ad_, k_, foto=False):
        k_ = pd.Series(k_, index=gk.index).fillna(False).astype(bool)
        p, n_ = (foto_poz, foto_neg) if foto else (poz, neg)
        kal.append(dict(kural=ad_, tur="foto" if foto else "sahne", poz=int(p.sum()), poz_yakalanan=int((k_ & p).sum()),
                        neg=int(n_.sum()), neg_kabul=int((k_ & n_).sum())))

    gg = gk["gecerli"] & ~gk["altbilgi"] & ~gk["filigran"].astype(bool)
    for a_, b_ in ((20, 20), (15, 15), (12, 12), (10, 10), (8, 8)):
        kural_ekle(f"sahne_geo inl≥{a_}, ayrık≥{b_} (altbilgi/filigran değil)", gg & (gk["inl"] >= a_) & (gk["umin"] >= b_))
    kural_ekle("sahne_geo inl≥12, ayrık≥12 (altbilgi/filigran süzgeci YOK)", gk["gecerli"] & (gk["inl"] >= 12) & (gk["umin"] >= 12))
    kural_ekle("sahne_geo (kullanılan: iki yön VEYA, daraltılmış filigran, ilgisiz vetosu)", gk["sahne_kenari"])
    kural_ekle(f"DINOv2 S≥{args.dino_tau} (dosya düzeyi)", gk["S"] >= args.dino_tau)
    kural_ekle("ham inl≥30 (geçerlilik/ayrıklık yok; karşılaştırma)", gk["inl"] >= 30)
    for t in (0.80, 0.85, 0.90):
        kural_ekle(f"foto NCC_hp≥{t}, inl≥{args.foto_inlier}, kapsam≥{args.foto_kapsam}",
                   gk["gecerli"] & (gk["inl"] >= args.foto_inlier) & (gk["umin"] >= 8)
                   & (gk["kap_kucuk"] >= args.foto_kapsam) & (gk["ncc_hp"] >= t), foto=True)
    kural_ekle("foto (kullanılan)", gk["foto_kenari"], foto=True)
    kal = pd.DataFrame(kal)
    kal_not = (f"Etiketli {len(gv)} çiftin {int(gv['ayni_kati'].sum()) if len(gv) else 0}'i aynı katı grupta, "
               f"{int((gv['termal_cift'] & ~gv['ayni_kati']).sum()) if len(gv) else 0}'i termal; kalibrasyona yalnız "
               f"'kesin' kararlar girer ({len(gk)} çift); eşleşme tablosunda olmayan: {eslesmemis} (0 olmalı).")
    oto = pd.DataFrame(dict(sinif=gv["sinif"], kesinlik=gv["kesinlik"],
                            ayni=sahne_oto[gv["gi"]] == sahne_oto[gv["gj"]])) if len(gv) else pd.DataFrame()
    oto_ozet = oto.groupby(["sinif", "kesinlik"])["ayni"].agg(["sum", "size"]).rename(
        columns={"sum": "ayni_sahne_grubunda", "size": "cift"}) if len(oto) else pd.DataFrame()

    # --- zincir tanısı: en büyük sahne gruplarını hangi kenar türleri tutuyor
    ken = []
    for tur, e in (("kati", e_kati), ("foto_geometrik", e_foto_geo), ("foto_gorsel", e_foto_gorsel),
                   ("sahne_geometrik", e_sahne_geo), ("dino_mknn", e_dino), ("exif", e_exif), ("dosyano", e_dosyano),
                   ("sahne_gorsel", e_sahne_gorsel)):
        if len(e):
            ken.append(pd.DataFrame({"tur": tur, "gi": e[:, 0], "gj": e[:, 1]}))
    ken = pd.concat(ken, ignore_index=True)
    ken["rel_i"], ken["rel_j"] = rel[ken["gi"]], rel[ken["gj"]]
    ken["S"], ken["phash_d"] = S[ken["gi"], ken["gj"]], D[ken["gi"], ken["gj"]]
    ken["sahne_grup"] = sgm[ken["gi"]]
    buyuk = m.loc[m["temsilci"], "sahne_grup"].value_counts().head(5)
    zincir = []
    for g, n_ in buyuk.items():
        d = m[(m["sahne_grup"] == g) & ~m["termal"]]
        zincir.append(dict(sahne_grup=int(g), v1=int(n_), rgb=len(d), kat=int(m.loc[(m["sahne_grup"] == g) & m["temsilci"], "kat_v1"].iloc[0]),
                           aile=d["aile"].value_counts().to_dict(), etiket=d["label"].value_counts().to_dict(),
                           kenar=ken[ken["sahne_grup"] == g]["tur"].value_counts().to_dict()))
    zincir = pd.DataFrame(zincir)

    # --- tablolar
    def kat_tablosu(kol, sutun="label"):
        q = m[m[kol] >= 0]
        t = pd.crosstab(q[kol], q[sutun])
        t["toplam"] = t.sum(axis=1)
        return t

    tv1, tv2, tk1 = kat_tablosu("kat_v1"), kat_tablosu("kat_v2"), kat_tablosu("kat_v1", "kaynak")
    rgb_maske = ~m["termal"].to_numpy()
    v1m = m["temsilci"].to_numpy()

    # --- yaz
    sutunlar = (["path", "rel_path", "dosya", "split", "label", "aile", "kaynak", "termal", "md5", "width", "height",
                 "kati_grup", "foto_grup", "foto_grup_boyutu", "sahne_grup", "sahne_grup_boyutu", "celiskili",
                 "celiski_etiketleri", "uygun", "temsilci", "kat_v1", "kat_v2"] + [f"kat_v1_t{t}" for t in tekrar])
    m[sutunlar].to_csv(cikti / "manifest_asama2.csv", index=False, encoding="utf-8-sig")
    ken.drop(columns=["gi", "gj"]).to_csv(cikti / "kenarlar.csv", index=False, encoding="utf-8-sig")
    es.drop(columns=["gi", "gj", "gkey"]).to_csv(cikti / "aday_eslesmeleri.csv", index=False, encoding="utf-8-sig")
    cel = m[m["celiskili"]].sort_values(["foto_grup", "rel_path"])[
        ["foto_grup", "kati_grup", "rel_path", "label", "celiski_etiketleri", "termal", "kaynak", "md5"]]
    cel.to_csv(cikti / "celiskili_gruplar.csv", index=False, encoding="utf-8-sig")
    ck = []
    for tur, e in (("kati", e_kati), ("foto_geometrik", e_foto_geo), ("foto_gorsel", e_foto_gorsel)):
        e = e[etiket[e[:, 0]] != etiket[e[:, 1]]] if len(e) else e
        for a, b in e:
            ck.append(dict(tur=tur, foto_grup=int(fgm[a]), termal=bool(m.at[a, "termal"]), rel_i=rel[a], rel_j=rel[b],
                           label_i=etiket[a], label_j=etiket[b], md5_ayni=bool(esit[a, b]), phash_d=int(D[a, b]),
                           S=float(S[a, b])))
    ck = pd.DataFrame(ck)
    if len(ck):
        ek = es[["gkey", "inl", "umin", "ncc_hp", "kap_kucuk", "ayna"]].drop_duplicates("gkey")
        ck["gkey"] = gkey(ck["rel_i"].map(r2i), ck["rel_j"].map(r2i))
        ck = ck.merge(ek, on="gkey", how="left").drop(columns="gkey")
        ck.sort_values(["termal", "foto_grup", "tur"]).to_csv(cikti / "celiski_kenarlari.csv", index=False,
                                                              encoding="utf-8-sig")
    es[es["celiski_suphesi"]].drop(columns=["gi", "gj", "gkey"]).assign(
        label_i=lambda d: d["rel_i"].map(dict(zip(rel, etiket))), label_j=lambda d: d["rel_j"].map(dict(zip(rel, etiket)))
    ).to_csv(cikti / "celiski_supheleri.csv", index=False, encoding="utf-8-sig")
    ihlal.drop(columns=[c for c in ("gi", "gj", "gkey") if c in ihlal.columns]).to_csv(
        cikti / "zorunlu_ayrik_ihlalleri.csv", index=False, encoding="utf-8-sig")
    sizinti.to_csv(cikti / "kat_arasi_sizinti.csv", index=False, encoding="utf-8-sig")
    yk.drop(columns=["gi", "gj", "gkey"]).to_csv(cikti / "kat_arasi_yakin_adaylar.csv", index=False, encoding="utf-8-sig")
    kal.to_csv(cikti / "kalibrasyon.csv", index=False, encoding="utf-8-sig")
    zincir.to_csv(cikti / "buyuk_gruplar.csv", index=False, encoding="utf-8-sig")
    tv1.to_csv(cikti / "kat_sinif_v1.csv", encoding="utf-8-sig")
    tv2.to_csv(cikti / "kat_sinif_v2.csv", encoding="utf-8-sig")
    tk1.to_csv(cikti / "kat_kaynak_v1.csv", encoding="utf-8-sig")
    tekrar_tab = {f"t{t}": pd.crosstab(m.loc[m[f"kat_v1_t{t}"] >= 0, f"kat_v1_t{t}"],
                                       m.loc[m[f"kat_v1_t{t}"] >= 0, "label"]).to_dict(orient="list") for t in tekrar}

    ozet = dict(
        surum="r4.1", dosya=N, md5_tekil=int(m["md5"].nunique()), rgb_dosya=int(rgb_maske.sum()),
        termal_dosya=int(m["termal"].sum()),
        kati_grup_rgb=grup_ozeti(m["kati_grup"].to_numpy(), etiket, rgb_maske),
        foto_grup_rgb=grup_ozeti(m["foto_grup"].to_numpy(), etiket, rgb_maske),
        sahne_grup_rgb=grup_ozeti(m["sahne_grup"].to_numpy(), etiket, rgb_maske),
        sahne_grup_v1=grup_ozeti(m["sahne_grup"].to_numpy(), etiket, v1m),
        kenar_sayisi=kenar_sayisi, aday_cift=len(es), zorunlu_ayrik_ihlali=len(ihlal),
        celiski_suphesi_cift=int(es["celiski_suphesi"].sum()),
        celiskili_foto_grup=int(m.loc[m["celiskili"], "foto_grup"].nunique()),
        celiskili_dosya=int(m["celiskili"].sum()),
        celiskili_rgb_dosya=int((m["celiskili"] & ~m["termal"]).sum()),
        uygun_dosya=int(m["uygun"].sum()),
        v1_ornek=int(v1m.sum()), v2_ornek=int((m["kat_v2"] >= 0).sum()),
        v1_sinif=m.loc[v1m, "label"].value_counts().to_dict(),
        v2_sinif=m.loc[m["kat_v2"] >= 0, "label"].value_counts().to_dict(),
        v1_sahne_sayisi_sinif=m[v1m].groupby("label")["sahne_grup"].nunique().to_dict(),
        tabaka=args.tabaka, tohum=args.tohum, tekrar_tohumlari=tekrar, tekrar_kat_sinif=tekrar_tab,
        girdi_ozetleri=dict(karmalar=md5_dosya(hcsv), gorsel=md5_dosya(args.gorsel), eslesme_kodu=kod_ozeti),
        parametreler={k_: v for k_, v in vars(args).items()}, sure_dk=round((time.time() - t0) / 60, 1))
    (cikti / "ozet.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    md = ["# 2. aşama manifesti (r4.1): özet\n",
          "Gruplar iç içe: kati ⊆ foto ⊆ sahne. Kat ataması sahne_grup üzerinden; çelişki ve temsilci foto_grup "
          "üzerinden. Ayrıntılı tanım: `08_asama2_manifest.py` başlığı.\n",
          "## Sayılar\n", "```", json.dumps({k_: v for k_, v in ozet.items() if k_ not in ("tekrar_kat_sinif", "parametreler")},
                                             ensure_ascii=False, indent=2, default=str), "```\n",
          "## En büyük sahne grupları (v1) ve onları tutan kenar türleri\n", "```", zincir.to_string(index=False), "```\n",
          "## Katlar arası çiftler (temsilci düzeyinde; kurgu gereği 0 olanlar: md5_ayni, phash_le4, foto_kenari, "
          "sahne_kenari, gorsel_ayni_*)\n",
          "Diğer sütunlar bağımsız yakın-kaçak göstergeleridir; tek başına sızıntı demek değildir (çoğu ilgisiz "
          "görüntü). Kesin oran için kör görsel örneklem denetimi gerekir.\n", "```", sizinti.to_string(index=False), "```\n",
          "## Kalibrasyon: görsel etiketli (kesin) çiftlerde kurallar\n",
          "sahne kuralı: poz = aynı fotoğraf/sahne, neg = ilgisiz. foto kuralı: poz = aynı fotoğraf, neg = aynı sahne + "
          "ilgisiz. Etiketler önceki denetimlerin taramalarından geldiği için örneklem yanlıdır (zor örneklere ağırlıklı).\n",
          kal_not + "\n", "```", kal.to_string(index=False), "```\n",
          "Yalnız otomatik kenarlarla (görsel etiketler hariç) aynı sahne_grup'a düşen etiketli çiftler:\n", "```",
          oto_ozet.to_string() if len(oto_ozet) else "(etiket yok)", "```\n",
          f"Aynı-fotoğraf yasağı ihlali (zincirle aynı foto_grup'a düşen 'ilgisiz'/'aynı sahne' çiftleri): {len(ihlal)} "
          "(liste: zorunlu_ayrik_ihlalleri.csv)\n",
          "## kat_v1 (önerilen): kat × sınıf\n", "```", tv1.to_string(), "```\n",
          "## kat_v1: kat × kaynak\n", "```", tk1.to_string(), "```\n",
          "## kat_v2 (kopyalar dahil, duyarlılık): kat × sınıf\n", "```", tv2.to_string(), "```\n"]
    (cikti / "ozet.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"Kaydedildi: {cikti.resolve()}  ({ozet['sure_dk']} dk)")


if __name__ == "__main__":
    main()
