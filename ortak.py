"""Betiklerin ortak kullandığı yardımcılar (görüntü okuma, klasör adından bölme/sınıf çıkarma)."""
import re
import sys
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# SolarFCD'nin 4 birleşik sınıfı (klasör adı yazımı ne olursa olsun eşleştirmek için)
SOLARFCD_CLASSES = {
    "healthy": "Healthy",
    "surfaceobstruction": "Surface_Obstruction",
    "structuralfault": "Structural_Fault",
    "electricalfault": "Electrical_Fault",
}


def utf8_konsol():
    """Windows konsolunda Türkçe karakterlerin düzgün görünmesi için."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def infer_split(parts):
    """Yol bileşenlerinden train/val/test bölmesini bulur (yoksa boş döner)."""
    for p in parts:
        k = norm(p)
        if k in SOLARFCD_CLASSES:
            continue
        if k.startswith("train"):
            return "train"
        if k in ("val", "valid", "validation", "dev") or k.startswith("valid"):
            return "val"
        if k.startswith("test"):
            return "test"
    return ""


def infer_label(parts):
    """En derin klasörden başlayarak SolarFCD sınıf adını arar.
    Bulamazsa (ör. ham veri) bir üst klasör adını etiket olarak döndürür."""
    for p in reversed(parts):
        k = norm(p)
        if k in SOLARFCD_CLASSES:
            return SOLARFCD_CLASSES[k], True
    return (parts[-1] if parts else ""), False


def to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    return s in ("1", "true", "yes", "y", "evet", "t", "augmented", "aug")


def load_rgb(path, target=None) -> Image.Image:
    """Görüntüyü RGB olarak okur. JPEG'lerde hedef boyutun 2 katına kadar
    küçültülmüş çözme (draft) kullanır; büyük drone görüntülerinde ciddi hız kazandırır."""
    img = Image.open(path)
    if target and img.format == "JPEG":
        try:
            img.draft("RGB", (int(target) * 2, int(target) * 2))
        except Exception:
            pass
    return img.convert("RGB")


def list_images(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)
