"""Adım 2 - Eğitim ve değerlendirme (timm + PyTorch).

İki protokol:
  official : SolarFCD yazarlarının train/val/test bölmesi (makaledeki %86,68 ile birebir karşılaştırma)
  grouped  : 01_veri_denetimi.py'nin ürettiği grup bazlı katlar (yakın kopyalar aynı tarafta, sızıntısız)

Örnekler:
  python 02_egit.py --manifest denetim_final/manifest.csv --model convnext_tiny.fb_in22k_ft_in1k --smoke
  python 02_egit.py --manifest denetim_final/manifest.csv --model resnetv2_101.a1h_in1k --epochs 30 --tta
  python 02_egit.py --manifest denetim_final/manifest.csv --model convnext_tiny.fb_in22k_ft_in1k --protocol grouped --fold 0

Çıktılar runs/<koşu_adı>/ altında: config.json, log.csv, best.pt, results.json,
test_predictions.csv, confusion_matrix.png
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ortak import load_rgb, to_bool, utf8_konsol


# ------------------------------------------------------------------ veri kümesi (modül düzeyinde: Windows için)
class ManifestDS:
    def __init__(self, yollar, etiketler, tf, img):
        self.yollar, self.etiketler, self.tf, self.img = yollar, etiketler, tf, img

    def __len__(self):
        return len(self.yollar)

    def __getitem__(self, i):
        return self.tf(load_rgb(self.yollar[i], self.img)), self.etiketler[i], i


def tohum(s):
    import torch
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def bolmeleri_hazirla(df, args):
    if "ok" in df.columns:
        df = df[df["ok"].map(to_bool)]
    if args.protocol == "official":
        tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
        if len(tr) == 0 or len(te) == 0:
            raise SystemExit("Manifestte train/test bölmesi bulunamadı. 01_veri_denetimi.py çıktısını kontrol et.")
        if len(va) == 0:
            print("[!] Doğrulama bölmesi yok; eğitimden grup bazlı %10 ayrılıyor.")
            tr, va = grup_ayir(tr, 0.1, args.seed)
    else:
        for s in (args.fold_col, args.group_col):
            if s not in df.columns:
                raise SystemExit(f"'{s}' sütunu yok; önce 01_veri_denetimi.py çalıştırılmalı.")
        te = df[df[args.fold_col] == args.fold]
        kalan = df[(df[args.fold_col] != args.fold) & (df[args.fold_col] >= 0)]
        tr, va = grup_ayir(kalan, args.val_frac, args.seed, args.group_col)
    if args.aug_col:
        if args.aug_col not in df.columns:
            raise SystemExit(f"'{args.aug_col}' sütunu manifestte yok.")
        if args.exclude_aug_eval:
            va = va[~va[args.aug_col].map(to_bool)]
            te = te[~te[args.aug_col].map(to_bool)]
        if args.exclude_aug_train:
            tr = tr[~tr[args.aug_col].map(to_bool)]
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def grup_ayir(df, oran, seed, grup_sutunu="group_id"):
    from sklearn.model_selection import GroupShuffleSplit
    gruplar = df[grup_sutunu] if grup_sutunu in df.columns else np.arange(len(df))
    gss = GroupShuffleSplit(n_splits=1, test_size=oran, random_state=seed)
    a, b = next(gss.split(df, df["label"], gruplar))
    return df.iloc[a], df.iloc[b]


def olcutler(y, p, prob, siniflar):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                                 cohen_kappa_score, confusion_matrix, matthews_corrcoef,
                                 precision_recall_fscore_support, roc_auc_score)
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    sonuc = dict(
        accuracy=float(accuracy_score(y, p)),
        balanced_accuracy=float(balanced_accuracy_score(y, p)),
        macro_precision=float(pr), macro_recall=float(rc), macro_f1=float(f1),
        weighted_f1=float(precision_recall_fscore_support(y, p, average="weighted", zero_division=0)[2]),
        mcc=float(matthews_corrcoef(y, p)),
        kappa=float(cohen_kappa_score(y, p)),
        confusion_matrix=confusion_matrix(y, p, labels=list(range(len(siniflar)))).tolist(),
        per_class=classification_report(y, p, labels=list(range(len(siniflar))), target_names=siniflar,
                                        output_dict=True, zero_division=0),
    )
    try:
        sonuc["macro_auc_ovr"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro",
                                                     labels=list(range(len(siniflar)))))
    except Exception:
        sonuc["macro_auc_ovr"] = None
    return sonuc


def karmasiklik_ciz(cm, siniflar, yol, baslik):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_xticks(range(len(siniflar)))
    ax.set_yticks(range(len(siniflar)))
    ax.set_xticklabels(siniflar, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(siniflar, fontsize=8)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title(baslik, fontsize=9)
    plt.tight_layout()
    plt.savefig(yol, dpi=130)
    plt.close(fig)


def main():
    utf8_konsol()
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True, help="timm model adı")
    ap.add_argument("--protocol", choices=["official", "grouped"], default="official")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--fold-col", default="grouped_fold", help="grouped_fold (pHash+derin) ya da grouped_fold_phash")
    ap.add_argument("--group-col", default="group_id", help="group_id ya da group_id_phash")
    ap.add_argument("--val-frac", type=float, default=0.125)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=2.0, help="Isınma (epoch)")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--class-weights", action="store_true", help="Ters sıklık sınıf ağırlıkları")
    ap.add_argument("--patience", type=int, default=8, help="Erken durdurma (doğrulama makro-F1)")
    ap.add_argument("--linear-probe", action="store_true", help="Omurgayı dondur, yalnız sınıflandırıcıyı eğit")
    ap.add_argument("--tta", action="store_true", help="Testte yatay/dikey çevirme ortalaması")
    ap.add_argument("--aug-col", default="", help="Manifestte artırılmış örnek bayrağı sütunu (varsa)")
    ap.add_argument("--exclude-aug-eval", action="store_true", help="Artırılmış örnekleri doğrulama/testten çıkar")
    ap.add_argument("--exclude-aug-train", action="store_true", help="Hazır artırılmış kopyaları eğitimden çıkar")
    ap.add_argument("--workers", type=int, default=max(0, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--tag", default="", help="Koşu adına eklenecek etiket")
    ap.add_argument("--smoke", action="store_true", help="Boru hattını 1 epoch / birkaç adımla dene")
    ap.add_argument("--no-pretrained", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import timm
    from torch.utils.data import DataLoader
    from torchvision import transforms

    tohum(args.seed)
    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    if cihaz == "cpu":
        print("[!] GPU bulunamadı; CPU'da çalışıyor (çok yavaş olur).")
    torch.backends.cudnn.benchmark = True
    amp_tipi = None
    if cihaz == "cuda":
        amp_tipi = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    olcekleyici = torch.amp.GradScaler("cuda", enabled=(amp_tipi == torch.float16))

    df = pd.read_csv(args.manifest)
    tr, va, te = bolmeleri_hazirla(df, args)
    siniflar = sorted(tr["label"].unique().tolist())
    for ad, d in (("doğrulama", va), ("test", te)):
        bilinmeyen = set(d["label"]) - set(siniflar)
        if bilinmeyen:
            raise SystemExit(f"{ad} bölmesinde eğitimde olmayan sınıf(lar): {bilinmeyen}")
    sinif_idx = {c: i for i, c in enumerate(siniflar)}

    kisa = args.model.split(".")[0]
    ad = f"{args.protocol}{args.fold if args.protocol == 'grouped' else ''}_{kisa}_{args.img_size}_s{args.seed}"
    if args.linear_probe:
        ad += "_lp"
    if args.tag:
        ad += "_" + args.tag
    if args.smoke:
        ad = "SMOKE_" + ad
    kosu = Path(args.out) / ad
    kosu.mkdir(parents=True, exist_ok=True)

    kw = {}
    if any(k in args.model for k in ("vit", "dinov2", "deit", "eva")):
        kw["img_size"] = args.img_size
    model = timm.create_model(args.model, pretrained=not args.no_pretrained, num_classes=len(siniflar),
                              drop_path_rate=args.drop_path, **kw)
    if args.linear_probe:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.get_classifier().parameters():
            p.requires_grad = True
    kanal_son = cihaz == "cuda" and not any(k in args.model for k in ("vit", "dinov2", "swin", "deit", "eva"))
    model = model.to(cihaz)
    if kanal_son:
        model = model.to(memory_format=torch.channels_last)

    cfg = timm.data.resolve_data_config({}, model=model)
    S = args.img_size
    egitim_tf = transforms.Compose([
        transforms.RandomResizedCrop(S, scale=(0.6, 1.0), ratio=(0.75, 1.333)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.RandomRotation(15)], p=0.3),
        transforms.ColorJitter(0.2, 0.2, 0.1, 0.02),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    deger_tf = transforms.Compose([transforms.Resize((S, S)), transforms.ToTensor(),
                                   transforms.Normalize(cfg["mean"], cfg["std"])])

    def yukleyici(d, tf, karistir):
        ds = ManifestDS(d["path"].tolist(), [sinif_idx[c] for c in d["label"]], tf, S)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=karistir, num_workers=args.workers,
                          pin_memory=(cihaz == "cuda"), persistent_workers=(args.workers > 0),
                          drop_last=karistir and len(ds) > args.batch_size)

    dl_tr, dl_va, dl_te = yukleyici(tr, egitim_tf, True), yukleyici(va, deger_tf, False), yukleyici(te, deger_tf, False)

    agirlik = None
    sayim = tr["label"].value_counts().reindex(siniflar).to_numpy(dtype=float)
    if args.class_weights:
        w = sayim.sum() / (len(siniflar) * sayim)
        agirlik = torch.tensor(w / w.mean(), dtype=torch.float32, device=cihaz)
    kayip_fn = nn.CrossEntropyLoss(weight=agirlik, label_smoothing=args.label_smoothing)

    from timm.optim import create_optimizer_v2
    opt = create_optimizer_v2(model, opt="adamw", lr=args.lr, weight_decay=args.wd)
    epochs = 1 if args.smoke else args.epochs
    adim_ep = 3 if args.smoke else len(dl_tr)
    toplam = max(1, epochs * adim_ep)
    isinma = min(int(args.warmup * adim_ep), toplam // 2)

    def lr_carpani(adim):
        if adim < isinma:
            return (adim + 1) / max(1, isinma)
        ilerleme = (adim - isinma) / max(1, toplam - isinma)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * ilerleme))

    zamanlayici = torch.optim.lr_scheduler.LambdaLR(opt, lr_carpani)

    def ileri(x):
        with torch.autocast(device_type=cihaz, dtype=amp_tipi or torch.float32, enabled=amp_tipi is not None):
            return model(x)

    def tahmin(dl, tta=False, sinir=None):
        model.eval()
        olas, hedef, sira = [], [], []
        with torch.no_grad():
            for b, (x, y, i) in enumerate(dl):
                if sinir and b >= sinir:
                    break
                x = x.to(cihaz, non_blocking=True)
                if kanal_son:
                    x = x.to(memory_format=torch.channels_last)
                p = ileri(x).float().softmax(1)
                if tta:
                    p = p + ileri(torch.flip(x, dims=[3])).float().softmax(1)
                    p = p + ileri(torch.flip(x, dims=[2])).float().softmax(1)
                    p = p / 3
                olas.append(p.cpu())
                hedef.append(y)
                sira.append(i)
        olas = torch.cat(olas).numpy()
        return olas, torch.cat(hedef).numpy(), torch.cat(sira).numpy()

    yapilandirma = dict(vars(args), siniflar=siniflar, n_train=len(tr), n_val=len(va), n_test=len(te),
                        train_sayim=dict(zip(siniflar, sayim.astype(int).tolist())),
                        cihaz=cihaz, gpu=(torch.cuda.get_device_name(0) if cihaz == "cuda" else None),
                        amp=str(amp_tipi), torch=torch.__version__, timm=timm.__version__)
    (kosu / "config.json").write_text(json.dumps(yapilandirma, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nKoşu: {ad}\nSınıflar: {siniflar}\nEğitim/Doğrulama/Test: {len(tr)}/{len(va)}/{len(te)}  | cihaz: {cihaz}")

    from sklearn.metrics import f1_score
    en_iyi_f1, en_iyi_ep, sabir, kayitlar = -1.0, -1, 0, []
    t_bas = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        t0, top_kayip, dogru, n = time.time(), 0.0, 0, 0
        for b, (x, y, _) in enumerate(dl_tr):
            if args.smoke and b >= adim_ep:
                break
            x, y = x.to(cihaz, non_blocking=True), y.to(cihaz, non_blocking=True)
            if kanal_son:
                x = x.to(memory_format=torch.channels_last)
            with torch.autocast(device_type=cihaz, dtype=amp_tipi or torch.float32, enabled=amp_tipi is not None):
                cikti = model(x)
                kayip = kayip_fn(cikti, y)
            opt.zero_grad(set_to_none=True)
            olcekleyici.scale(kayip).backward()
            olcekleyici.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            olcekleyici.step(opt)
            olcekleyici.update()
            zamanlayici.step()
            top_kayip += kayip.item() * y.size(0)
            dogru += (cikti.argmax(1) == y).sum().item()
            n += y.size(0)
        olas_va, y_va, _ = tahmin(dl_va, sinir=3 if args.smoke else None)
        p_va = olas_va.argmax(1)
        f1_va = f1_score(y_va, p_va, average="macro", zero_division=0)
        acc_va = float((p_va == y_va).mean())
        kayit = dict(epoch=ep, train_loss=top_kayip / max(1, n), train_acc=dogru / max(1, n), val_acc=acc_va,
                     val_macro_f1=f1_va, lr=opt.param_groups[0]["lr"], sure_sn=round(time.time() - t0, 1))
        kayitlar.append(kayit)
        print(f"ep {ep:02d} | kayıp {kayit['train_loss']:.4f} | eğitim acc {kayit['train_acc']:.4f} | "
              f"doğr. acc {acc_va:.4f} | doğr. F1 {f1_va:.4f} | {kayit['sure_sn']} sn")
        if f1_va > en_iyi_f1:
            en_iyi_f1, en_iyi_ep, sabir = f1_va, ep, 0
            torch.save(model.state_dict(), kosu / "best.pt")
        else:
            sabir += 1
            if sabir >= args.patience:
                print(f"Erken durdurma (en iyi epoch {en_iyi_ep}).")
                break
    pd.DataFrame(kayitlar).to_csv(kosu / "log.csv", index=False)
    egitim_dk = round((time.time() - t_bas) / 60, 1)

    model.load_state_dict(torch.load(kosu / "best.pt", map_location=cihaz))
    sinir = 3 if args.smoke else None
    olas, y_te, sira = tahmin(dl_te, sinir=sinir)
    test = olcutler(y_te, olas.argmax(1), olas, siniflar)
    sonuc = dict(config=yapilandirma, best_epoch=en_iyi_ep, best_val_macro_f1=en_iyi_f1,
                 train_minutes=egitim_dk, test=test)
    if args.tta:
        olas_t, y_t, _ = tahmin(dl_te, tta=True, sinir=sinir)
        sonuc["test_tta"] = olcutler(y_t, olas_t.argmax(1), olas_t, siniflar)

    tahminler = te.iloc[sira][["path", "label"]].copy()
    tahminler["tahmin"] = [siniflar[k] for k in olas.argmax(1)]
    for k, c in enumerate(siniflar):
        tahminler[f"p_{c}"] = olas[:, k]
    tahminler.to_csv(kosu / "test_predictions.csv", index=False, encoding="utf-8-sig")
    (kosu / "results.json").write_text(json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
    karmasiklik_ciz(test["confusion_matrix"], siniflar, kosu / "confusion_matrix.png",
                    f"{ad}\nacc={test['accuracy']:.4f}  makro-F1={test['macro_f1']:.4f}")

    print(f"\n=== TEST ({ad}) ===")
    print(f"Doğruluk: {test['accuracy']:.4f} | Makro-F1: {test['macro_f1']:.4f} | MCC: {test['mcc']:.4f} | "
          f"Kappa: {test['kappa']:.4f}")
    if args.tta:
        t2 = sonuc["test_tta"]
        print(f"TTA ile  doğruluk: {t2['accuracy']:.4f} | makro-F1: {t2['macro_f1']:.4f}")
    print("Karmaşıklık matrisi (satır=gerçek, sütun=tahmin):", siniflar)
    for satir in test["confusion_matrix"]:
        print("   ", satir)
    print(f"Eğitim süresi: {egitim_dk} dk | Çıktılar: {kosu.resolve()}")


if __name__ == "__main__":
    main()
