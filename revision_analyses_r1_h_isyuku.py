"""İç değerlendirme R1 kontrolü - H: ltt_sinif_*.csv (r1_a_ic_alan.py ciktisi) -> 1.000 goruntu basina isletme birimleri. Yalniz okur."""
from pathlib import Path
import pandas as pd
H = Path(__file__).parent
for ad in ("orij", "orijbel", "temiz"):
    R = pd.read_csv(H / f"ltt_sinif_{ad}.csv")
    te_f = (R.ariza_kabul_n / R.kabul_ariza).where(R.kabul_ariza > 0)
    per = lambda col: (R[col] / R.te_n * 1000).mean()
    hata = {a: R[f"hata_{a}"].sum() / R[f"nk_{a}"].sum() for a in ("EF", "H", "SF", "SO")}
    print(f"[{ad}] per 1,000 test images: reviewed {(1 - R.kapsam).mean() * 1000:.0f}; faults accepted automatically "
          f"{per('ariza_kabul_n'):.1f}; accepted faults labelled no-fault {per('ariza_kacan'):.2f}; non-faults accepted as faults "
          f"{per('yanlis_alarm'):.2f} | mean test faults per split {te_f.mean():.1f} | error among accepted by class "
          + " ".join(f"{k} {v:.3f}" for k, v in hata.items()))
