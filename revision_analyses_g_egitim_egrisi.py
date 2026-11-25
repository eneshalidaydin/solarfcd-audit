"""R2.9: convergence summary from the saved training logs (log.csv) of all P3 runs. Read-only inputs."""
import glob
import os
import re
import sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
R = []
for f in glob.glob("C:/solarfcd/pilot/runs_asama2/grouped*_kat_v1_asama2_none/log.csv"):
    d = pd.read_csv(f)
    m = re.search(r"grouped(\d)_(.+)_s(4[234])_kat", os.path.basename(os.path.dirname(f)))
    R.append(dict(model=m.group(2), fold=int(m.group(1)), seed=int(m.group(3)), epochs=len(d),
                  train_acc_ep25=d.train_acc.iloc[24], train_acc_ep30=d.train_acc.iloc[-1],
                  train_loss_ep20=d.train_loss.iloc[19], train_loss_ep25=d.train_loss.iloc[24], train_loss_ep30=d.train_loss.iloc[-1]))
R = pd.DataFrame(R)
print(R.groupby("model").agg(runs=("fold", "size"), epochs=("epochs", "mean"), acc25=("train_acc_ep25", "mean"),
                             acc30=("train_acc_ep30", "mean"), loss20=("train_loss_ep20", "mean"),
                             loss25=("train_loss_ep25", "mean"), loss30=("train_loss_ep30", "mean")).round(4).to_string())
