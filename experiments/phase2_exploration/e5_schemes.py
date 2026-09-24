"""
Exploration before the protocol was frozen: P2 folds x repeats and P5 metric details (Appendix A.2, A.3).

3/5/10 folds x 10/20 repeats and leave-one-group-out; clipping and rounding
effects on MAE. Runtime ~1 min.

EXPLORATORY - not the official Phase 2 results (those come from train_static.py).
Settings differ from the final protocol (e.g. 100 trees, smaller grids).
Run from the repo root:  python experiments/phase2_exploration/e5_schemes.py
"""
import numpy as np, time
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
from common import *
logt = lambda m: TransformedTargetRegressor(regressor=m, func=np.log, inverse_func=np.exp)
b = models_default(100)
ms = {"median": Median(), "knn_logk": logt(b["knn"]), "rf_logk": logt(b["rf"]), "ridge_logk": logt(b["ridge"])}
def run(K, R):
    per = {n: [] for n in ms}; raw = {n: [] for n in ms}; ntr = []
    for r in range(R):
        for tr, te in StratifiedGroupKFold(K, shuffle=True, random_state=r).split(X, bracket, groups):
            ntr.append(len(tr))
            for n, m in ms.items():
                p = clone(m).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
                per[n].append((r, mae(y[te], p), len(te))); raw[n].append((y[te], p))
    return per, raw, np.mean(ntr)
print("scheme      train  test  #fits | MAE median / knn / rf | sd(repeat-mean) knn | MDD knn-vs-median (corrected t, 5%)")
store = {}
for K, R in [(3, 10), (5, 10), (5, 20), (10, 10)]:
    t0 = time.time(); per, raw, ntr = run(K, R); dt = time.time() - t0
    store[(K, R)] = raw
    rm = {n: np.array([np.mean([v[1] for v in per[n] if v[0] == r]) for r in range(R)]) for n in ms}
    dif = np.array([a[1] - c[1] for a, c in zip(per["knn_logk"], per["median"])])
    J = len(dif); nte = 100 - ntr
    mdd = stats.t.ppf(0.975, J - 1) * np.sqrt((1 / J + nte / ntr) * dif.var(ddof=1))
    print(f"{K:2d}-fold x{R:2d}  {ntr:5.1f} {nte:5.1f} {J*len(ms):5d} | {rm['median'].mean():.2f} / {rm['knn_logk'].mean():.2f} / {rm['rf_logk'].mean():.2f} | {rm['knn_logk'].std():.3f} | {mdd:.2f}   ({dt:.0f}s)")
# LOGO: leave one group out
errs = {n: [] for n in ms}
for tr, te in LeaveOneGroupOut().split(X, y, groups):
    for n, m in ms.items():
        errs[n] += list(np.abs(y[te] - clip(clone(m).fit(X.iloc[tr], y[tr]).predict(X.iloc[te]))))
print(f"LOGO (91 folds, ~99 train, deterministic, no repeats): MAE median {np.mean(errs['median']):.2f} / knn {np.mean(errs['knn_logk']):.2f} / rf {np.mean(errs['rf_logk']):.2f}")

print("\nP5 (5x10): clipping and rounding")
raw = store[(5, 10)]
for n in ms:
    yt = np.concatenate([a for a, _ in raw[n]]); p = np.concatenate([c for _, c in raw[n]])
    print(f"  {n:10s} preds<2: {np.mean(p<2):5.1%} | MAE unclipped {np.abs(yt-p).mean():.3f} clipped {np.abs(yt-np.maximum(p,2)).mean():.3f} rounded {np.abs(yt-np.round(np.maximum(p,2))).mean():.3f} | exact {np.mean(np.round(np.maximum(p,2))==yt):.3f} within±1 {np.mean(np.abs(np.round(np.maximum(p,2))-yt)<=1):.3f} | median |err| {np.median(np.abs(yt-np.maximum(p,2))):.2f} | mean rel err |pred-k|/k {np.mean(np.abs(np.maximum(p,2)-yt)/yt):.2f}")
