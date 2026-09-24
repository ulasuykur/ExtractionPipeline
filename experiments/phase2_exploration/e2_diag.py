"""
Exploration before the protocol was frozen: Why the static features fail, first diagnosis (Appendix A.6).

Spearman correlation of each feature with k, error per k bracket and per source,
and the predictions for the k >= 20 datasets (default models, log k). Runtime <1 min.

EXPLORATORY - not the official Phase 2 results (those come from train_static.py).
Settings differ from the final protocol (e.g. 100 trees, smaller grids).
Run from the repo root:  python experiments/phase2_exploration/e2_diag.py
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
from common import *
print("Spearman rho of each feature with k_true (whole pool, descriptive only):")
for f in F:
    r, p = spearmanr(X[f], y); print(f"  {f:10s} rho={r:+.2f} p={p:.3f}")
logt = lambda m: TransformedTargetRegressor(regressor=m, func=np.log, inverse_func=np.exp)
ms = {"median": Median(), "ridge_logk": logt(models_default(100)["ridge"]), "rf_logk": logt(models_default(100)["rf"]), "knn_logk": logt(models_default(100)["knn"])}
R = 10
oof = {m: np.zeros((R, len(y))) for m in ms}
for r in range(R):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, bracket, groups):
        for n, m in ms.items():
            oof[n][r, te] = clip(clone(m).fit(X.iloc[tr], y[tr]).predict(X.iloc[te]))
src = d["source"].str.startswith("image").map({True: "image", False: "tabular"}).to_numpy()
print("\nOut-of-fold, averaged over 10 repeats. MAE per bracket (n):")
hdr = "  bracket     " + "".join(f"{n:>12s}" for n in ms); print(hdr)
for b in list("ABCDE"):
    mask = bracket == b
    print(f"  {b} (n={mask.sum():2d})   " + "".join(f"{np.abs(oof[n][:, mask]-y[mask]).mean():12.2f}" for n in ms))
for s in ["tabular", "image"]:
    mask = src == s
    print(f"  {s:8s}(n={mask.sum():2d})" + "".join(f"{np.abs(oof[n][:, mask]-y[mask]).mean():12.2f}" for n in ms))
mask = y >= 20
print(f"  k>=20 (n={mask.sum()})    " + "".join(f"{np.abs(oof[n][:, mask]-y[mask]).mean():12.2f}" for n in ms))
print(f"  all but k>=20" + "".join(f"{np.abs(oof[n][:, ~mask]-y[~mask]).mean():12.2f}" for n in ms))
print("\nOther views of the same predictions:")
for n in ms:
    p = oof[n]
    rho = np.mean([spearmanr(p[r], y)[0] if p[r].std() > 0 else np.nan for r in range(R)])
    logerr = np.abs(np.log(p) - np.log(y)).mean()
    print(f"  {n:10s} Spearman(pred,true)={rho:+.2f}  mean |log pred - log k|={logerr:.3f} (x{np.exp(logerr):.2f} typical factor off)  exact={np.mean(np.round(p)==y):.3f}")
print("\nRF_logk mean prediction for k>=20 datasets:")
for i in np.where(mask)[0]:
    print(f"  {d.name[i]:22s} k={int(y[i]):2d}  rf={oof['rf_logk'][:, i].mean():5.1f} knn={oof['knn_logk'][:, i].mean():5.1f} ridge={oof['ridge_logk'][:, i].mean():5.1f}")
