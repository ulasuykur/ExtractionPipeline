"""
Exploration before the protocol was frozen: P8 learning-curve design (Appendix A.4).

Noise of a learning-curve step for two designs (fixed test folds vs. subsampling
the pool first). Prints only noise, not the curve itself. Runtime ~4 min.

EXPLORATORY - not the official Phase 2 results (those come from train_static.py).
Settings differ from the final protocol (e.g. 100 trees, smaller grids).
Run from the repo root:  python experiments/phase2_exploration/e6_lc_noise.py
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
from common import *
logt = lambda m: TransformedTargetRegressor(regressor=m, func=np.log, inverse_func=np.exp)
b = models_default(100)
ms = {"knn_logk": logt(b["knn"]), "rf_logk": logt(b["rf"])}
R = 10
def subsets(idx, seed):
    """nested 25/50/75/100% subsets of idx, by group, stratified by bracket"""
    parts = [idx[te] for _, te in StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(idx, bracket[idx], groups[idx])]
    return [np.concatenate(parts[:j]) for j in (1, 2, 3, 4)]
for n, m in ms.items():
    # design (a): fixed outer test folds, training part subsampled; D draws per fold
    for D in (1, 5):
        steps = []  # per repeat: mean MAE at each fraction
        for r in range(R):
            acc = np.zeros(4); cnt = 0
            for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, bracket, groups)):
                for dd in range(D):
                    for j, sub in enumerate(subsets(tr, 1000 * r + 10 * f + dd)):
                        acc[j] += mae(y[te], clone(m).fit(X.iloc[sub], y[sub]).predict(X.iloc[te]))
                    cnt += 1
            steps.append(acc / cnt)
        s = np.array(steps)
        print(f"(a) fixed test folds, {D} draw(s): {n}: sd over repeats of MAE at 25/50/75/100% = {np.round(s.std(0),3)}; sd of step 75->100% = {np.std(s[:,3]-s[:,2]):.3f}; sd of 10-repeat mean step = {np.std(s[:,3]-s[:,2])/np.sqrt(R):.3f}")
    # design (b): subsample the pool to N datasets first, then 5-fold CV on the subsample
    steps = []
    for r in range(R):
        subs = subsets(np.arange(100), 5000 + r); row = []
        for sub in subs:
            e = [mae(y[sub][te], clone(m).fit(X.iloc[sub[tr]], y[sub][tr]).predict(X.iloc[sub[te]]))
                 for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(sub, bracket[sub], groups[sub])]
            row.append(np.mean(e))
        steps.append(row)
    s = np.array(steps)
    print(f"(b) subsample pool, then CV:        {n}: sd over repeats of MAE at 25/50/75/100 datasets = {np.round(s.std(0),3)}; sd of step 75->100 = {np.std(s[:,3]-s[:,2]):.3f}")
