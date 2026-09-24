"""
Exploration before the protocol was frozen: P1 splitter choice, extended (Appendix A.1).

Same comparison with 50 repeats, a third option (6 bins, k >= 20 separate) and
raw k vs. log k targets. Runtime ~5 min.

EXPLORATORY - not the official Phase 2 results (those come from train_static.py).
Settings differ from the final protocol (e.g. 100 trees, smaller grids).
Run from the repo root:  python experiments/phase2_exploration/e1b.py
"""
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
from common import *

bins6 = pd.cut(y, [1, 2, 3, 5, 9, 19, 50], labels=list("ABCDEF")).astype(str).to_numpy()
def folds(cv, seed, strat):
    lab = np.empty(len(y), int)
    for f, (tr, te) in enumerate(cv(5, shuffle=True, random_state=seed).split(X, strat, groups)):
        lab[te] = f
    return lab

def logt(m):
    return TransformedTargetRegressor(regressor=m, func=np.log, inverse_func=np.exp)

ms = models_default(100)
ms = {**ms, **{n + "_logk": logt(clone(m)) for n, m in ms.items() if n != "median"}}
R = 50
for name, cv, strat in [("GroupKFold", GroupKFold, bracket), ("SGKF 5 brackets", StratifiedGroupKFold, bracket), ("SGKF 6 bins (k>=20 own)", StratifiedGroupKFold, bins6)]:
    labs = [folds(cv, s, strat) for s in range(R)]
    big = [[(y[l == f] >= 20).sum() for f in range(5)] for l in labs]
    print(f"\n== {name}: k>=20 per test fold {np.min(big)}-{np.max(big)}")
    for mname, m in ms.items():
        rep, rep1 = [], []
        for l in labs:
            fm, f1 = [], []
            for f in range(5):
                tr, te = l != f, l == f
                p = clone(m).fit(X[tr], y[tr]).predict(X[te])
                fm.append(mae(y[te], p)); f1.append(within1(y[te], p))
            rep.append(np.mean(fm)); rep1.append(np.mean(f1))
        print(f"  {mname:11s} MAE {np.mean(rep):.3f} (sd of repeat-means {np.std(rep):.3f}) | within±1 {np.mean(rep1):.3f} (sd {np.std(rep1):.3f})")
