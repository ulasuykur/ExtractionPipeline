"""
Exploration before the protocol was frozen: P1 splitter choice (Appendix A.1).

Plain GroupKFold vs. StratifiedGroupKFold (5 pool brackets): how different the
splits are across seeds, how balanced the test folds are, and how much the scores of
the default models wobble between repeats (20 repeats). Runtime ~1 min.

EXPLORATORY - not the official Phase 2 results (those come from train_static.py).
Settings differ from the final protocol (e.g. 100 trees, smaller grids).
Run from the repo root:  python experiments/phase2_exploration/e1_splitters.py
"""
import numpy as np, time
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import adjusted_rand_score
from sklearn.base import clone
from common import *

def folds(cv, seed):
    lab = np.empty(len(y), int)
    for f, (tr, te) in enumerate(cv(5, shuffle=True, random_state=seed).split(X, bracket, groups)):
        lab[te] = f
    return lab

for name, cv in [("GroupKFold", GroupKFold), ("StratifiedGroupKFold", StratifiedGroupKFold)]:
    labs = [folds(cv, s) for s in range(20)]
    uniq = len({tuple(pd.factorize(l)[0]) for l in labs})
    ari = np.mean([adjusted_rand_score(labs[i], labs[j]) for i in range(20) for j in range(i+1, 20)])
    # fold composition
    sizes = [np.bincount(l) for l in labs]
    Ecount = [[(bracket[l == f] == "E").sum() for f in range(5)] for l in labs]
    Acount = [[(bracket[l == f] == "A").sum() for f in range(5)] for l in labs]
    big = [[(y[l == f] >= 20).sum() for f in range(5)] for l in labs]
    print(f"\n== {name}")
    print(f" distinct partitions in 20 seeds: {uniq}; mean ARI between seeds: {ari:.3f} (0=unrelated, 1=identical)")
    print(f" test-fold size range: {np.min(sizes)}-{np.max(sizes)}")
    print(f" k=2 per test fold: {np.min(Acount)}-{np.max(Acount)} ; bracket E: {np.min(Ecount)}-{np.max(Ecount)} ; k>=20: {np.min(big)}-{np.max(big)}")
    # stability of model estimates across repeats
    for mname, m in models_default().items():
        rep = []
        fold_maes = []
        for s in range(20):
            l = labs[s]; fm = []
            for f in range(5):
                tr, te = l != f, l == f
                mm = clone(m).fit(X[tr], y[tr])
                fm.append(mae(y[te], mm.predict(X[te])))
            rep.append(np.mean(fm)); fold_maes += fm
        print(f"  {mname:6s} MAE mean {np.mean(rep):.3f} | sd across folds {np.std(fold_maes):.2f} | sd of repeat-means {np.std(rep):.3f}")
