"""
Exploration before the protocol was frozen: P4 tuned (nested CV) vs. default hyperparameters.

10 repeats, log k target; also counts which hyperparameters were chosen.
Runtime ~8 min (mostly Random Forest).

EXPLORATORY - not the official Phase 2 results (those come from train_static.py).
Settings differ from the final protocol (e.g. 100 trees, smaller grids).
Run from the repo root:  python experiments/phase2_exploration/e3_nested.py
"""
import numpy as np, time, collections, json
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, GridSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.base import clone
from common import *
logt = lambda m: TransformedTargetRegressor(regressor=m, func=np.log, inverse_func=np.exp)
grids = {n: {"regressor__" + k: v for k, v in g.items()} for n, g in GRIDS.items()}
base = models_default(100)
R = 10
res = collections.defaultdict(list); picks = collections.defaultdict(collections.Counter); t = collections.Counter()
for r in range(R):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(X, bracket, groups):
        Xtr, ytr, gtr = X.iloc[tr], y[tr], groups[tr]
        res["median"].append((r, mae(y[te], Median().fit(Xtr, ytr).predict(X.iloc[te])), within1(y[te], Median().fit(Xtr, ytr).predict(X.iloc[te]))))
        for n in ["ridge", "knn", "rf"]:
            for mode in ["default", "nested"]:
                t0 = time.time()
                m = logt(clone(base[n]))
                if mode == "nested":
                    m = GridSearchCV(m, grids[n], cv=GroupKFold(5), scoring="neg_mean_absolute_error", n_jobs=1)
                    m.fit(Xtr, ytr, groups=gtr)
                    picks[n][json.dumps({k.split("__")[-1]: (round(v, 4) if isinstance(v, float) else v) for k, v in m.best_params_.items()})] += 1
                else:
                    m.fit(Xtr, ytr)
                p = m.predict(X.iloc[te])
                t[(n, mode)] += time.time() - t0
                res[f"{n}_{mode}"].append((r, mae(y[te], p), within1(y[te], p)))
    print("repeat", r, "done", flush=True)
print("\nlog(k) target, StratifiedGroupKFold 5x10")
for k, v in res.items():
    a = np.array(v); rm = [a[a[:, 0] == r, 1].mean() for r in range(R)]; r1 = [a[a[:, 0] == r, 2].mean() for r in range(R)]
    print(f"  {k:15s} MAE {np.mean(rm):.3f} (sd {np.std(rm):.3f}) within±1 {np.mean(r1):.3f}")
print("\nruntime (s, 50 outer folds):", {f"{a}_{b}": round(v, 1) for (a, b), v in t.items()})
for n, c in picks.items():
    print(f"\nchosen hyperparameters {n} (out of 50 outer folds):"); [print("  ", c_, k) for k, c_ in c.most_common(6)]
