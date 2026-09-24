"""
Exploration before the protocol was frozen: Why the static features fail, second diagnosis (Appendix A.6).

Permutation test (k shuffled), feature subsets (size only), loss matched to MAE,
subsets of the pool, and the controlled check with the image class-subset variants.
Runtime ~3 min.

EXPLORATORY - not the official Phase 2 results (those come from train_static.py).
Settings differ from the final protocol (e.g. 100 trees, smaller grids).
Run from the repo root:  python experiments/phase2_exploration/e7_why.py
"""
import numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, GridSearchCV
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from common import *
bins6 = pd.cut(y, [1, 2, 3, 5, 9, 19, 50], labels=list("ABCDEF")).astype(str).to_numpy()
logt = lambda m: TransformedTargetRegressor(regressor=m, func=np.log, inverse_func=np.exp)
Xs = X.copy(); Xs[SLOG] = signed_log(Xs[SLOG])
def cv_mae(model, cols, yy=y, R=10, sub=None):
    idx = np.arange(100) if sub is None else np.where(sub)[0]
    out = []
    for r in range(R):
        e = []
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(idx, bins6[idx] if sub is None else bracket[idx], groups[idx]):
            tr, te = idx[tr], idx[te]
            p = clone(model).fit(Xs.iloc[tr][cols], yy[tr]).predict(Xs.iloc[te][cols]); e.append(mae(yy[te], p))
        out.append(np.mean(e))
    return np.mean(out)
ridge = lambda: logt(Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=10))]))
knn = lambda: logt(Pipeline([("s", StandardScaler()), ("m", KNeighborsRegressor(10, weights="distance"))]))
rf = lambda: logt(RandomForestRegressor(200, min_samples_leaf=5, random_state=0, n_jobs=2))
rfabs = lambda: logt(RandomForestRegressor(100, min_samples_leaf=5, criterion="absolute_error", random_state=0, n_jobs=2))
hgb = lambda: logt(HistGradientBoostingRegressor(loss="absolute_error", max_depth=3, min_samples_leaf=5, learning_rate=0.05, max_iter=200, random_state=0))
print("1) Permutation test: shuffle k (breaks any link to features), how good do the models look by luck?")
rng = np.random.default_rng(1)
real = {n: cv_mae(m(), F, R=5) for n, m in [("ridge", ridge), ("knn", knn), ("rf", rf)]}
perm = {n: [] for n in real}
for i in range(20):
    yp = y.copy()
    # shuffle whole groups' k together would be ideal; shuffle rows (groups rarely matter: 88/100 singletons)
    yp = rng.permutation(y)
    for n, m in [("ridge", ridge), ("knn", knn), ("rf", rf)]:
        perm[n].append(cv_mae(m(), F, yy=yp, R=2))
med = cv_mae(Median(), F, R=5)
for n in real:
    pv = (1 + np.sum(np.array(perm[n]) <= real[n])) / 21
    print(f"   {n:6s} real MAE {real[n]:.3f} | shuffled-k MAE {np.mean(perm[n]):.3f} ± {np.std(perm[n]):.3f} | p≈{pv:.2f}   (median baseline {med:.3f})")
print("\n2) Fewer features (8 features, most noise, can drown the one useful one):")
for cols in [F, ["log_n", "log_p"], ["log_n"]]:
    print(f"   {str(cols) if len(cols)<8 else 'all 8':22s}" + "".join(f" {n}={cv_mae(m(), cols):.3f}" for n, m in [("ridge", ridge), ("knn", knn), ("rf", rf)]))
print("\n3) Loss matched to MAE (absolute error instead of squared error):")
print(f"   RF abs-error {cv_mae(rfabs(), F, R=5):.3f} | GradBoost abs-error {cv_mae(hgb(), F, R=5):.3f}")
print("\n4) Subsets (MAE model vs median baseline on the same subset):")
tab = ~d["source"].str.startswith("image").to_numpy()
for nm, sub in [("tabular only (83)", tab), ("k<20 only (94)", y < 20)]:
    print(f"   {nm:18s} median {cv_mae(Median(), F, sub=sub):.3f} | ridge {cv_mae(ridge(), F, sub=sub):.3f} knn {cv_mae(knn(), F, sub=sub):.3f} rf {cv_mae(rf(), F, sub=sub):.3f}")
print("\n5) Controlled check: same source, only k changes (image class-subset variants):")
for g in ["img_554", "img_40996", "img_41982"]:
    s = d[d.group == g].sort_values("k_true")
    print(s[["name", "k_true"] + F].round(3).to_string(index=False)); print()
