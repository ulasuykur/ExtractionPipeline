"""
Calibration of stage 2 (sandbox, 2026-09-24): can a structure-based estimator ("the k with the
best silhouette", k-means on a fixed grid) find k in the heterogeneous artificial datasets at
all? Result 2026-09-24: Spearman rho = 0.18 -> stage 2 is too strict to decide.
EXPLORATORY - see docs/metalearner_methodology.md, section 10.1. Needs no OpenML data.
"""
import warnings
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from stage2_heterogeneous import make_dataset  # same generator and seed -> same first datasets

warnings.filterwarnings("ignore")
GRID = [2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50]

ks, best = [], []
for i in range(120):
    k, p, X = make_dataset()
    Z = StandardScaler().fit_transform(X)
    sil = [silhouette_score(Z, KMeans(g, n_init=3, random_state=0).fit_predict(Z)) for g in GRID]
    ks.append(k)
    best.append(GRID[int(np.argmax(sil))])
ks, best = np.array(ks), np.array(best)
print("silhouette-best k vs true k on the heterogeneous datasets (120):")
print("  Spearman rho = %.2f | exact = %.0f%% | within factor 1.5 = %.0f%%" % (
    stats.spearmanr(ks, best)[0], 100 * np.mean(best == ks),
    100 * np.mean(np.maximum(best / ks, ks / best) <= 1.5)))
