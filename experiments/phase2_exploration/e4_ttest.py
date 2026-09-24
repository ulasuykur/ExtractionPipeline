"""
Exploration before the protocol was frozen: P6 statistical test (Appendix A.5).

300 synthetic pools where two models are equally good by construction: how often
do the plain and the corrected (Nadeau-Bengio) t-test wrongly call a difference?
Runtime ~1.5 min.

EXPLORATORY - not the official Phase 2 results (those come from train_static.py).
Settings differ from the final protocol (e.g. 100 trees, smaller grids).
Run from the repo root:  python experiments/phase2_exploration/e4_ttest.py
"""
import numpy as np
from scipy import stats
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsRegressor
from common import y as yreal, groups, bracket
rng = np.random.default_rng(0)
def tests(dif, ntest, ntrain):
    J = len(dif); m = dif.mean(); s2 = dif.var(ddof=1)
    t_naive = m / np.sqrt(s2 / J)
    t_corr = m / np.sqrt((1 / J + ntest / ntrain) * s2)
    return 2 * stats.t.sf(abs(t_naive), J - 1), 2 * stats.t.sf(abs(t_corr), J - 1)
S = 300; rej = np.zeros(2)
for s in range(S):
    Xs = rng.normal(size=(100, 8)); ys = rng.permutation(yreal)
    dif = []
    for r in range(10):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=r).split(Xs, bracket, groups):
            ea = np.abs(np.exp(KNeighborsRegressor(5).fit(Xs[tr][:, :4], np.log(ys[tr])).predict(Xs[te][:, :4])) - ys[te]).mean()
            eb = np.abs(np.exp(KNeighborsRegressor(5).fit(Xs[tr][:, 4:], np.log(ys[tr])).predict(Xs[te][:, 4:])) - ys[te]).mean()
            dif.append(ea - eb)
    pn, pc = tests(np.array(dif), 20, 80)
    rej += [pn < 0.05, pc < 0.05]
print(f"{S} simulated pools where both models are truly equal.")
print(f"  naive paired t-test says 'significant difference' in {rej[0]/S:.0%} of pools (should be 5%)")
print(f"  corrected (Nadeau-Bengio) t-test:                 {rej[1]/S:.0%} of pools (should be 5%)")
