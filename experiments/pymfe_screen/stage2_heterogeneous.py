"""
Stage 2 of the pymfe screening (sandbox, 2026-09-24): heterogeneous artificial datasets.
Every dataset gets its own structure, so features must track k ACROSS heterogeneous
datasets (like the real pool), not only within a controlled series.
EXPLORATORY - see docs/metalearner_methodology.md, section 10.1. Needs no OpenML data.
"""
import warnings, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
import stage1_controlled as s
warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

def make_dataset():
    k = int(rng.choice([2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50]))
    p = int(rng.integers(4, 81))                      # 4..80 columns
    d = int(rng.integers(2, max(3, p) + 1))           # dimensions that carry cluster structure
    n = 1000
    sizes = rng.dirichlet(np.full(k, rng.uniform(0.5, 5)))           # unequal group sizes
    sizes = np.maximum(1, np.round(sizes * n)).astype(int)
    sep = rng.uniform(1.0, 8.0)                       # separation of group centres
    X = []
    for j in range(k):
        A = rng.normal(size=(d, d)) * rng.uniform(0.3, 1.5)          # random within-group correlation
        centre = rng.normal(size=d) * sep
        X.append(centre + rng.normal(size=(sizes[j], d)) @ A)
    X = np.vstack(X)
    X = X @ rng.normal(size=(d, d))                   # random rotation/mixing of the structured dims
    noise = rng.normal(size=(len(X), p - d)) * rng.uniform(0.5, 3) if p > d else np.empty((len(X), 0))
    X = np.hstack([X, noise])
    X = X * rng.lognormal(0, 1, size=X.shape[1])      # different scales per column
    return k, p, X

if __name__ == "__main__":
  rows = []
  for i in range(300):
      k, p, X = make_dataset()
      rows.append({"k": k, "p": p, "log_p": np.log10(p), **s.candidates(X)})
  df = pd.DataFrame(rows); out = Path(__file__).resolve().parents[2] / "outputs" / "pymfe_screen"; out.mkdir(parents=True, exist_ok=True); df.to_csv(out / "stage2_hetero.csv", index=False)
  feats = [c for c in df.columns if c[:2] in ("A_", "B_", "C_", "D_") or c.startswith("ref_")]
  def partial_rho(f):
      # Spearman of residuals after removing log_p (rank-based), i.e. information beyond p
      r = df[[f, "k", "log_p"]].rank()
      res = lambda a: a - np.polyval(np.polyfit(r["log_p"], a, 1), r["log_p"])
      return stats.pearsonr(res(r[f]), res(r["k"]))[0]
  out = pd.DataFrame([{"feature": f, "rho_hetero": stats.spearmanr(df["k"], df[f])[0] if df[f].nunique() > 1 else np.nan,
                       "rho_beyond_p": partial_rho(f) if df[f].nunique() > 1 else np.nan} for f in feats])
  print(out.round(2).to_string(index=False))
