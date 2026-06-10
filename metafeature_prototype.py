"""
Prototype: meta-feature extraction pipeline over OpenML datasets.

Output: a pandas DataFrame (also saved as CSV) with
    rows    = OpenML datasets
    columns = 4 meta-features:
        nr_inst       -- number of instances n
        nr_attr       -- number of attributes p
        inst_to_attr  -- examples per dimension n/p
        skewness.mean -- skewness of feature x_i, averaged over features
                         (pymfe computes per-feature skewness; a summary
                          function is required to get one fixed-length
                          value per dataset)

Install dependencies first:
    pip install pymfe openml pandas numpy scipy scikit-learn
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import openml
from pymfe.mfe import MFE

warnings.filterwarnings("ignore")  # pymfe is chatty about NaNs/precision


# ---------------------------------------------------------------------
# 1. Dataset selection: OpenML id -> (name, true number of classes k)
#    The class count doubles as ground-truth k for the later meta-learner.
# ---------------------------------------------------------------------

DATASETS: dict[int, str] = {
    61:   "iris",          # k=3,  n=150,    p=4
    187:  "wine",          # k=3,  n=178,    p=13
    1499: "seeds",         # k=3,  n=210,    p=7
    41:   "glass",         # k=6,  n=214,    p=9
    39:   "ecoli",         # k=8,  n=336,    p=7
    54:   "vehicle",       # k=4,  n=846,    p=18
    36:   "segment",       # k=7,  n=2310,   p=19
    28:   "optdigits",     # k=10, n=5620,   p=64
    32:   "pendigits",     # k=10, n=10992,  p=16
    1462: "banknote-auth", # k=2,  n=1372,   p=4
}

# The 4 meta-features, by their pymfe identifiers
FEATURES = ["nr_inst", "nr_attr", "inst_to_attr", "skewness"]


# ---------------------------------------------------------------------
# 2. Loading: OpenML dataset -> numeric matrix X
# ---------------------------------------------------------------------

def load_openml_numeric(dataset_id: int) -> np.ndarray:
    """Fetch a dataset from OpenML and return its numeric feature matrix.

    The target column is excluded (this is an unsupervised setting --
    labels only serve as ground-truth k later). Non-numeric columns are
    dropped so that skewness is well defined; rows with NaNs are removed.
    """
    ds = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=False,
        download_features_meta_data=False,
    )
    X_df, _, _, _ = ds.get_data(
        target=ds.default_target_attribute, dataset_format="dataframe"
    )
    X_num = X_df.select_dtypes(include=[np.number]).dropna(axis=0)
    return X_num.to_numpy(dtype=float)


# ---------------------------------------------------------------------
# 3. Extraction: numeric matrix -> {meta_feature_name: value} via pymfe
# ---------------------------------------------------------------------

def extract_metafeatures(X: np.ndarray) -> dict[str, float]:
    """Run pymfe on X (unsupervised: no y passed to fit).

    summary=["mean"] collapses per-feature measures (here: skewness of
    each x_i) into one value; nr_inst / nr_attr / inst_to_attr are
    scalars already and pass through unchanged.
    """
    mfe = MFE(features=FEATURES, summary=["mean"])
    mfe.fit(X)
    names, values = mfe.extract()
    return dict(zip(names, values))


# ---------------------------------------------------------------------
# 4. Pipeline: loop datasets -> meta-feature table
# ---------------------------------------------------------------------

def build_metafeature_table(datasets: dict[int, str]) -> pd.DataFrame:
    rows: dict[str, dict[str, float]] = {}
    for did, name in datasets.items():
        try:
            print(f"[{did:>5}] {name}: downloading ...", end=" ")
            X = load_openml_numeric(did)
            rows[name] = extract_metafeatures(X)
            print(f"done (X shape = {X.shape})")
        except Exception as exc:  # failure isolation: one bad dataset != crash
            print(f"FAILED ({exc})")
            rows[name] = {}

    table = pd.DataFrame.from_dict(rows, orient="index")
    # stable column order
    ordered = ["nr_inst", "nr_attr", "inst_to_attr", "skewness.mean"]
    return table.reindex(columns=ordered)


if __name__ == "__main__":
    table = build_metafeature_table(DATASETS)

    pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
    print("\n=== Meta-feature table (datasets x meta-features) ===\n")
    print(table)

    table.to_csv("metafeatures.csv", index_label="dataset")
    print("\nSaved to metafeatures.csv")
