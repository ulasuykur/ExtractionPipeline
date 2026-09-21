"""
Dataset pool selection (Phase 1, Thesis Writing Notes M4).

Selects the ~100 datasets for the meta-dataset from OpenML METADATA only
(nothing is downloaded here) and writes them to data/pool.csv, which
extractor.py reads.

Pool = 83 tabular datasets (stratified by k) + 17 image datasets
       (8 originals + 9 class-subset variants, fixed n per variant).

Run once; the result is fixed by RANDOM_STATE and committed to git.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import openml
import pandas as pd

RANDOM_STATE = 0
POOL_PATH = Path("data") / "pool.csv"
POOL_SIZE = 100

# ---------------------------------------------------------------------
# 1. Candidate filters (same as the first candidate count, M4)
# ---------------------------------------------------------------------
MIN_K, MAX_K = 2, 50
MIN_N, MAX_N = 100, 100_000
MIN_FEATURES = 3  # OpenML counts the target -> at least 2 real attributes
# Width cap (user, 2026-09-21): pymfe's correlation matrix grows with p^2
# (p = 20,000 would need ~3 GB). 3,100 keeps CIFAR-10 (3,072 columns).
MAX_FEATURES = 3_100

# Synthetic OpenML families that are near-copies of other datasets
SYNTHETIC_PATTERNS = [
    r"^bng\(",
    r"^fri_c",
    r"_seed_\d+_nrows_",
]  # last: OpenML subsample copies

# Image families: chosen by hand below, so excluded from the tabular draw
IMAGE_NAME_PATTERN = r"mnist|fashion|kuzushiji|kmnist|cifar|usps|optdigits|svhn|coil|stl|alpha_digits|devnagari"

# ---------------------------------------------------------------------
# 2. Image datasets (hand-picked; widely used in deep clustering)
# ---------------------------------------------------------------------
IMAGE_ORIGINALS = {  # OpenML id -> (expected name, k), checked against metadata
    554: ("mnist_784", 10),
    40996: ("Fashion-MNIST", 10),
    41982: ("Kuzushiji-MNIST", 10),
    41082: ("USPS", 10),
    28: ("optdigits", 10),
    40927: ("CIFAR_10", 10),
    46782: ("binary_alpha_digits", 36),  # 20x16 binary images of digits/letters
    40923: ("Devnagari-Script", 46),  # 32x32 images of handwritten characters
}
# Class-subset variants: (source id, k). Classes are drawn in extractor.py
# with a fixed seed; every variant is subsampled to VARIANT_N rows,
# balanced over its classes, so that n does not reveal k.
IMAGE_VARIANTS = [(src, k) for src in (554, 40996, 41982) for k in (2, 3, 5)]
VARIANT_N = 5_000

# ---------------------------------------------------------------------
# 3. k brackets (balanced, not forced) and targets for the whole pool
# ---------------------------------------------------------------------
BRACKETS = {  # name -> (k_min, k_max), inclusive
    "A (k=2)": (2, 2),
    "B (k=3)": (3, 3),
    "C (k=4-5)": (4, 5),
    "D (k=6-9)": (6, 9),
    "E (k=10-50)": (10, 50),
}
TARGET_PER_BRACKET = POOL_SIZE // len(BRACKETS)  # 20; also the k=2 cap (20%)


def bracket_of(k: int) -> str:
    for name, (lo, hi) in BRACKETS.items():
        if lo <= k <= hi:
            return name
    raise ValueError(k)


def family_of(name: str) -> str:
    """Dataset family = first word of the name without digits, e.g.
    'volcanoes-b4' / 'volcanoes-e1' -> 'volcanoes', 'timing-attack-...-13' ->
    'timing', 'wine-quality-red' -> 'wine'. At most one dataset per family is
    kept, because family members are near-duplicates of each other."""
    first = re.split(r"[_\-\s.()]+", name.lower().strip())[0]
    return re.sub(r"\d+", "", first) or name.lower()


def normalize_name(name: str) -> str:
    """Name without case and separators, for near-duplicate detection
    (e.g. 'mfeat-fourier' / 'mfeat_fourier'). Exact copies under other names
    are caught by the size-profile check (n, p, k), related datasets by the
    family rule (family_of)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


# ---------------------------------------------------------------------
# 4. Tabular candidates
# ---------------------------------------------------------------------


def tabular_candidates(meta: pd.DataFrame) -> pd.DataFrame:
    c = meta[
        meta.NumberOfClasses.between(MIN_K, MAX_K)
        & meta.NumberOfInstances.between(MIN_N, MAX_N)
        & meta.NumberOfFeatures.between(MIN_FEATURES, MAX_FEATURES)
        & (meta.NumberOfSymbolicFeatures <= 1)  # numeric only (1 = target)
        & (meta.NumberOfMissingValues == 0)
    ].copy()
    lower = c.name.str.lower()
    c = c[~lower.str.contains("|".join(SYNTHETIC_PATTERNS), regex=True)]
    c = c[~c.name.str.lower().str.contains(IMAGE_NAME_PATTERN, regex=True)]

    # near-duplicates: newest version per normalized name, then one per
    # size profile (n, p, k)
    c["norm_name"] = c.name.map(normalize_name)
    c = c.sort_values("version").drop_duplicates("norm_name", keep="last")
    c = c.drop_duplicates(
        ["NumberOfInstances", "NumberOfFeatures", "NumberOfClasses"], keep="last"
    )
    # one dataset per family (random but fixed choice)
    c["family"] = c.name.map(family_of)
    c = c.sample(frac=1, random_state=RANDOM_STATE).drop_duplicates("family")
    c = c.sort_index()
    c["k"] = c.NumberOfClasses.astype(int)
    c["bracket"] = c.k.map(bracket_of)
    return c


def is_binarized(did: int) -> bool:
    """True if OpenML marks the dataset as a binarized copy (target column
    'binaryClass': regression target thresholded, or multi-class reduced to
    one-vs-rest). Their 2 classes do not reflect the data's structure, which
    breaks 'number of classes = number of clusters' (D4). Metadata only."""
    try:
        ds = openml.datasets.get_dataset(
            int(did),
            download_data=False,
            download_qualities=False,
            download_features_meta_data=False,
        )
        return str(ds.default_target_attribute).lower() == "binaryclass"
    except Exception as exc:  # unreadable metadata -> exclude to be safe
        print(f"WARNING: could not read target of {did} ({exc}); excluded")
        return True


def drop_binarized(c: pd.DataFrame) -> pd.DataFrame:
    """Check every k = 2 candidate's target name (a few minutes, no data
    download) and remove binarized copies."""
    k2 = c.index[c.k == 2]
    print(f"Checking target names of {len(k2)} k=2 candidates ...")
    flagged = [did for did in k2 if is_binarized(did)]
    print(f"Excluded {len(flagged)} binarized datasets (target 'binaryClass')")
    return c.drop(index=flagged)


def allocate(available: dict[str, int], targets: dict[str, int]) -> dict[str, int]:
    """Take min(target, available) per bracket; hand missing places to
    brackets that still have candidates (never to the k=2 bracket)."""
    alloc = {b: min(targets[b], available[b]) for b in targets}
    missing = sum(targets.values()) - sum(alloc.values())
    while missing > 0:
        open_b = [
            b for b in targets if alloc[b] < available[b] and not b.startswith("A")
        ]
        if not open_b:
            break
        for b in open_b:
            if missing == 0:
                break
            alloc[b] += 1
            missing -= 1
    return alloc


# ---------------------------------------------------------------------
# 5. Build the pool
# ---------------------------------------------------------------------


def build_pool() -> pd.DataFrame:
    meta = openml.datasets.list_datasets(output_format="dataframe", status="active")
    meta = meta.set_index("did", drop=False)

    # --- images ---------------------------------------------------------
    image_rows = []
    for did, (expected, k_img) in IMAGE_ORIGINALS.items():
        found = meta.loc[did, "name"] if did in meta.index else "NOT FOUND"
        if found != expected:
            print(f"WARNING: OpenML id {did} is '{found}', expected '{expected}'")
        image_rows.append(
            dict(
                dataset_id=did,
                name=found,
                source="image",
                k=k_img,
                subset_k=np.nan,
                n_max=np.nan,
                group=f"img_{did}",
            )
        )
    for i, (did, k) in enumerate(IMAGE_VARIANTS):
        base = meta.loc[did, "name"] if did in meta.index else str(did)
        image_rows.append(
            dict(
                dataset_id=did,
                name=f"{base}_k{k}",
                source="image_variant",
                k=k,
                subset_k=k,
                n_max=VARIANT_N,
                group=f"img_{did}",
            )
        )
    images = pd.DataFrame(image_rows)
    images["bracket"] = images.k.map(bracket_of)

    # --- tabular: targets = 20 per bracket minus the images in it ---------
    cand = drop_binarized(tabular_candidates(meta))
    img_per_bracket = images.bracket.value_counts()
    targets = {b: TARGET_PER_BRACKET - int(img_per_bracket.get(b, 0)) for b in BRACKETS}
    available = cand.bracket.value_counts().reindex(BRACKETS, fill_value=0)
    alloc = allocate(available.to_dict(), targets)

    print("\nBracket     candidates  images  target(tab)  drawn(tab)  total")
    rng = np.random.default_rng(RANDOM_STATE)
    tab_parts = []
    for b in BRACKETS:
        pool_b = cand[cand.bracket == b]
        idx = rng.choice(len(pool_b), size=alloc[b], replace=False)
        tab_parts.append(pool_b.iloc[np.sort(idx)])
        n_img = int(img_per_bracket.get(b, 0))
        print(
            f"{b:<12}{available[b]:>10}{n_img:>8}{targets[b]:>13}"
            f"{alloc[b]:>12}{alloc[b] + n_img:>7}"
        )

    tab = pd.concat(tab_parts)
    tabular = pd.DataFrame(
        dict(
            dataset_id=tab.did.astype(int),
            name=tab.name,
            source="tabular",
            k=tab.k,
            subset_k=np.nan,
            n_max=np.nan,
            group=[f"tab_{d}" for d in tab.did],
            bracket=tab.bracket,
        )
    )

    pool = pd.concat([tabular, images], ignore_index=True)
    print(
        f"\nTotal: {len(pool)} datasets "
        f"({(pool.source == 'tabular').sum()} tabular, "
        f"{(pool.source != 'tabular').sum()} image); "
        f"share k=2: {(pool.k == 2).mean():.0%}"
    )
    return pool


if __name__ == "__main__":
    pool = build_pool()
    POOL_PATH.parent.mkdir(exist_ok=True)
    pool.to_csv(POOL_PATH, index=False)
    print(f"Saved to {POOL_PATH}")
