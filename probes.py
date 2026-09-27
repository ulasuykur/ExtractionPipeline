"""
probes.py - short DEC/IDEC "probe" runs whose training signals become meta-features (Phase 3).

Design (all fixed before any probe run): docs/metalearner_methodology.md, Section 11;
Design Decisions Log D12. In short:

    prepare_data(spec)            OpenML dataset -> at most 5,000 rows, 90 % training / 10 % held-out,
                                  constant columns dropped, scaled (images: one mean/sd; tabular: per column)
    pretrain(data, seed)          clustpy autoencoder  p-500-500-2000-min(10, p), exactly 2,000 steps
    run_probe(ae, data, k, ...)   clustpy's DEC/IDEC module and loss for 500 steps; measurements M1-M9 every
                                  50 steps, M10 (silhouette) and M11 (runtime) at the end, final labels (M12)
    self_test()                   our loop must give the same result as clustpy's own fit functions

Only the step loops and the measurements are our own code (decision 4). Everything that defines DEC/IDEC -
network, k-means start, soft assignments, target distribution, losses - is clustpy 0.0.2, unchanged.
Labels are never used here, except for the bookkeeping value k_true.
"""

from __future__ import annotations

import copy
import os
import time

# Must be set before CUDA is first used: makes GPU matrix products deterministic (same seed -> same result).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from clustpy.deep._data_utils import get_data_dim_from_dataloader, get_dataloader
from clustpy.deep._utils import encode_batchwise, run_initial_clustering, set_torch_seed
from clustpy.deep.dec import _DEC_Module, _dec_compression_loss_fn, _dec_predict
from clustpy.deep.neural_networks import FeedforwardAutoencoder

torch.use_deterministic_algorithms(True)

# ---------------------------------------------------------------------
# Settings (methodology §11.3 / D12). Changing any of them changes the experiment.
# ---------------------------------------------------------------------
SETTINGS = {
    "max_rows": 5000,  # rows drawn per dataset (random, fixed seed, no labels)
    "heldout_share": 0.10,  # part kept out of training, for the G ratio (candidate G1)
    "data_seed": 0,  # seed for row sample and split (the same for all training seeds)
    "hidden_layers": [
        500,
        500,
        2000,
    ],  # autoencoder p-500-500-2000-embedding (Xie et al. 2016)
    "max_embedding": 10,  # embedding size = min(10, p)
    "batch_size": 256,
    "lr_pretrain": 1e-3,  # Adam, clustpy default for pretraining
    "lr_probe": 1e-4,  # Adam, clustpy default for the clustering phase
    "pretrain_steps": 2000,
    "probe_steps": 500,
    "checkpoint_every": 50,  # measurements at steps 0, 50, ..., 500
    "alpha": 1.0,  # DEC's Student-t degrees of freedom (clustpy default)
    # loss weights (clustering, reconstruction): DEC trains only the clustering loss;
    # IDEC uses clustpy's defaults 0.1 / 1.0
    "loss_weights": {"DEC": (1.0, 0.0), "IDEC": (0.1, 1.0)},
    "used_share": 0.01,  # a cluster counts as "used" if it holds >= 1 % of the points (M8)
    "silhouette_rows": 2000,  # M10 is computed on at most this many training points
}
IMAGE_SOURCES = {"image", "image_variant"}  # `source` column of data/pool.csv


# ---------------------------------------------------------------------
# 1. Data preparation
# ---------------------------------------------------------------------


def scale(X_train: np.ndarray, X_heldout: np.ndarray, is_image: bool):
    """Standardize with statistics of the TRAINING part only (the held-out part stays unseen).

    Images: one mean and one sd over the whole array, so pixels keep their relative brightness
    (as in the ClustPy benchmark code). Tabular: per column, because columns have different units.
    """
    if is_image:
        mean, sd = X_train.mean(), X_train.std()
    else:
        mean, sd = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mean) / sd, (X_heldout - mean) / sd


def split_and_scale(X: np.ndarray, is_image: bool, rng_key: list[int]) -> dict:
    """Row cap, 90/10 split, constant columns, scaling. Deterministic for a given rng_key."""
    rng = np.random.default_rng([SETTINGS["data_seed"], *rng_key])
    n_rows = min(len(X), SETTINGS["max_rows"])
    rows = rng.permutation(len(X))[:n_rows]  # random rows, no labels involved
    n_heldout = int(round(SETTINGS["heldout_share"] * n_rows))
    heldout_rows, train_rows = rows[:n_heldout], rows[n_heldout:]
    X_train, X_heldout = X[train_rows], X[heldout_rows]
    # drop columns that are constant in the training part (their sd would be 0)
    keep = np.ptp(X_train, axis=0) > 0
    X_train, X_heldout = scale(X_train[:, keep], X_heldout[:, keep], is_image)
    return {
        "X_train": X_train.astype(np.float32),
        "X_heldout": X_heldout.astype(np.float32),
        "rows_train": train_rows,  # row numbers in the loaded data (for bookkeeping)
        "n_train": len(train_rows),
        "n_heldout": len(heldout_rows),
        "p_used": int(keep.sum()),
    }


def prepare_data(spec: dict) -> dict:
    """Load one pool dataset exactly like extractor.py does (incl. the image class subsets), then
    apply split_and_scale. spec = one row of data/pool.csv."""
    import pandas as pd
    from extractor import drop_constant_columns, load_openml_numeric, take_class_subset

    dataset_id = int(spec["dataset_id"])
    X, y = load_openml_numeric(dataset_id)
    subset_k = spec.get("subset_k")
    has_subset = subset_k is not None and pd.notna(subset_k)
    if has_subset:
        X, y, _ = take_class_subset(X, y, dataset_id, int(subset_k), int(spec["n_max"]))
    X = drop_constant_columns(X)
    is_image = spec["source"] in IMAGE_SOURCES
    rng_key = [dataset_id, int(subset_k) if has_subset else 0]
    data = split_and_scale(X, is_image, rng_key)
    data.update(
        name=spec["name"],
        source=spec["source"],
        n_loaded=len(X),
        k_true=int(y.nunique()),  # bookkeeping only, never used by the probes
    )
    return data


# ---------------------------------------------------------------------
# 2. Pretraining (shared by all k, DEC and IDEC, both protocols)
# ---------------------------------------------------------------------


def endless(loader):
    """Yield batches forever; every new pass reshuffles, exactly like a new clustpy epoch."""
    while True:
        for batch in loader:
            yield batch


def pretrain(
    X_train: np.ndarray, seed: int, device: torch.device, steps: int | None = None
):
    """Train clustpy's feedforward autoencoder with its own reconstruction loss for a fixed
    number of steps. Seeding follows clustpy's fit functions (set_torch_seed with a RandomState).
    """
    steps = SETTINGS["pretrain_steps"] if steps is None else steps
    p = X_train.shape[1]
    layers = [p, *SETTINGS["hidden_layers"], min(SETTINGS["max_embedding"], p)]
    random_state = np.random.RandomState(seed)
    set_torch_seed(random_state)  # seeds torch (weights, batch order), numpy and random
    ae = FeedforwardAutoencoder(layers=layers, random_state=random_state).to(device)
    loader = get_dataloader(X_train, SETTINGS["batch_size"], shuffle=True)
    set_torch_seed(
        random_state
    )  # clustpy's AE.fit seeds once more before training; we do the same
    optimizer = torch.optim.Adam(ae.parameters(), lr=SETTINGS["lr_pretrain"])
    loss_fn = torch.nn.MSELoss()
    batches = endless(loader)
    ae.train()
    for _ in range(steps):
        loss, _, _ = ae.loss(
            next(batches), loss_fn, device
        )  # clustpy's reconstruction loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    ae.eval()
    ae.fitted = True  # tells clustpy the network is pretrained
    return ae


# ---------------------------------------------------------------------
# 3. One probe run + measurements
# ---------------------------------------------------------------------


def entropy(probabilities: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Shannon entropy (natural log); 0 * log 0 counts as 0."""
    p = probabilities.clamp_min(1e-12)
    return -(probabilities * p.log()).sum(dim)


@torch.no_grad()
def measure(
    ae, dec_module, X_train_t, X_heldout_t, k, start_labels, loss_fn
) -> tuple[dict, np.ndarray]:
    """Measurements M1-M9 on the current state (no training, no randomness used)."""
    z_train, z_heldout = ae.encode(X_train_t), ae.encode(X_heldout_t)
    q_train = _dec_predict(
        dec_module.centers, z_train, dec_module.alpha, None
    )  # soft assignments
    q_heldout = _dec_predict(dec_module.centers, z_heldout, dec_module.alpha, None)
    labels = q_train.argmax(1)
    sizes = torch.bincount(labels, minlength=k).double()
    shares = sizes / sizes.sum()
    labels = labels.cpu().numpy()
    log_k = float(np.log(k))
    values = {
        # M1/M2: DEC's clustering loss (KL to the sharpened target), clustpy's function, on the whole part
        "M1_loss_train": _dec_compression_loss_fn(q_train).item(),
        "M2_loss_heldout": _dec_compression_loss_fn(q_heldout).item(),
        # M3: reconstruction error (mean squared error), same loss function as in training
        "M3_recon_train": loss_fn(ae.decode(z_train), X_train_t).item(),
        "M4_confidence": q_train.max(1).values.mean().item(),
        "M5_assign_entropy": (entropy(q_train, 1).mean() / log_k).item(),
        "M6_size_entropy": (entropy(shares) / log_k).item(),
        "M7_smallest_share": shares.min().item(),
        "M8_clusters_used": int((shares >= SETTINGS["used_share"]).sum().item()),
        "M9_ari_vs_start": adjusted_rand_score(start_labels, labels),
    }
    return values, labels


def run_probe(
    ae_pretrained,
    data: dict,
    k: int,
    algorithm: str,
    probe_seed: int,
    device: torch.device,
    steps: int | None = None,
) -> dict:
    """One short DEC or IDEC run with k clusters, starting from a copy of the pretrained network.

    Seeding and order of operations follow clustpy's DEC.fit / IDEC.fit exactly (checked by
    self_test): seed -> data loaders -> k-means start on the embedding -> clustering phase.
    """
    steps = SETTINGS["probe_steps"] if steps is None else steps
    every = SETTINGS["checkpoint_every"]
    clustering_weight, recon_weight = SETTINGS["loss_weights"][algorithm]
    start_time = time.perf_counter()

    random_state = np.random.RandomState(probe_seed)
    set_torch_seed(random_state)
    ae = copy.deepcopy(ae_pretrained).to(
        device
    )  # the pretrained network itself stays unchanged
    X_train, X_heldout = data["X_train"], data["X_heldout"]
    loader = get_dataloader(X_train, SETTINGS["batch_size"], shuffle=True)
    # clustpy's DEC.fit reads one batch here to check the data dimension, which advances the random
    # generator; we do the same so that both see exactly the same batch order afterwards
    get_data_dim_from_dataloader(loader)
    X_train_t = torch.from_numpy(X_train).to(device)
    X_heldout_t = torch.from_numpy(X_heldout).to(device)

    # k-means start in the embedding of the training part (clustpy's functions and defaults; the
    # embedding is computed batch by batch in fixed order, exactly as clustpy does it)
    z_train = encode_batchwise(
        get_dataloader(X_train, SETTINGS["batch_size"], shuffle=False), ae
    )
    _, start_labels, start_centers, _ = run_initial_clustering(
        z_train, k, KMeans, {}, random_state
    )

    dec_module = _DEC_Module(start_centers, SETTINGS["alpha"]).to(device)
    optimizer = torch.optim.Adam(
        list(ae.parameters()) + list(dec_module.parameters()), lr=SETTINGS["lr_probe"]
    )
    loss_fn = torch.nn.MSELoss()

    checkpoints = []
    values, labels = measure(
        ae, dec_module, X_train_t, X_heldout_t, k, start_labels, loss_fn
    )
    checkpoints.append({"step": 0, **values})
    batches = endless(loader)
    for step in range(1, steps + 1):
        # clustpy's DEC/IDEC loss for one batch: clustering loss (+ reconstruction loss for IDEC)
        loss = dec_module._loss(
            next(batches), ae, clustering_weight, recon_weight, loss_fn, device
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % every == 0 or step == steps:
            values, labels = measure(
                ae, dec_module, X_train_t, X_heldout_t, k, start_labels, loss_fn
            )
            checkpoints.append({"step": step, **values})

    # M10: silhouette of the final clusters in the embedding (on at most 2,000 training points)
    with torch.no_grad():
        z_final = ae.encode(X_train_t).cpu().numpy()
    sample_rng = np.random.default_rng(
        probe_seed
    )  # own generator: does not touch the training seeds
    rows = sample_rng.permutation(len(z_final))[: SETTINGS["silhouette_rows"]]
    n_found = len(np.unique(labels[rows]))
    silhouette = (
        silhouette_score(z_final[rows], labels[rows])
        if 1 < n_found < len(rows)
        else float("nan")
    )

    return {
        "checkpoints": checkpoints,  # M1-M9 at steps 0, 50, ..., steps
        "M10_silhouette": float(silhouette),
        "M11_runtime_s": time.perf_counter() - start_time,
        "labels": labels.astype(
            np.int16
        ),  # final clusters of the training points (for M12)
        "centers": dec_module.centers.detach().cpu().numpy(),
    }


# ---------------------------------------------------------------------
# 4. Self-test: our loops must reproduce clustpy's own fit functions
# ---------------------------------------------------------------------


def self_test(device: torch.device | None = None) -> bool:
    """Compare our pretraining and probe loops with clustpy's AE.fit, DEC.fit and IDEC.fit on a small
    artificial dataset. The step counts are chosen as whole epochs, so both sides see the same batches.
    Returns True if all comparisons agree."""
    from clustpy.deep import DEC, IDEC

    device = device or torch.device("cpu")
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 5, size=(4, 12))
    X = np.vstack(
        [c + rng.normal(0, 1, size=(250, 12)) for c in centers]
    )  # 1,000 points, 4 groups
    data = split_and_scale(X, is_image=False, rng_key=[999, 0])
    X_train = data["X_train"]
    batches_per_epoch = int(
        np.ceil(len(X_train) / SETTINGS["batch_size"])
    )  # 900 rows -> 4
    epochs, seed, k = 5, 3, 4
    ok = True

    # (a) pretraining: our 20 steps vs clustpy's AE.fit with 5 epochs
    ours = pretrain(X_train, seed, device, steps=epochs * batches_per_epoch)
    random_state = np.random.RandomState(seed)
    set_torch_seed(random_state)
    p = X_train.shape[1]
    layers = [p, *SETTINGS["hidden_layers"], min(SETTINGS["max_embedding"], p)]
    theirs = FeedforwardAutoencoder(layers=layers, random_state=random_state).to(device)
    theirs.fit(
        n_epochs=epochs,
        optimizer_params={"lr": SETTINGS["lr_pretrain"]},
        dataloader=get_dataloader(X_train, SETTINGS["batch_size"], shuffle=True),
        ssl_loss_fn=torch.nn.MSELoss(),
    )
    diff = max(
        (a - b).abs().max().item()
        for a, b in zip(ours.parameters(), theirs.parameters())
    )
    print(f"  pretraining: largest weight difference ours vs clustpy = {diff:.1e}")
    ok &= diff < 1e-5

    # (b) probe: our run_probe vs clustpy's DEC.fit / IDEC.fit, same pretrained network and seed
    for name, cls in (("DEC", DEC), ("IDEC", IDEC)):
        probe = run_probe(
            ours, data, k, name, seed, device, steps=epochs * batches_per_epoch
        )
        model = cls(
            n_clusters=k,
            batch_size=SETTINGS["batch_size"],
            clustering_epochs=epochs,
            neural_network=copy.deepcopy(ours),
            device=device,
            random_state=seed,
        )
        model.fit(X_train)
        same_labels = np.array_equal(probe["labels"], model.dec_labels_)
        center_diff = np.abs(probe["centers"] - model.dec_cluster_centers_).max()
        print(
            f"  {name}: identical final clusters = {same_labels}, "
            f"largest center difference = {center_diff:.1e}"
        )
        ok &= same_labels and center_diff < 1e-4
    print("  self-test:", "PASSED" if ok else "FAILED")
    return bool(ok)


if __name__ == "__main__":
    self_test()
