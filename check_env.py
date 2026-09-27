"""
check_env.py - check that the Python environment matches the project's pinned versions
and that PyTorch + clustpy work (on the GPU if one is visible).

Run from the repo root:  python check_env.py   (local PC, or inside a Slurm job)
Exit code 0 = everything OK, 1 = at least one problem (all problems are listed at the end).
"""

import importlib.metadata as md
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

CONSTRAINTS = Path(__file__).parent / "env" / "constraints.txt"
problems = []  # every failed check adds one line here; printed at the end


def normalize(name):
    """pip treats 'Jinja2'/'jinja2' and 'typing_extensions'/'typing-extensions' as the same package."""
    return name.lower().replace("_", "-")


def read_constraints(path):
    """Return {package name: pinned version} from a constraints file (comment lines skipped)."""
    pins = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, version = line.split("==")
            pins[normalize(name)] = version
    return pins


def check_system():
    print("== 0. System")
    print("  python  :", sys.version.split()[0])
    print("  platform:", platform.platform())
    if sys.platform.startswith("linux"):
        print("  glibc   :", platform.libc_ver()[1])  # torch wheels need glibc >= 2.28


def check_versions():
    print("== 1. Installed versions vs. env/constraints.txt")
    pins = read_constraints(CONSTRAINTS)
    installed = {normalize(d.metadata["Name"]): d.version for d in md.distributions()}
    for name, wanted in sorted(pins.items()):
        have = installed.get(name)
        if have is None:
            # A few pins are only needed on Windows (e.g. colorama), so "missing" is only reported.
            print(f"  not installed (info): {name}=={wanted}")
        elif have != wanted:
            print(f"  MISMATCH: {name} installed {have}, pinned {wanted}")
            problems.append(f"version mismatch: {name}")
    print(f"  {len(pins)} pins checked")


def compatible_arch(arch_list, major, minor):
    """Return the build's code version that runs on this GPU (the closest one), or None.
    NVIDIA: code for compute capability X.y runs on any GPU X.z with z >= y
    (same major revision, same or higher minor revision)."""
    best, best_minor = None, -1
    for arch in arch_list:
        digits = "".join(ch for ch in arch.removeprefix("sm_") if ch.isdigit())
        if arch.startswith("sm_") and len(digits) >= 2:
            arch_major, arch_minor = int(digits[:-1]), int(digits[-1])
            if arch_major == major and best_minor < arch_minor <= minor:
                best, best_minor = arch, arch_minor
    return best


def check_torch():
    """Returns the torch device to use for the clustpy test (GPU if available, else CPU)."""
    import torch

    print("== 2. PyTorch")
    print(f"  torch {torch.__version__}, built for CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("  no GPU visible -> clustpy test runs on the CPU")
        problems.append("torch sees no GPU (expected only if the job/PC has none)")
        return torch.device("cpu")

    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch_list = (
        torch.cuda.get_arch_list()
    )  # GPU generations this torch build contains code for
    print(f"  GPU: {name}, compute capability sm_{major}{minor}")
    usable = compatible_arch(arch_list, major, minor)
    print(f"  torch build supports: {arch_list} -> code used on this GPU: {usable}")
    if usable is None:
        problems.append(
            f"this torch build has no code that runs on sm_{major}{minor} ({name})"
        )

    # Small calculation on the GPU; result must match the CPU up to float32 rounding.
    a = torch.randn(1000, 1000, generator=torch.Generator().manual_seed(0))
    diff = ((a.to(device) @ a.to(device)).cpu() - a @ a).abs().max().item()
    print(f"  1000x1000 matrix product, GPU vs CPU: max difference {diff:.1e}")
    if diff > 1e-2:
        problems.append("GPU result differs from CPU result")
    return device


def run_check(check, *args):
    """Run one check; if it crashes, record the crash as a problem instead of stopping,
    so that the remaining checks and the summary at the end still run."""
    try:
        return check(*args)
    except Exception as error:
        traceback.print_exc()  # full error message, useful when reading the Slurm log
        problems.append(f"{check.__name__} crashed: {type(error).__name__}: {error}")
        return None


def check_clustpy(device):
    import clustpy
    from clustpy.deep import DEC, IDEC

    print(f"== 3. clustpy {clustpy.__version__}: tiny DEC and IDEC run on {device}")
    # Artificial data: 3 well-separated groups of 200 points in 20 columns.
    rng = np.random.default_rng(0)
    centers = rng.normal(0, 10, size=(3, 20))
    X = np.vstack([c + rng.normal(0, 1, size=(200, 20)) for c in centers]).astype(
        np.float32
    )
    for algorithm in (DEC, IDEC):
        start = time.perf_counter()
        model = algorithm(
            n_clusters=3,
            pretrain_epochs=5,
            clustering_epochs=5,
            device=device,
            random_state=0,
        )
        model.fit(X)
        sizes = np.bincount(model.labels_, minlength=3).tolist()
        print(
            f"  {algorithm.__name__}: {time.perf_counter() - start:.1f} s, cluster sizes {sizes}"
        )
        if len(model.labels_) != len(X):
            problems.append(f"{algorithm.__name__} returned the wrong number of labels")


if __name__ == "__main__":
    run_check(check_system)
    run_check(check_versions)
    device = run_check(check_torch)  # None if the torch check crashed
    run_check(
        check_clustpy, device
    )  # device=None lets clustpy choose the device itself
    print("== Result:", "OK" if not problems else "PROBLEMS")
    for p in problems:
        print("  -", p)
    sys.exit(1 if problems else 0)
