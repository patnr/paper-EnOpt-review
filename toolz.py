"""Mainly used in `robust-grad.py`."""

import multiprocessing

import numpy as np
import scipy.linalg as sla


def parallelize(fun, arr,
                max_batch_size=100,
                max_CPUs=multiprocessing.cpu_count() - 1,
                leave=False,
                desc="batch",
                **kwargs):
    """Multiprocess `fun` over `arr`.

    Splits `arr` into batches sized so as to reduce the num. of interprocess communications,
    shows a progressbar (using `p_map` which also provides `tqdm`),
    and takes care of concatenating the output (must be `ndarray`) over axis 0.

    Use `max_CPUs = 1` to disable multiprocessing.
    """
    # Number of batches (separate jobs to be dispatched)
    nBatch = len(arr) / max_batch_size  # As few (⇒large) batches as possible (minimize coms overhead)
    nBatch = max(nBatch, max_CPUs)      # but ensure enough batches to use all cores
    nBatch = min(nBatch, len(arr))      # yet don't have more batches than jobs
    batches = np.array_split(arr, nBatch)

    if max_CPUs <= 1:
        from tqdm.auto import tqdm
        # outputs = tqdm(map(fun, batches), desc=desc, total=len(batches), leave=leave, **kwargs)
        outputs = []
        for b in tqdm(batches):
            outputs.append(fun(b))
    else:
        from p_tqdm import p_map
        # Make sure np uses only 1 core. Our problem is embarrasingly parallelzable,
        # so we are more efficient manually instigating multiprocessing.
        import threadpoolctl
        threadpoolctl.threadpool_limits(1)

        outputs = p_map(fun, batches, num_cpus=max_CPUs, desc=desc, leave=leave, **kwargs)

    outputs = np.concatenate(list(outputs), axis=0)
    return outputs

def cntr(xx):
    return xx - xx.mean(0)

def rinv(A, reg, tikh=True, nMax=None):
    """Reproduces `sla.pinv(..., rtol=reg)` for `tikh=False`."""
    # Decompose
    U, s, VT = sla.svd(A, full_matrices=False)

    # "Relativize" the regularisation param
    reg = reg * s[0]

    # Compute inverse (regularized or truncated)
    if tikh:
        s1 = s / (s**2 + reg**2)
    else:
        s0 = s >= reg
        s1 = np.zeros_like(s)
        s1[s0] = 1/s[s0]

    if nMax:
        s1[nMax:] = 0

    # Re-compose
    return (VT.T * s1) @ U.T


def sort_legend(ax, order):
    """Re-order the legend of a plot `ax`.

    The `order` (list) need not match exactly, it's only tried by `__contains__`.
    """
    lines0, labels0 = ax.get_legend_handles_labels()
    lines1, labels1 = [], []
    for tag in order:
        for line, label in zip(lines0, labels0):
            if tag in label:
                lines1.append(line)
                lines0.remove(line)
                labels1.append(label)
                labels0.remove(label)
    if labels0:
        print(f"Warning: could not sort the legend items for: {labels0}")
    lines1.extend(lines0)
    labels1.extend(labels0)
    ax.legend(lines1, labels1)
