"""Signal complexity measures, one value per beat (row).

Used for RQ3: are the anomalies a model misses the *simpler* ones? Higher
values mean more complex for every measure.

    zlib     : compressed size of the 8-bit quantized beat (Serrà et al., 2020)
    spectral : fraction of energy above a cutoff frequency
    tv       : total variation, sum of |x[i+1] - x[i]|
    var      : variance (only meaningful for data that isn't z-normalized per beat)

    python src/complexity.py    # per-class summary on the ECG5000 test set
"""
import zlib

import numpy as np

# Fixed quantization range so every beat (real or synthetic) is quantized the
# same way. ECG5000 values lie in about [-7.1, 7.4]; anything outside is clipped.
QUANT_RANGE = (-8.0, 8.0)


def zlib_size(X, bits=8, value_range=QUANT_RANGE):
    """Bytes after zlib compression of each beat quantized to `bits` bits."""
    lo, hi = value_range
    levels = 2 ** bits - 1
    q = np.round((np.clip(X, lo, hi) - lo) / (hi - lo) * levels).astype(np.uint8)
    return np.array([len(zlib.compress(row.tobytes(), 9)) for row in q], dtype=float)


def spectral_high(X, cutoff=10):
    """Fraction of (mean-removed) signal energy in FFT bins >= `cutoff`."""
    power = np.abs(np.fft.rfft(X - X.mean(axis=1, keepdims=True), axis=1)) ** 2
    return power[:, cutoff:].sum(axis=1) / power.sum(axis=1)


def total_variation(X):
    return np.abs(np.diff(X, axis=1)).sum(axis=1)


def variance(X):
    return X.var(axis=1)


def compute_all(X, include_var=False):
    """Dict of complexity arrays, one value per beat."""
    out = {
        "complexity_zlib": zlib_size(X),
        "complexity_spectral": spectral_high(X),
        "complexity_tv": total_variation(X),
    }
    if include_var:
        out["complexity_var"] = variance(X)
    return out


if __name__ == "__main__":
    import data

    splits = data.load_splits()
    X, y = splits["X_test"], splits["y_test"]
    scores = compute_all(X)

    names = list(scores)
    print(f"{'class':>6} {'n':>5} " + " ".join(f"{n.removeprefix('complexity_'):>14}" for n in names))
    for c in np.unique(y):
        m = y == c
        cells = " ".join(f"{np.median(scores[n][m]):>14.3f}" for n in names)
        print(f"{c:>6} {m.sum():>5} {cells}")
    print("(medians; class 1 = normal)")

    # Do the three measures agree on which beats are complex?
    from scipy.stats import spearmanr
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            rho = spearmanr(scores[names[i]], scores[names[j]]).statistic
            print(f"spearman {names[i]} vs {names[j]}: {rho:.2f}")
